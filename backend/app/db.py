"""
Optional Postgres persistence for line-crossing events + occupancy.

Entirely optional: if DATABASE_URL isn't set, every function here is a
no-op and the rest of the app runs exactly as before (in-memory counts
only, nothing persisted). Set DATABASE_URL to turn it on.

IMPORTANT: DATABASE_URL is read lazily, inside functions, NOT at module
import time. main.py calls load_dotenv() before it imports/uses this
module in normal operation, but relying on import order is fragile —
reading the env var eagerly at import time previously caused a real bug
where DATABASE_URL appeared unset because this module got imported
before load_dotenv() ran. Reading it lazily on each call sidesteps that
class of bug entirely.

Uses plain psycopg2 (sync) rather than an async driver: the processing
thread that logs crossing events is already a raw background thread,
not asyncio, so a blocking insert is fine there. The one place this is
called from an async context (the /events HTTP endpoint) wraps it in
starlette's run_in_threadpool.
"""

import os
from contextlib import contextmanager

import psycopg2
import psycopg2.extras

_SCHEMA = """
CREATE TABLE IF NOT EXISTS crossing_events (
    id BIGSERIAL PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL DEFAULT now(),
    direction TEXT NOT NULL,
    label TEXT NOT NULL,
    occupancy_count INTEGER
);
"""


def _database_url():
    return os.getenv("DATABASE_URL", "").strip()


def enabled():
    return bool(_database_url())


@contextmanager
def _connect():
    conn = psycopg2.connect(_database_url())
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    """Create the table if it doesn't exist yet. Safe to call every boot."""
    if not enabled():
        return
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(_SCHEMA)
        conn.commit()


def log_event(direction, label, occupancy_count):
    """Insert one crossing event. Called from the (sync) processor thread."""
    if not enabled():
        return
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO crossing_events (direction, label, occupancy_count) "
                "VALUES (%s, %s, %s)",
                (direction, label, occupancy_count),
            )
        conn.commit()


def fetch_recent_events(limit=50):
    """Most recent crossing events, newest first. Returns [] if DB disabled."""
    if not enabled():
        return []
    with _connect() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, ts, direction, label, occupancy_count "
                "FROM crossing_events ORDER BY ts DESC LIMIT %s",
                (limit,),
            )
            rows = cur.fetchall()
    # ts is a datetime, not JSON-serializable as-is; isoformat it here so
    # callers (main.py's JSONResponse) don't need to know about this detail.
    return [
        {
            "id": r["id"],
            "ts": r["ts"].isoformat(),
            "direction": r["direction"],
            "label": r["label"],
            "occupancy_count": r["occupancy_count"],
        }
        for r in rows
    ]


def fetch_occupancy_summary():
    """
    Aggregate stats across all logged crossings: total events and average
    occupancy_count (ignoring nulls). Returns None if DB disabled or empty.
    """
    if not enabled():
        return None
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*), AVG(occupancy_count) FROM crossing_events "
                "WHERE occupancy_count IS NOT NULL"
            )
            total, avg = cur.fetchone()
    if total == 0:
        return {"total_events_with_occupancy": 0, "avg_occupancy": None}
    # psycopg2 returns AVG(...) as a Decimal, which json.dumps can't
    # serialize (raises TypeError) — this bit us in testing. Cast to
    # float explicitly rather than passing the Decimal straight through.
    return {
        "total_events_with_occupancy": total,
        "avg_occupancy": round(float(avg), 2) if avg is not None else None,
    }
