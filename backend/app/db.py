"""
Optional event history via Postgres. Entirely opt-in: if DATABASE_URL
isn't set, every function here is a no-op and the rest of the app
behaves exactly as it did before this file existed.

Synchronous on purpose (psycopg2, not asyncpg): the frame-processing
loop in main.py's Processor is a plain threading.Thread, not asyncio,
so a sync driver is the simpler fit — no event-loop hand-off needed to
log an event from that thread. The two read endpoints (/events,
/stats/summary) are async FastAPI routes, so they call into this module
via starlette's run_in_threadpool to avoid blocking the event loop.

Recommended host: Neon or Supabase (permanent free tier) rather than
Render's own Postgres, whose free tier expires 30 days after creation
and is then deleted — fine for a demo, not for anything you want to
keep.
"""

import os

import psycopg2
from psycopg2 import pool as pg_pool
from psycopg2.extras import RealDictCursor

DATABASE_URL = None  # set by init(), not at import time — main.py imports
                     # this module before load_dotenv() runs, so reading
                     # os.environ here at module load would always miss it.

_pool = None


def init():
    """Call once at startup, after load_dotenv(). Safe to call with no
    DATABASE_URL set."""
    global _pool, DATABASE_URL
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        print("[db] DATABASE_URL not set — event history disabled (counting still works).")
        return

    _pool = pg_pool.SimpleConnectionPool(1, 5, DATABASE_URL)
    conn = _pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS vehicle_events (
                    id BIGSERIAL PRIMARY KEY,
                    ts TIMESTAMPTZ NOT NULL DEFAULT now(),
                    line TEXT NOT NULL DEFAULT 'main',
                    direction TEXT NOT NULL,
                    label TEXT NOT NULL,
                    occupancy_estimate INTEGER,
                    occupancy_confidence TEXT
                )
            """)
            # Existing deployments from before multi-line support won't
            # have this column yet — add it without touching their data.
            cur.execute("ALTER TABLE vehicle_events ADD COLUMN IF NOT EXISTS line TEXT NOT NULL DEFAULT 'main'")
        conn.commit()
        print("[db] connected, vehicle_events table ready.")
    finally:
        _pool.putconn(conn)


def close():
    global _pool
    if _pool:
        _pool.closeall()
        _pool = None


def log_event(line, direction, label, occupancy_estimate=None):
    if _pool is None:
        return
    conn = _pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO vehicle_events (line, direction, label, occupancy_estimate, occupancy_confidence)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (line, direction, label, occupancy_estimate, "low" if occupancy_estimate is not None else None),
            )
        conn.commit()
    except Exception as e:
        print(f"[db] failed to log event: {e}")
        conn.rollback()
    finally:
        _pool.putconn(conn)


def recent_events(limit=50, line=None):
    if _pool is None:
        return []
    conn = _pool.getconn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if line:
                cur.execute(
                    """
                    SELECT id, ts, line, direction, label, occupancy_estimate
                    FROM vehicle_events WHERE line = %s ORDER BY ts DESC LIMIT %s
                    """,
                    (line, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT id, ts, line, direction, label, occupancy_estimate
                    FROM vehicle_events ORDER BY ts DESC LIMIT %s
                    """,
                    (limit,),
                )
            rows = [dict(r) for r in cur.fetchall()]
        for r in rows:
            r["ts"] = r["ts"].isoformat()
        return rows
    finally:
        _pool.putconn(conn)


def summary():
    if _pool is None:
        return []
    conn = _pool.getconn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT line, label, direction, COUNT(*) AS n,
                       ROUND(AVG(occupancy_estimate)::numeric, 2) AS avg_occupancy
                FROM vehicle_events
                GROUP BY line, label, direction
                ORDER BY line, label, direction
            """)
            rows = [dict(r) for r in cur.fetchall()]
        for r in rows:
            # Postgres NUMERIC comes back as Decimal, which json.dumps
            # can't serialize; None stays None (no occupancy data yet).
            r["avg_occupancy"] = float(r["avg_occupancy"]) if r["avg_occupancy"] is not None else None
        return rows
    finally:
        _pool.putconn(conn)
