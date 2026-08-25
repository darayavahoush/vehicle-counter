// The backend already draws boxes + the tripwire line onto the MJPEG stream,
// so the frontend's only real jobs are: point an <img> at the stream, keep
// the count readouts in sync via a small WebSocket, and run the on-screen
// clock overlay. No framework needed for that.

const BACKEND_HTTP = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";
const BACKEND_WS = BACKEND_HTTP.replace(/^http/, "ws");

const streamEl = document.getElementById("stream");
const statusEl = document.getElementById("status");
const statusLabelEl = document.getElementById("status-label");
const clockEl = document.getElementById("clock");
const fpsEl = document.getElementById("fps-readout");

const readouts = {
  in: { valueEl: document.getElementById("count-in"), cardEl: document.getElementById("card-in"), prev: 0 },
  out: { valueEl: document.getElementById("count-out"), cardEl: document.getElementById("card-out"), prev: 0 },
  total: { valueEl: document.getElementById("count-total"), cardEl: document.getElementById("card-total"), prev: 0 },
};

const chipsEl = document.getElementById("label-chips");
const eventsListEl = document.getElementById("events-list");
const eventsHintEl = document.getElementById("events-hint");

function pad4(n) {
  return String(n).padStart(4, "0");
}

function setStatus(connected) {
  statusLabelEl.textContent = connected ? "live" : "offline";
  statusEl.className = `status ${connected ? "status--online" : "status--offline"}`;
}

function updateReadout(key, value) {
  const r = readouts[key];
  r.valueEl.textContent = pad4(value);
  if (value !== r.prev) {
    r.cardEl.classList.add("readout--flash");
    setTimeout(() => r.cardEl.classList.remove("readout--flash"), 500);
  }
  r.prev = value;
}

function tickClock() {
  const now = new Date();
  const hh = String(now.getHours()).padStart(2, "0");
  const mm = String(now.getMinutes()).padStart(2, "0");
  const ss = String(now.getSeconds()).padStart(2, "0");
  clockEl.textContent = `${hh}:${mm}:${ss}`;
}
tickClock();
setInterval(tickClock, 1000);

// ---- Label breakdown + occupancy chips -----------------------------------
// Rebuilt from each /ws/count payload (already includes count_by_label and
// occupancy_now — no extra request needed for this part).
function updateChips(countByLabel, occupancyNow) {
  const parts = [];
  for (const [label, count] of Object.entries(countByLabel || {}).sort()) {
    parts.push(`<span class="chip"><span>${label}</span><span class="chip-value">${count}</span></span>`);
  }
  parts.push(
    `<span class="chip chip--occupancy"><span>occupancy</span><span class="chip-value">${occupancyNow ?? 0}</span></span>`
  );
  chipsEl.innerHTML = parts.join("");
}

// ---- Recent crossings panel -----------------------------------------------
// Polls the backend's /events endpoint (backed by Postgres if DATABASE_URL
// is configured there; returns an empty list otherwise, in which case we
// just show a note instead of an empty-looking panel).
function formatEventTime(isoTs) {
  const d = new Date(isoTs);
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  const ss = String(d.getSeconds()).padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
}

function renderEvents(events, dbEnabled) {
  if (!dbEnabled) {
    eventsHintEl.textContent = "persistence off";
    eventsListEl.innerHTML = `<li class="events-list__empty">Set DATABASE_URL on the backend to log crossings here.</li>`;
    return;
  }
  eventsHintEl.textContent = "";
  if (!events.length) {
    eventsListEl.innerHTML = `<li class="events-list__empty">No crossings logged yet.</li>`;
    return;
  }
  eventsListEl.innerHTML = events
    .map((e) => {
      const arrow = e.direction === "in" ? "&#8595;" : "&#8593;";
      const dirClass = e.direction === "in" ? "event-row__dir--in" : "event-row__dir--out";
      const occ = e.occupancy_count != null ? `occ ${e.occupancy_count}` : "";
      return `
        <li class="event-row">
          <span class="event-row__time">${formatEventTime(e.ts)}</span>
          <span class="event-row__dir ${dirClass}">${arrow}</span>
          <span class="event-row__label">${e.label}</span>
          <span class="event-row__occ">${occ}</span>
        </li>`;
    })
    .join("");
}

async function pollEvents() {
  try {
    const res = await fetch(`${BACKEND_HTTP}/events?limit=8`);
    const data = await res.json();
    renderEvents(data.events || [], data.db_enabled);
  } catch {
    eventsHintEl.textContent = "unreachable";
  }
}
pollEvents();
setInterval(pollEvents, 5000);

// MJPEG stream: point the <img> at the endpoint. The browser decodes the
// multipart/x-mixed-replace stream natively — no JS needed on this side.
streamEl.src = `${BACKEND_HTTP}/video_feed`;

function connectWebSocket() {
  const ws = new WebSocket(`${BACKEND_WS}/ws/count`);

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    updateReadout("in", data.count_in);
    updateReadout("out", data.count_out);
    updateReadout("total", data.total);
    setStatus(data.camera_connected);
    updateChips(data.count_by_label, data.occupancy_now);
    if (typeof data.fps === "number") {
      fpsEl.textContent = `${data.fps.toFixed(1)} fps`;
    }
  };

  ws.onclose = () => {
    setStatus(false);
    setTimeout(connectWebSocket, 2000); // auto-reconnect
  };

  ws.onerror = () => ws.close();
}

connectWebSocket();
