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
const breakdownEl = document.getElementById("breakdown");
const occupancyEl = document.getElementById("occupancy");
const logListEl = document.getElementById("log-list");

const readouts = {
  in: { valueEl: document.getElementById("count-in"), cardEl: document.getElementById("card-in"), prev: 0 },
  out: { valueEl: document.getElementById("count-out"), cardEl: document.getElementById("card-out"), prev: 0 },
  total: { valueEl: document.getElementById("count-total"), cardEl: document.getElementById("card-total"), prev: 0 },
};

let lastEventTimestamp = null;
const MAX_LOG_ROWS = 40;

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

function updateBreakdown(countByLabel) {
  const entries = Object.entries(countByLabel || {});
  if (entries.length === 0) {
    breakdownEl.innerHTML = "";
    return;
  }
  breakdownEl.innerHTML = entries
    .map(([label, counts]) => {
      const total = (counts.in || 0) + (counts.out || 0);
      return `<span class="breakdown-chip">${label} <strong>${pad4(total)}</strong></span>`;
    })
    .join("");
}

function updateOccupancy(estimate) {
  if (!estimate || !estimate.total) {
    occupancyEl.innerHTML = "";
    return;
  }
  const parts = Object.entries(estimate.by_label || {})
    .filter(([, n]) => n > 0)
    .map(([label, n]) => `${label} ~${n}`)
    .join(" &middot; ");
  occupancyEl.innerHTML =
    `<span class="occupancy-total">&asymp; ${Math.round(estimate.total)} people estimated</span>` +
    (parts ? `<span class="occupancy-breakdown">${parts}</span>` : "");
}

function formatClockTime(unixSeconds) {
  const d = new Date(unixSeconds * 1000);
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  const ss = String(d.getSeconds()).padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
}

function prependLogRow(ev) {
  const empty = logListEl.querySelector(".log-empty");
  if (empty) empty.remove();

  const li = document.createElement("li");
  li.className = "log-row--new";
  const dirGlyph = ev.direction === "in" ? "&#8595;" : "&#8593;";
  const dirClass = ev.direction === "in" ? "log-dir--in" : "log-dir--out";
  li.innerHTML = `
    <span class="log-time">${formatClockTime(ev.timestamp)}</span>
    <span class="log-dir ${dirClass}">${dirGlyph}</span>
    <span class="log-label">${ev.label} #${ev.track_id}</span>
  `;
  logListEl.prepend(li);

  while (logListEl.children.length > MAX_LOG_ROWS) {
    logListEl.removeChild(logListEl.lastChild);
  }
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
    updateBreakdown(data.count_by_label);
    updateOccupancy(data.estimated_occupancy);
    setStatus(data.camera_connected);
    if (typeof data.fps === "number") {
      fpsEl.textContent = `${data.fps.toFixed(1)} fps`;
    }
    if (data.latest_event && data.latest_event.timestamp !== lastEventTimestamp) {
      lastEventTimestamp = data.latest_event.timestamp;
      prependLogRow(data.latest_event);
    }
  };

  ws.onclose = () => {
    setStatus(false);
    setTimeout(connectWebSocket, 2000); // auto-reconnect
  };

  ws.onerror = () => ws.close();
}

connectWebSocket();
