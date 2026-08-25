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
