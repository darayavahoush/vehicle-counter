// No framework needed: the backend already draws boxes + overlay onto the
// MJPEG stream, so the frontend's only jobs are (1) point an <img> at the
// stream and (2) keep the count numbers in sync via a small WebSocket.

const BACKEND_HTTP = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";
const BACKEND_WS = BACKEND_HTTP.replace(/^http/, "ws");

const streamEl = document.getElementById("stream");
const statusEl = document.getElementById("status");
const inEl = document.getElementById("count-in");
const outEl = document.getElementById("count-out");
const totalEl = document.getElementById("count-total");

function setStatus(connected) {
  statusEl.textContent = connected ? "camera online" : "camera offline";
  statusEl.className = `status ${connected ? "status--online" : "status--offline"}`;
}

// MJPEG stream: just point the <img> at the endpoint. The browser handles
// the multipart/x-mixed-replace stream natively — no JS decoding needed.
streamEl.src = `${BACKEND_HTTP}/video_feed`;

function connectWebSocket() {
  const ws = new WebSocket(`${BACKEND_WS}/ws/count`);

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    inEl.textContent = data.count_in;
    outEl.textContent = data.count_out;
    totalEl.textContent = data.total;
    setStatus(data.camera_connected);
  };

  ws.onclose = () => {
    setStatus(false);
    setTimeout(connectWebSocket, 2000); // auto-reconnect
  };

  ws.onerror = () => ws.close();
}

connectWebSocket();
