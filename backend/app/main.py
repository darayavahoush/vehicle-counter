"""
FastAPI backend for live vehicle counting from CCTV.

Design for efficiency: detection/tracking runs ONCE in a single
background thread regardless of how many browser tabs are watching.
Every /video_feed client and /ws/count client just reads the latest
already-computed frame/counts from shared state. This means CPU cost
is constant no matter how many viewers connect.
"""

import asyncio
import json
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from threading import Lock

import cv2
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.websockets import WebSocketState

from app.capture import LatestFrameReader
from app.detector import BackgroundSubtractionDetector, YoloOnnxDetector
from app.tracker import LineCrossingCounter

load_dotenv()

# ---- Config (env-overridable) ----------------------------------------
RTSP_URL = os.getenv("RTSP_URL", "0")  # "0" = default webcam for local testing
PROCESS_WIDTH = int(os.getenv("PROCESS_WIDTH", "640"))
DETECT_EVERY_N_FRAMES = int(os.getenv("DETECT_EVERY_N_FRAMES", "2"))
OUTPUT_FPS = float(os.getenv("OUTPUT_FPS", "15"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "70"))

# Detector selection. "yolo" labels vehicle type (Car/Truck/Bus/Motorbike)
# via a YOLOv8n ONNX model run through cv2.dnn — no PyTorch needed at
# runtime. Falls back to "bg_subtraction" automatically if the model file
# isn't found, so a missing model never crashes the service.
DETECTOR_BACKEND = os.getenv("DETECTOR_BACKEND", "yolo")
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "models/yolov8n.onnx")
YOLO_INPUT_SIZE = int(os.getenv("YOLO_INPUT_SIZE", "320"))
YOLO_CONF_THRESHOLD = float(os.getenv("YOLO_CONF_THRESHOLD", "0.4"))
YOLO_NMS_THRESHOLD = float(os.getenv("YOLO_NMS_THRESHOLD", "0.45"))

# Counting line as fractions of the processed frame (0..1), so it scales
# with PROCESS_WIDTH automatically. Default is a VERTICAL line at
# mid-width — correct for a sideview camera where vehicles travel
# left<->right across the frame. (A horizontal line, as in earlier
# top-down setups, would rarely be crossed by side-view traffic.)
LINE_X1 = float(os.getenv("LINE_X1", "0.5"))
LINE_Y1 = float(os.getenv("LINE_Y1", "0.1"))
LINE_X2 = float(os.getenv("LINE_X2", "0.5"))
LINE_Y2 = float(os.getenv("LINE_Y2", "0.9"))

# How many recent crossing events to keep in memory for the /events log.
# In-memory (not a DB) is deliberate for now: it's zero-setup and enough
# to show "vehicles are being logged" in the demo. On Render's free tier
# this resets on redeploy/restart — swap in a real DB if events need to
# survive that.
EVENT_LOG_SIZE = int(os.getenv("EVENT_LOG_SIZE", "200"))

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")

# Webcam index convenience: env RTSP_URL="0" -> int 0 for cv2.VideoCapture
_capture_source = int(RTSP_URL) if RTSP_URL.strip().isdigit() else RTSP_URL


def _build_detector():
    if DETECTOR_BACKEND == "yolo":
        if os.path.isfile(YOLO_MODEL_PATH):
            print(f"[detector] loading YOLOv8n ONNX from {YOLO_MODEL_PATH}")
            return YoloOnnxDetector(
                YOLO_MODEL_PATH,
                input_size=YOLO_INPUT_SIZE,
                conf_threshold=YOLO_CONF_THRESHOLD,
                nms_threshold=YOLO_NMS_THRESHOLD,
            )
        print(
            f"[detector] DETECTOR_BACKEND=yolo but '{YOLO_MODEL_PATH}' not found — "
            "falling back to background subtraction (no vehicle-type labels)."
        )
    return BackgroundSubtractionDetector()


# ---- Shared state -------------------------------------------------------
class SharedState:
    def __init__(self):
        self.lock = Lock()
        self.jpeg_bytes = None
        self.count_in = 0
        self.count_out = 0
        self.count_by_label = {}
        self.connected = False
        self.fps = 0.0
        self.events = deque(maxlen=EVENT_LOG_SIZE)


state = SharedState()


class Processor:
    """Runs in a background thread: capture -> detect (every N frames) ->
    track -> draw overlay -> encode -> publish to shared state."""

    def __init__(self):
        self.reader = LatestFrameReader(_capture_source).start()
        self.detector = _build_detector()
        self.counter = None  # built once we know frame size
        self._frame_idx = 0
        self._last_detections = []
        self._running = False

    def _build_line(self, w, h):
        line = ((LINE_X1 * w, LINE_Y1 * h), (LINE_X2 * w, LINE_Y2 * h))
        self.counter = LineCrossingCounter(line=line)

    def _draw_overlay(self, frame, tracks):
        h, w = frame.shape[:2]
        (x1, y1), (x2, y2) = self.counter.line
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 255), 2)

        for t in tracks:
            x, y, bw, bh = t.box
            cv2.rectangle(frame, (int(x), int(y)), (int(x + bw), int(y + bh)), (60, 220, 60), 2)
            cv2.putText(
                frame, f"{t.label} #{t.id}", (int(x), int(y) - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 220, 60), 1, cv2.LINE_AA,
            )

        label = f"IN: {self.counter.count_in}   OUT: {self.counter.count_out}   TOTAL: {self.counter.total}"
        cv2.rectangle(frame, (0, 0), (w, 30), (0, 0, 0), -1)
        cv2.putText(frame, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    def run(self):
        self._running = True
        frame_interval = 1.0 / OUTPUT_FPS
        last_emit = 0.0

        while self._running:
            frame = self.reader.read()
            state.connected = self.reader.connected

            if frame is None:
                time.sleep(0.05)
                continue

            # Downscale for cheap processing.
            h, w = frame.shape[:2]
            scale = PROCESS_WIDTH / float(w)
            frame = cv2.resize(frame, (PROCESS_WIDTH, int(h * scale)))

            if self.counter is None:
                self._build_line(*frame.shape[1::-1])

            self._frame_idx += 1
            if self._frame_idx % DETECT_EVERY_N_FRAMES == 0:
                self._last_detections = self.detector.detect(frame)

            tracks, events = self.counter.update(self._last_detections)
            frame = self._draw_overlay(frame, tracks)

            now = time.time()
            if now - last_emit >= frame_interval:
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if ok:
                    with state.lock:
                        state.jpeg_bytes = buf.tobytes()
                        state.count_in = self.counter.count_in
                        state.count_out = self.counter.count_out
                        state.count_by_label = dict(self.counter.count_by_label)
                        state.fps = 1.0 / (now - last_emit) if last_emit else 0.0
                        for ev in events:
                            state.events.appendleft(ev)
                last_emit = now
            elif events:
                # Even if we're not due to publish a frame, don't drop a
                # crossing event — the log should never miss a vehicle.
                with state.lock:
                    state.count_in = self.counter.count_in
                    state.count_out = self.counter.count_out
                    state.count_by_label = dict(self.counter.count_by_label)
                    for ev in events:
                        state.events.appendleft(ev)

    def stop(self):
        self._running = False
        self.reader.stop()


processor = Processor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    import threading
    t = threading.Thread(target=processor.run, daemon=True)
    t.start()
    yield
    processor.stop()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _mjpeg_generator():
    boundary = b"--frame"
    while True:
        with state.lock:
            jpeg = state.jpeg_bytes
        if jpeg is not None:
            yield (
                boundary + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                + jpeg + b"\r\n"
            )
        time.sleep(1.0 / OUTPUT_FPS)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/stats")
def stats():
    with state.lock:
        return JSONResponse({
            "count_in": state.count_in,
            "count_out": state.count_out,
            "total": state.count_in + state.count_out,
            "count_by_label": state.count_by_label,
            "camera_connected": state.connected,
            "fps": round(state.fps, 1),
        })


@app.get("/events")
def events(limit: int = 50):
    """Most recent crossing events first. Each: direction, label,
    track_id, confidence, timestamp (unix seconds)."""
    with state.lock:
        return JSONResponse(list(state.events)[:limit])


@app.websocket("/ws/count")
async def ws_count(websocket: WebSocket):
    await websocket.accept()
    last_sent = None
    last_event_count = 0
    try:
        while True:
            with state.lock:
                latest_event = state.events[0] if state.events else None
                payload = {
                    "count_in": state.count_in,
                    "count_out": state.count_out,
                    "total": state.count_in + state.count_out,
                    "count_by_label": state.count_by_label,
                    "camera_connected": state.connected,
                    "fps": round(state.fps, 1),
                    "latest_event": latest_event,
                }
            if payload != last_sent:
                await websocket.send_text(json.dumps(payload))
                last_sent = payload
            await asyncio.sleep(0.3)
    except WebSocketDisconnect:
        pass
    finally:
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close()
