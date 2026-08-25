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
from contextlib import asynccontextmanager
from threading import Lock

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.websockets import WebSocketState

from starlette.concurrency import run_in_threadpool

from app import db
from app.capture import LatestFrameReader
from app.detector import build_detector
from app.occupancy import OccupancyEstimator, build_roi_from_line
from app.tracker import LineCrossingCounter

load_dotenv()

# ---- Config (env-overridable) ----------------------------------------
RTSP_URL = os.getenv("RTSP_URL", "0")  # "0" = default webcam for local testing
PROCESS_WIDTH = int(os.getenv("PROCESS_WIDTH", "640"))
DETECT_EVERY_N_FRAMES = int(os.getenv("DETECT_EVERY_N_FRAMES", "2"))
OUTPUT_FPS = float(os.getenv("OUTPUT_FPS", "15"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "70"))

# "yolo" gives labeled boxes (car/truck/bus/motorcycle) via a YOLOv8n
# ONNX model through cv2.dnn — no PyTorch needed on the device. "bgsub"
# is the older no-label classical-CV fallback if you don't have (or
# don't want to run) the .onnx file, e.g. on a very constrained device.
DETECTOR_BACKEND = os.getenv("DETECTOR_BACKEND", "yolo")
YOLO_ONNX_PATH = os.getenv("YOLO_ONNX_PATH", "models/yolov8n.onnx")
YOLO_INPUT_SIZE = int(os.getenv("YOLO_INPUT_SIZE", "320"))
YOLO_CONF_THRESHOLD = float(os.getenv("YOLO_CONF_THRESHOLD", "0.35"))
YOLO_NMS_THRESHOLD = float(os.getenv("YOLO_NMS_THRESHOLD", "0.45"))

# Counting line as fractions of the processed frame (0..1), so it scales
# with PROCESS_WIDTH automatically. Default here is a VERTICAL line
# through the middle — correct for a sideview camera, where vehicles
# travel left<->right across the frame rather than toward/away from it.
# (For a top-down/angled camera, flip these back to a horizontal line,
# e.g. 0.05,0.55 -> 0.95,0.55.)
LINE_X1 = float(os.getenv("LINE_X1", "0.5"))
LINE_Y1 = float(os.getenv("LINE_Y1", "0.05"))
LINE_X2 = float(os.getenv("LINE_X2", "0.5"))
LINE_Y2 = float(os.getenv("LINE_Y2", "0.95"))

# Reuses the same YOLO forward pass (every box's class scores already
# include "person") to also spot people near the windshield/cabin area
# as a vehicle crosses the line — zero extra inference cost. Only has
# an effect with DETECTOR_BACKEND=yolo; bgsub has no notion of labels.
DETECT_PERSON = os.getenv("DETECT_PERSON", "true").lower() == "true"
# How far the occupancy "windshield" ROI extends past the counting line,
# as a fraction of the frame's shorter dimension.
OCCUPANCY_ROI_HALF_WIDTH_FRAC = float(os.getenv("OCCUPANCY_ROI_HALF_WIDTH_FRAC", "0.12"))

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")

# Webcam index convenience: env RTSP_URL="0" -> int 0 for cv2.VideoCapture
_capture_source = int(RTSP_URL) if RTSP_URL.strip().isdigit() else RTSP_URL


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
        # Live estimate of people currently near the windshield ROI —
        # not a running total, just "how many right now" (see occupancy.py).
        self.occupancy_now = 0


state = SharedState()


class Processor:
    """Runs in a background thread: capture -> detect (every N frames) ->
    track -> draw overlay -> encode -> publish to shared state."""

    def __init__(self):
        self.reader = LatestFrameReader(_capture_source).start()
        self.detector = self._build_detector()
        self.counter = None  # built once we know frame size
        self.occupancy = None  # built alongside counter, same frame size
        self._frame_idx = 0
        self._last_boxes = []
        self._running = False

    def _build_detector(self):
        if DETECTOR_BACKEND == "yolo":
            try:
                return build_detector(
                    "yolo",
                    onnx_path=YOLO_ONNX_PATH,
                    input_size=YOLO_INPUT_SIZE,
                    conf_threshold=YOLO_CONF_THRESHOLD,
                    nms_threshold=YOLO_NMS_THRESHOLD,
                    detect_person=DETECT_PERSON,
                )
            except Exception as e:
                # Most common cause: the .onnx file hasn't been exported/
                # copied into place yet (see scripts/export_yolo_onnx.py).
                # Fall back rather than crashing the whole backend, since
                # counting-without-labels still beats not running at all.
                print(
                    f"[detector] Couldn't load YOLO ONNX model at "
                    f"'{YOLO_ONNX_PATH}' ({e}); falling back to "
                    f"background-subtraction (no vehicle labels)."
                )
        return build_detector("bgsub")

    def _build_line(self, w, h):
        line = ((LINE_X1 * w, LINE_Y1 * h), (LINE_X2 * w, LINE_Y2 * h))
        self.counter = LineCrossingCounter(line=line)
        roi = build_roi_from_line(line, w, h, half_width_frac=OCCUPANCY_ROI_HALF_WIDTH_FRAC)
        self.occupancy = OccupancyEstimator(roi=roi)

    def _draw_overlay(self, frame, tracks, person_detections, occupancy_now):
        h, w = frame.shape[:2]
        (x1, y1), (x2, y2) = self.counter.line
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 255), 2)

        if self.occupancy is not None:
            rx1, ry1, rx2, ry2 = self.occupancy.roi
            cv2.rectangle(
                frame, (int(rx1), int(ry1)), (int(rx2), int(ry2)), (180, 120, 255), 1, cv2.LINE_AA,
            )

        for t in tracks:
            x, y, bw, bh = t.box
            cv2.rectangle(frame, (int(x), int(y)), (int(x + bw), int(y + bh)), (60, 220, 60), 2)
            cv2.putText(
                frame, f"#{t.id} {t.label}", (int(x), int(y) - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 220, 60), 1, cv2.LINE_AA,
            )

        for d in person_detections:
            x, y, bw, bh = d[0], d[1], d[2], d[3]
            cv2.rectangle(frame, (int(x), int(y)), (int(x + bw), int(y + bh)), (255, 160, 60), 1)

        breakdown = "  ".join(f"{k}:{v}" for k, v in sorted(self.counter.count_by_label.items()))
        label = f"IN: {self.counter.count_in}   OUT: {self.counter.count_out}   TOTAL: {self.counter.total}"
        if breakdown:
            label += f"   ({breakdown})"
        label += f"   OCC: {occupancy_now}"
        cv2.rectangle(frame, (0, 0), (w, 30), (0, 0, 0), -1)
        cv2.putText(frame, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
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
                self._last_boxes = self.detector.detect(frame)

            # Same detector call above already scored "person" alongside
            # vehicle classes when DETECT_PERSON is on — split them out
            # here so the tracker/counter only ever sees vehicles (a
            # pedestrian must never be counted as a vehicle crossing).
            vehicle_boxes = [d for d in self._last_boxes if d.label != "person"]
            person_boxes = [d for d in self._last_boxes if d.label == "person"]

            tracks, events = self.counter.update(vehicle_boxes)
            occupancy_now = self.occupancy.count_in_roi(person_boxes) if self.occupancy else 0

            for event in events:
                # Best-effort: how many people were in the windshield ROI
                # at the moment this vehicle crossed. Not per-vehicle
                # association, just a snapshot at crossing time.
                try:
                    db.log_event(event["direction"], event["label"], occupancy_now)
                except Exception as e:
                    print(f"[db] Failed to log crossing event: {e}")

            frame = self._draw_overlay(frame, tracks, person_boxes, occupancy_now)

            now = time.time()
            if now - last_emit >= frame_interval:
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if ok:
                    with state.lock:
                        state.jpeg_bytes = buf.tobytes()
                        state.count_in = self.counter.count_in
                        state.count_out = self.counter.count_out
                        state.count_by_label = dict(self.counter.count_by_label)
                        state.occupancy_now = occupancy_now
                        state.fps = 1.0 / (now - last_emit) if last_emit else 0.0
                last_emit = now

    def stop(self):
        self._running = False
        self.reader.stop()


processor = Processor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    import threading

    if db.enabled():
        await run_in_threadpool(db.init_db)
    else:
        print("[db] DATABASE_URL not set — crossing events won't be persisted.")

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
            "occupancy_now": state.occupancy_now,
        })


@app.get("/events")
async def events(limit: int = 50):
    """Recent logged crossing events (newest first). Empty list if no DB configured."""
    rows = await run_in_threadpool(db.fetch_recent_events, limit)
    return JSONResponse({"events": rows, "db_enabled": db.enabled()})


@app.get("/occupancy/summary")
async def occupancy_summary():
    """Aggregate occupancy stats across all logged crossings."""
    summary = await run_in_threadpool(db.fetch_occupancy_summary)
    return JSONResponse({"summary": summary, "db_enabled": db.enabled()})


@app.websocket("/ws/count")
async def ws_count(websocket: WebSocket):
    await websocket.accept()
    last_sent = None
    try:
        while True:
            with state.lock:
                payload = {
                    "count_in": state.count_in,
                    "count_out": state.count_out,
                    "total": state.count_in + state.count_out,
                    "count_by_label": state.count_by_label,
                    "camera_connected": state.connected,
                    "fps": round(state.fps, 1),
                    "occupancy_now": state.occupancy_now,
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
