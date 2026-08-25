# Live Vehicle Counter

Counts vehicles crossing a virtual line in a live CCTV (RTSP) feed.
Backend does all the work (capture, detection, tracking, counting,
overlay drawing); frontend just displays the stream and the numbers.

## How it works

- **Capture**: a background thread continuously reads the RTSP stream
  and keeps only the latest frame, so processing is never lagging
  behind a growing buffer (`backend/app/capture.py`).
- **Detection**: YOLOv8n exported to ONNX, run through OpenCV's own
  `cv2.dnn` module — no PyTorch or onnxruntime needed at inference
  time, which is what makes it light enough for a Raspberry Pi. Gives
  a real vehicle label per box (car / truck / bus / motorcycle). An
  older MOG2 background-subtraction detector (no labels, but zero
  model file needed) is still available as a `DETECTOR_BACKEND=bgsub`
  fallback (`backend/app/detector.py`). Both speak the same
  `detect(frame) -> [Detection(x, y, w, h, label, conf), ...]`
  interface, so `tracker.py`/`main.py` don't care which one is running.
- **Tracking + counting**: a centroid tracker assigns persistent IDs
  frame-to-frame; a track is counted when it crosses the configured
  virtual line, with direction (in/out) (`backend/app/tracker.py`).
- **Serving**: detection/tracking runs ONCE in a background thread
  regardless of viewer count. `/video_feed` streams the already-
  annotated frame as MJPEG (`multipart/x-mixed-replace`) — the
  simplest thing a plain `<img>` tag can consume, no client-side
  video decoding needed. `/ws/count` pushes count updates only when
  they change.

## Run the backend

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # then edit RTSP_URL to your camera
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

For local testing without a real camera, set `RTSP_URL=0` in `.env` to
use your laptop webcam, or point it at a video file path.

**One-time model export** (do this once, on your Mac/PC — not the Pi):

```bash
pip install -r backend/requirements-export.txt
python backend/scripts/export_yolo_onnx.py
```

This downloads YOLOv8n's pretrained weights and exports
`backend/models/yolov8n.onnx` (~12MB). Copy just that one file to
wherever the backend runs:

```bash
scp backend/models/yolov8n.onnx pi@<pi-ip>:~/vehicle-counter/backend/models/yolov8n.onnx
```

The Pi's `requirements.txt` never installs `ultralytics`/`onnx` — it
only ever loads the exported `.onnx` file via `cv2.dnn`. If the file
isn't there yet, the backend logs a warning and falls back to the
label-less `bgsub` detector rather than crashing.

## Running on a Raspberry Pi

Same install steps as above (`pip install -r requirements.txt` — no
extra packages needed beyond `requirements.txt`). Pi-friendly starting
point for `backend/.env`:

```
PROCESS_WIDTH=480
DETECT_EVERY_N_FRAMES=3
YOLO_INPUT_SIZE=320
```

If it's still not keeping up with a 15fps source, raise
`DETECT_EVERY_N_FRAMES` further (tracking still runs every frame and
interpolates between detections) before dropping `YOLO_INPUT_SIZE` —
a smaller input size hurts small/distant-vehicle accuracy faster than
skipping a frame does.

## Run the frontend

```bash
cd frontend
npm install
cp .env.example .env   # only needed if backend isn't on localhost:8000
npm run dev
```

Open http://localhost:5173.

## Tuning for your camera / hardware

All in `backend/.env`:

| Variable | Effect |
|---|---|
| `PROCESS_WIDTH` | Lower = faster processing, less accurate on small vehicles |
| `DETECT_EVERY_N_FRAMES` | Higher = less CPU, laggier box updates between detections |
| `OUTPUT_FPS` | Caps both encoding CPU and stream bandwidth |
| `JPEG_QUALITY` | Lower = less bandwidth, blockier image |
| `LINE_X1/Y1/X2/Y2` | Counting line position, as fractions (0-1) of frame — vertical for sideview cameras, horizontal for top-down/angled ones |
| `DETECTOR_BACKEND` | `yolo` (labeled) or `bgsub` (no labels, no model file) |
| `YOLO_INPUT_SIZE` | Lower = faster inference, worse on small/distant vehicles |
| `YOLO_CONF_THRESHOLD` / `YOLO_NMS_THRESHOLD` | Detection confidence cutoff / overlap-merge threshold |

Camera orientation determines line orientation: a **sideview** camera
(vehicles cross left↔right) wants a **vertical** line — the shipped
default. A top-down or angled camera (vehicles move toward/away from
it) wants a **horizontal** line instead — flip the fractions back,
e.g. `LINE_X1=0.05,LINE_Y1=0.55 → LINE_X2=0.95,LINE_Y2=0.55`.
