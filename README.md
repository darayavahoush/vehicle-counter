# Tripwire — Live Vehicle Counter

Counts and labels vehicles crossing a virtual "tripwire" line in a live
CCTV/RTSP feed. Backend does all the work (capture, detection, tracking,
counting, overlay drawing, crossing-event logging); frontend just
displays the stream, the counts, and the log.

Live demo: https://tripwire-inky.vercel.app (backend on Render, looping
a real side-view traffic clip until wired to an actual camera)

## How it works

- **Capture**: a background thread continuously reads the RTSP stream (or
  loops a local video file for testing) and keeps only the latest frame,
  so processing never lags behind a growing buffer (`backend/app/capture.py`).
- **Detection**: `YoloOnnxDetector` runs a YOLOv8n model — exported to
  ONNX once, ahead of time — through OpenCV's built-in `cv2.dnn` module.
  No PyTorch or other ML runtime needed on the serving device; only a
  small (~12MB) `.onnx` file. Labels each detection as Car / Truck / Bus
  / Motorbike. Falls back automatically to `BackgroundSubtractionDetector`
  (zero-model, cheaper, but generic "Vehicle" labels only) if the model
  file isn't present (`backend/app/detector.py`).
- **Tracking + counting**: a centroid tracker assigns persistent IDs and
  carries each track's label frame-to-frame. A track is counted, and a
  crossing event is logged, when it crosses the configured virtual line —
  direction (in/out) included (`backend/app/tracker.py`).
- **Logging**: every crossing (timestamp, direction, label, track id,
  confidence) is kept in an in-memory rolling log, exposed via `GET
  /events` and pushed live over the websocket. In-memory is deliberate
  for now — zero setup, resets on redeploy. Swap in a real DB if events
  need to persist.
- **Serving**: detection/tracking runs ONCE in a background thread
  regardless of viewer count. `/video_feed` streams the already-annotated
  frame as MJPEG (`multipart/x-mixed-replace`) — the simplest thing a
  plain `<img>` tag can consume. `/ws/count` pushes count + latest-event
  updates only when they change.

## Tripwire line orientation

Default is a **vertical** line at mid-width — correct for a sideview
camera where vehicles travel left↔right across frame. For a top-down or
overhead camera, use a horizontal line instead (see `.env.example`).

## Run the backend

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # then edit RTSP_URL to your camera
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

For local testing without a real camera, set `RTSP_URL=0` in `.env` to
use your laptop webcam, or point it at a local video file path.

### Getting vehicle-type labels (YOLOv8n ONNX)

The backend runs fine without this (falls back to generic "Vehicle"
labels), but for real Car/Truck/Bus/Motorbike labels:

```bash
cd backend
pip install -r requirements-export.txt   # one-time, needs torch — not needed at runtime
python3 scripts/export_yolo_onnx.py --imgsz 320
git add models/yolov8n.onnx              # commit the ~12MB result; NOT yolov8n.pt
```

`--imgsz` must match `YOLO_INPUT_SIZE` in `.env`. Smaller = faster/cheaper
(better for a Raspberry Pi), larger = more accurate on small/far vehicles.

## Run the frontend

```bash
cd frontend
npm install
cp .env.example .env   # only needed if backend isn't on localhost:8000
npm run dev
```

Open http://localhost:5173.

## Deploying: Render (backend) + Vercel (frontend)

Vercel's serverless functions can't run this backend — it needs a
persistent background thread, a long-lived websocket, and an infinite
MJPEG stream. Render (or Railway) runs FastAPI as a normal long-running
process instead, which is what it actually needs.

1. **Backend on Render**: dashboard → New → Blueprint → connect the repo
   (auto-detects `render.yaml`). First build downloads the demo footage;
   commit `backend/models/yolov8n.onnx` beforehand if you want labels.
2. **Frontend on Vercel**: `vercel --prod` from `frontend/`, with
   `VITE_BACKEND_URL` set to the Render service's URL.
3. **CORS**: set `CORS_ORIGINS` on Render to the Vercel production URL.

Render's free tier spins down after 15 minutes idle — first request
after a quiet period takes ~30-60s to wake up.

## Tuning for your camera / hardware

All in `backend/.env`:

| Variable | Effect |
|---|---|
| `PROCESS_WIDTH` | Lower = faster processing, less accurate on small vehicles |
| `DETECT_EVERY_N_FRAMES` | Higher = less CPU, laggier box updates between detections |
| `OUTPUT_FPS` | Caps both encoding CPU and stream bandwidth |
| `JPEG_QUALITY` | Lower = less bandwidth, blockier image |
| `DETECTOR_BACKEND` | `yolo` (labeled) or `bg_subtraction` (cheaper, unlabeled) |
| `YOLO_INPUT_SIZE` | Smaller = faster/cheaper — try 256 or 192 on a weak Pi |
| `LINE_X1/Y1/X2/Y2` | Tripwire position, as fractions (0-1) of frame |

For a Raspberry Pi specifically: start with `DETECTOR_BACKEND=bg_subtraction`
to confirm the capture/streaming pipeline runs smoothly, then switch to
`yolo` and tune `YOLO_INPUT_SIZE` down (192-256) and `DETECT_EVERY_N_FRAMES`
up (4-6) until `/stats`' `fps` reading is acceptable for your Pi model.
