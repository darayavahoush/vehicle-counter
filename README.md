# Live Vehicle Counter

Counts vehicles crossing a virtual line in a live CCTV (RTSP) feed.
Backend does all the work (capture, detection, tracking, counting,
overlay drawing); frontend just displays the stream and the numbers.

## How it works

- **Capture**: a background thread continuously reads the RTSP stream
  and keeps only the latest frame, so processing is never lagging
  behind a growing buffer (`backend/app/capture.py`).
- **Detection**: MOG2 background subtraction + contour filtering —
  no deep learning model, cheap enough for CPU-only hardware, and
  well-suited to a fixed camera angle (`backend/app/detector.py`).
  Swap in a DL model later behind the same `detect(frame) -> boxes`
  interface if a specific camera needs it.
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
| `LINE_X1/Y1/X2/Y2` | Counting line position, as fractions (0-1) of frame |

If the background-subtraction detector misfires a lot on your specific
camera (e.g. heavy shadows, moving trees, lighting flicker at night),
that's the point to swap `BackgroundSubtractionDetector` for a small
ONNX-exported YOLOv8n model — the tracker and counting logic don't
need to change, since both speak the same `[(x, y, w, h), ...]` box
format.
