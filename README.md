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
  `update()` returns each frame's line-crossing events explicitly
  (track id, label, direction), so downstream code never has to guess
  which crossing a count delta corresponds to.
- **Occupancy (optional)**: the same YOLO forward pass already scores
  a "person" class alongside vehicle classes, so spotting people costs
  nothing extra. `backend/app/occupancy.py` counts person-detections
  falling inside a "windshield ROI" — a band straddling the counting
  line — as a rough per-crossing occupancy estimate. It's a proxy, not
  real per-vehicle association: a pedestrian standing right at the
  line will register too.
- **Persistence (optional)**: if `DATABASE_URL` is set, every crossing
  event (direction, label, occupancy snapshot) is logged to Postgres
  (`backend/app/db.py`). Leave it unset and the app behaves exactly as
  before — in-memory counts only.
- **Serving**: detection/tracking runs ONCE in a background thread
  regardless of viewer count. `/video_feed` streams the already-
  annotated frame as MJPEG (`multipart/x-mixed-replace`) — the
  simplest thing a plain `<img>` tag can consume, no client-side
  video decoding needed. `/ws/count` pushes count updates only when
  they change. `/events` and `/occupancy/summary` read from Postgres
  when persistence is enabled (empty/null otherwise).

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

## Occupancy estimation

Off by default cost-wise (it's free — same forward pass — but the
detection itself is controlled by `DETECT_PERSON`, default `true`).
Only has an effect with `DETECTOR_BACKEND=yolo`.

| Variable | Effect |
|---|---|
| `DETECT_PERSON` | Include the "person" class in YOLO's kept output, enabling occupancy estimation |
| `OCCUPANCY_ROI_HALF_WIDTH_FRAC` | How far the "windshield" ROI extends past the counting line, as a fraction of the frame's shorter dimension |

The current estimate is in `/stats` and `/ws/count` as `occupancy_now`,
and drawn as a purple ROI box + `OCC: N` in the video overlay. It's
attached to each crossing event at the moment of crossing when logged
to Postgres (see below) — a snapshot, not tracked per-vehicle.

## Persisting crossing events (Postgres)

Set `DATABASE_URL` in `backend/.env` to log every crossing event
(timestamp, direction, label, occupancy snapshot) to Postgres. Leave
it unset and nothing changes — counts stay in-memory only, and
`/events` / `/occupancy/summary` just return empty/null.

**Use [Neon](https://neon.tech), not Render's own Postgres**, if you
want this data to actually stick around: Render's free Postgres tier
[expires and gets deleted after 30 days](https://render.com/changelog/free-postgresql-instances-now-expire-after-30-days-previously-90).
Neon's free tier has no such expiry. Create a free Neon project, copy
its connection string into `DATABASE_URL`:

```
DATABASE_URL=postgresql://user:password@ep-example-12345.us-east-2.aws.neon.tech/vehicledb?sslmode=require
```

The table (`crossing_events`) is created automatically on startup if
it doesn't exist — no manual migration step.

New endpoints once persistence is on:

| Endpoint | Returns |
|---|---|
| `GET /events?limit=50` | Most recent crossing events, newest first |
| `GET /occupancy/summary` | Total logged crossings with an occupancy reading + their average |

The frontend polls `/events` every 5s to show a "recent crossings"
panel; if `DATABASE_URL` isn't set it just shows a note explaining why
the panel is empty, instead of looking broken.

## Camera on a local network + a cloud-hosted frontend (e.g. Vercel)

If your CCTV/RTSP feed is only reachable on your home/local network
(no port-forwarding, static IP, or DVR cloud-relay feature) but you
want the frontend on a public host like Vercel, the backend — which is
the thing that actually needs to reach the camera — has to run
**somewhere on that same local network**, not on Vercel or Render.
Vercel/Render can happily host the *frontend*, but they cannot reach
into your home network to pull an RTSP stream that isn't exposed to
the internet.

Practical options, roughly in order of effort:

1. **Run the backend on a machine on your LAN** (the Raspberry Pi
   itself, or any always-on machine near the camera) and expose just
   that machine's port 8000 to the internet via a tunnel — e.g.
   [Tailscale Funnel](https://tailscale.com/kb/1223/funnel),
   [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/),
   or `ngrok`. Point `VITE_BACKEND_URL` (frontend `.env`) at the
   tunnel's public URL. This keeps the RTSP credentials off the public
   internet entirely — only the already-processed MJPEG/JSON leaves
   your network.
2. **Enable your DVR/NVR's own cloud-relay feature**, if it has one
   (many consumer CCTV systems do) — then the backend (wherever it
   runs) can pull the feed over that instead of raw LAN-only RTSP.
3. **Port-forward + dynamic DNS**, the traditional approach — more
   exposure (RTSP itself becomes reachable from the internet, ideally
   behind a VPN instead) so it's the least recommended of these three
   unless you're already comfortable hardening that.

Whichever you pick, the frontend on Vercel and `render.yaml`'s hosted
backend demo are unaffected either way — they use a bundled demo video
file (`RTSP_URL=demo_videos/car-detection.mp4`), not your camera.
