#!/usr/bin/env python3
"""
Count vehicles in a *static* dense scene (drone/aerial parking-lot shot,
stadium overflow lot, etc.) — as opposed to counting vehicles that cross
a tripwire line in a live traffic feed, which is what the rest of this
repo (capture.py / tracker.py / main.py) is built for.

WHY THE EXISTING PIPELINE DOESN'T APPLY HERE
----------------------------------------------
1. tracker.py's LineCrossingCounter only increments a count when a
   track's centroid flips sides of a configured line. Parked vehicles
   never move, so nothing ever crosses the line — the repo's existing
   "count" would be 0 (or whatever few vehicles happen to be driving
   past on the road at the edge of frame) no matter how many hundred
   vehicles are sitting in the lot.
2. detector.py's YoloOnnxDetector resizes the *entire* frame down to
   320x320 (the fixed size the ONNX graph was exported at) in one shot.
   For a 1920x1080 drone frame with hundreds of small vehicles, that
   squashes each car down to a handful of pixels — on a sample frame
   from the uploaded video this found 3 vehicles out of a lot with
   ~150-250 visible. It's not that the model is bad; it never sees
   the cars at a usable resolution.

WHAT THIS SCRIPT DOES INSTEAD
--------------------------------
- Grabs one frame (from an image, or a specific frame of a video).
- Slices it into overlapping 320x320 tiles (matching the ONNX model's
  fixed input) and runs the *same* existing YoloOnnxDetector on each
  tile at native resolution, so small/distant vehicles are actually
  visible to the model.
- Merges tile detections back into full-frame coordinates and runs one
  global NMS pass to collapse duplicate boxes from overlapping tiles.
- Reports a total count + per-class breakdown, and saves an annotated
  image so you can visually sanity-check misses/doubles.

This reuses backend/app/detector.py as-is (no retraining, no new
model) — drop this file in backend/scripts/ next to pick_lines.py.

Usage:
    python backend/scripts/count_static_vehicles.py --source DJI_0025.MP4
    python backend/scripts/count_static_vehicles.py --source DJI_0025.MP4 --frame 100
    python backend/scripts/count_static_vehicles.py --source lot.jpg --tile 256 --overlap 96 --conf 0.2

Tuning notes:
    --conf         Lower (e.g. 0.15-0.2) recovers more of the small/
                   distant vehicles at the cost of more false positives.
    --tile/--overlap
                   Smaller tiles = each vehicle takes up more of the
                   model's input = better recall on tiny/far vehicles,
                   but more tiles = slower. More overlap reduces vehicles
                   getting cut in half at a tile boundary, at the cost
                   of more duplicate-detection work for the NMS pass.
    Very dense, tightly-packed clusters (e.g. a depot of buses parked
    nose-to-tail) are a hard case for ANY detector, including a human
    glancing at the image once — expect undercounting there specifically,
    and treat the output as a strong estimate, not an exact tally.
"""

import argparse
import os
import sys
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from detector import YoloOnnxDetector  # noqa: E402


def grab_frame(source, frame_index=None):
    """Read a single frame from an image path or a video path."""
    ext = os.path.splitext(source)[1].lower()
    if ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        frame = cv2.imread(source)
        if frame is None:
            raise RuntimeError(f"Couldn't read image {source!r}.")
        return frame

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Couldn't open video {source!r}.")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_index is None:
        frame_index = total // 2  # middle frame: avoids startup/end artifacts
    frame_index = max(0, min(frame_index, max(total - 1, 0)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Couldn't read frame {frame_index} from {source!r} ({total} frames total).")
    print(f"Using frame {frame_index}/{max(total - 1, 0)} from {source!r}.")
    return frame


def tiled_detect(detector, frame, tile, overlap, conf_threshold, nms_threshold):
    h, w = frame.shape[:2]
    stride = tile - overlap
    if stride <= 0:
        raise ValueError("--overlap must be smaller than --tile")

    boxes, confidences, labels = [], [], []

    ys = list(range(0, max(h - tile, 0) + 1, stride)) or [0]
    if ys[-1] + tile < h:
        ys.append(h - tile)
    xs = list(range(0, max(w - tile, 0) + 1, stride)) or [0]
    if xs[-1] + tile < w:
        xs.append(w - tile)

    for ty in ys:
        for tx in xs:
            crop = frame[ty:ty + tile, tx:tx + tile]
            ch, cw = crop.shape[:2]
            if ch < tile or cw < tile:
                # pad the last row/column of tiles so the model always sees
                # a full tile x tile input (avoids distorting a partial tile)
                padded = np.zeros((tile, tile, 3), dtype=frame.dtype)
                padded[:ch, :cw] = crop
                crop = padded
            for (bx, by, bw, bh, label, conf) in detector.detect(crop):
                boxes.append([bx + tx, by + ty, bw, bh])
                confidences.append(conf)
                labels.append(label)

    if not boxes:
        return []

    keep = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    keep = np.array(keep).flatten() if len(keep) else []
    return [(boxes[i], labels[i], confidences[i]) for i in keep]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="Image or video file path")
    parser.add_argument("--frame", type=int, default=None, help="Video frame index (default: middle frame)")
    parser.add_argument("--model", default=os.path.join(os.path.dirname(__file__), "..", "models", "yolov8n.onnx"))
    parser.add_argument("--tile", type=int, default=320, help="Tile size in px (must match the ONNX export size)")
    parser.add_argument("--overlap", type=int, default=80, help="Overlap between adjacent tiles, in px")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--nms", type=float, default=0.4, help="NMS IoU threshold for the global merge pass")
    parser.add_argument("--out", default="vehicle_count_annotated.png", help="Annotated output image path")
    args = parser.parse_args()

    frame = grab_frame(args.source, args.frame)
    h, w = frame.shape[:2]
    print(f"Frame size: {w}x{h}")

    detector = YoloOnnxDetector(args.model, input_size=args.tile, conf_threshold=args.conf, nms_threshold=args.nms)
    detections = tiled_detect(detector, frame, args.tile, args.overlap, args.conf, args.nms)

    counts = Counter(label for _, label, _ in detections)
    print(f"\nTotal vehicles detected: {len(detections)}")
    for label, n in counts.most_common():
        print(f"  {label}: {n}")

    out = frame.copy()
    for (box, label, conf) in detections:
        x, y, bw, bh = box
        cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 0, 255), 1)
    cv2.imwrite(args.out, out)
    print(f"\nAnnotated image saved to {args.out} — check it for missed/duplicate boxes before trusting the count.")


if __name__ == "__main__":
    main()
