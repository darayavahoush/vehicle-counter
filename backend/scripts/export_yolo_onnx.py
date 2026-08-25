#!/usr/bin/env python3
"""
One-time export: YOLOv8n -> ONNX, sized for cheap CPU inference (Render's
free tier, a Raspberry Pi, etc).

Run this once on any normal machine — it needs ultralytics + torch, which
are NOT needed on the actual inference device. Only the resulting small
.onnx file needs to reach the backend / Pi.

Usage:
    pip install -r requirements-export.txt
    python scripts/export_yolo_onnx.py --imgsz 320
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--imgsz", type=int, default=320,
        help="Must match YOLO_INPUT_SIZE in .env — the model only runs at this size.",
    )
    parser.add_argument("--out", default=None, help="Output path (default: backend/models/yolov8n.onnx)")
    args = parser.parse_args()

    model = YOLO("yolov8n.pt")  # auto-downloads the pretrained COCO checkpoint
    exported = model.export(format="onnx", imgsz=args.imgsz, opset=12)

    out_path = Path(args.out) if args.out else Path(__file__).parent.parent / "models" / "yolov8n.onnx"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(exported, out_path)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\nExported {out_path} ({size_mb:.1f} MB) at input size {args.imgsz}x{args.imgsz}.")
    print("Commit this file to the repo (git add backend/models/yolov8n.onnx) — the")
    print("backend loads it directly via cv2.dnn, no PyTorch needed at runtime.")
    print("\nDon't commit yolov8n.pt (the raw checkpoint this script downloaded) —")
    print("only the exported .onnx is needed.")


if __name__ == "__main__":
    main()
