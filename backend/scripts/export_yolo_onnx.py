"""
One-time export step: YOLOv8n (PyTorch) -> ONNX.

Run this on your dev machine (Mac/PC), NOT on the Raspberry Pi. The Pi
only ever needs the resulting .onnx file — it loads it via OpenCV's
cv2.dnn, so it never needs PyTorch, ultralytics, or onnxruntime
installed. That's the whole point of doing it this way: keeps the Pi
side to just `opencv-python-headless`.

Usage:
    pip install -r backend/requirements-export.txt
    python backend/scripts/export_yolo_onnx.py

Produces backend/models/yolov8n.onnx (~12MB). Copy that one file to
the Pi (e.g. scp) into the same backend/models/ path, or point
YOLO_ONNX_PATH at wherever you put it.
"""

import os

from ultralytics import YOLO

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
IMG_SIZE = int(os.getenv("YOLO_EXPORT_IMG_SIZE", "320"))  # match main.py's YOLO_INPUT_SIZE


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = YOLO("yolov8n.pt")  # downloads the pretrained COCO weights on first run
    exported_path = model.export(format="onnx", imgsz=IMG_SIZE, simplify=True, opset=12)

    dest = os.path.join(OUTPUT_DIR, "yolov8n.onnx")
    if os.path.abspath(exported_path) != os.path.abspath(dest):
        os.replace(exported_path, dest)

    size_mb = os.path.getsize(dest) / (1024 * 1024)
    print(f"\nExported {dest} ({size_mb:.1f} MB) at input size {IMG_SIZE}x{IMG_SIZE}.")
    print("Copy this one file to the Pi, e.g.:")
    print(f"  scp {dest} pi@<pi-ip>:~/vehicle-counter/backend/models/yolov8n.onnx")


if __name__ == "__main__":
    main()
