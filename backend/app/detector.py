"""
Vehicle detectors, both behind the same detect(frame) -> list[Detection]
interface, so main.py and tracker.py never need to know which one is
running.

Two options:

1. BackgroundSubtractionDetector — MOG2 + contour filtering. Cheapest
   possible option on CPU, but gives no vehicle *label* (just "vehicle"),
   and only works well for a genuinely fixed camera.

2. YoloOnnxDetector — YOLOv8n exported to ONNX, run through OpenCV's
   built-in cv2.dnn module. No PyTorch and no extra runtime dependency
   on the device that actually runs this (opencv-python-headless already
   ships dnn) — the export to ONNX happens once, on a dev machine, via
   scripts/export_yolo_onnx.py. This is what gives per-box vehicle
   labels (car / truck / bus / motorcycle) and is what should run on a
   Raspberry Pi: the .onnx file is ~12MB and inference at a small input
   size (default 320x320) is fast enough on a Pi 4/5 CPU at 15 fps if
   you keep DETECT_EVERY_N_FRAMES >= 2 (see main.py / README).

Pick between them with build_detector(kind, **kwargs).
"""

from collections import namedtuple

import cv2
import numpy as np

# (x, y, w, h) box in frame pixel coords, a string label, and a 0..1
# confidence (None for the background-subtraction detector, which has
# no real notion of confidence).
Detection = namedtuple("Detection", ["x", "y", "w", "h", "label", "conf"])


class BackgroundSubtractionDetector:
    """Classical CV fallback — no ML model, no labels beyond 'vehicle'."""

    def __init__(
        self,
        min_area=900,
        max_area=60000,
        min_aspect=0.3,
        max_aspect=3.5,
        history=500,
        var_threshold=40,
        detect_shadows=True,
    ):
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect

        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )
        self._kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    def detect(self, frame):
        fg_mask = self._bg.apply(frame)

        # Shadows are labeled 127 by MOG2; drop them, keep solid foreground (255).
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Clean up noise, then close gaps so a vehicle forms one solid blob.
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self._kernel_open)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self._kernel_close, iterations=2)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area or area > self.max_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / float(h)
            if aspect < self.min_aspect or aspect > self.max_aspect:
                continue
            detections.append(Detection(x, y, w, h, "vehicle", None))

        return detections


# COCO class ids for the vehicle classes we care about (YOLOv8's default
# training set is COCO). Anything else (person, dog, etc.) is discarded.
_COCO_VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


class YoloOnnxDetector:
    """YOLOv8n via cv2.dnn — no PyTorch/onnxruntime needed on the device."""

    def __init__(
        self,
        onnx_path,
        input_size=320,
        conf_threshold=0.35,
        nms_threshold=0.45,
        classes=None,
    ):
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.classes = classes or _COCO_VEHICLE_CLASSES

        self._net = cv2.dnn.readNetFromONNX(onnx_path)
        # CPU is the only backend/target guaranteed present everywhere
        # (including a Pi with no GPU), so pin it explicitly rather than
        # relying on OpenCV's default.
        self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def detect(self, frame):
        h, w = frame.shape[:2]
        size = self.input_size

        # Letterbox resize: pad to square with the original aspect ratio
        # kept, so the model doesn't see a squashed image. scale/pad are
        # kept so we can map boxes back to original frame coordinates.
        scale = min(size / w, size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (size, size), swapRB=True, crop=False)
        self._net.setInput(blob)
        raw = self._net.forward()  # shape (1, 84, N) for YOLOv8: 4 box + 80 class scores

        return self._postprocess(raw, scale, pad_x, pad_y)

    def _postprocess(self, raw, scale, pad_x, pad_y):
        # (1, 84, N) -> (N, 84): rows are candidate boxes, columns are
        # [cx, cy, w, h, class_0_score, ..., class_79_score].
        preds = raw[0].T

        boxes, confidences, class_ids = [], [], []
        for row in preds:
            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))
            conf = float(class_scores[class_id])
            if conf < self.conf_threshold or class_id not in self.classes:
                continue

            cx, cy, bw, bh = row[:4]
            # Undo letterbox padding/scale to get back to original frame coords.
            x = (cx - bw / 2 - pad_x) / scale
            y = (cy - bh / 2 - pad_y) / scale
            bw, bh = bw / scale, bh / scale

            boxes.append([int(x), int(y), int(bw), int(bh)])
            confidences.append(conf)
            class_ids.append(class_id)

        if not boxes:
            return []

        keep = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        keep = keep.flatten() if len(keep) else []

        detections = []
        for i in keep:
            x, y, bw, bh = boxes[i]
            detections.append(
                Detection(max(x, 0), max(y, 0), bw, bh, self.classes[class_ids[i]], confidences[i])
            )
        return detections


def build_detector(kind, **kwargs):
    if kind == "yolo":
        return YoloOnnxDetector(**kwargs)
    if kind == "bgsub":
        return BackgroundSubtractionDetector(**kwargs)
    raise ValueError(f"Unknown detector kind: {kind!r} (expected 'yolo' or 'bgsub')")
