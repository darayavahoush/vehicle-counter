"""
Lightweight vehicle detector using classical background subtraction.

Rationale: for a fixed CCTV camera on CPU-only hardware, MOG2 background
subtraction + contour filtering is orders of magnitude cheaper than any
deep-learning detector, and works well because the background (road,
surroundings) is static while vehicles move through it.

This module exposes a single class, BackgroundSubtractionDetector, with
a detect(frame) -> list[(x, y, w, h)] interface. If a specific camera's
lighting/shadows make this unreliable, swap in a DL-based detector
(e.g. YOLOv8n via ONNX Runtime) behind the same interface without
touching tracker.py or main.py.
"""

import cv2


class BackgroundSubtractionDetector:
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
        """Returns a list of (x, y, w, h) bounding boxes for detected vehicles."""
        fg_mask = self._bg.apply(frame)

        # Shadows are labeled 127 by MOG2; drop them, keep solid foreground (255).
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Clean up noise, then close gaps so a vehicle forms one solid blob.
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self._kernel_open)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self._kernel_close, iterations=2)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area or area > self.max_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / float(h)
            if aspect < self.min_aspect or aspect > self.max_aspect:
                continue
            boxes.append((x, y, w, h))

        return boxes
