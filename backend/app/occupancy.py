"""
Rough vehicle-occupancy estimation.

This does NOT track people individually or try to associate a specific
person with a specific vehicle track — that would need real depth/pose
info this camera doesn't have. Instead it's a cheap proxy: define a
"windshield ROI" (a fractional box near the counting line, where a
vehicle's cabin is expected to be as it crosses), and count how many
"person" detections from the same YOLO forward pass fall inside it on
a given frame. A pedestrian walking past elsewhere in the frame doesn't
land in the ROI and isn't counted; someone visible through a windshield
right as the vehicle crosses does.

The estimate is attached to each crossing event (best-effort "how many
people were near the line, in the windshield area, at that moment") and
also exposed as a live "current occupancy near line" stat.
"""


def _centroid(detection):
    x, y, w, h = detection[0], detection[1], detection[2], detection[3]
    return (x + w / 2.0, y + h / 2.0)


class OccupancyEstimator:
    def __init__(self, roi):
        """
        roi: (x1, y1, x2, y2) in pixel coords of the processed frame —
             the region around the counting line where a passing
             vehicle's windshield/cabin is expected to be.
        """
        self.roi = roi

    def count_in_roi(self, person_detections):
        """Count how many person-detection centroids fall inside the ROI."""
        x1, y1, x2, y2 = self.roi
        count = 0
        for d in person_detections:
            cx, cy = _centroid(d)
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                count += 1
        return count


def build_roi_from_line(line, frame_w, frame_h, half_width_frac=0.12):
    """
    Derive a windshield ROI as a band straddling the counting line,
    sized as a fraction of the frame's shorter dimension. Works for
    either a vertical (sideview) or horizontal (top-down) line since it
    just pads outward from the line's bounding box.
    """
    (x1, y1), (x2, y2) = line
    pad = half_width_frac * min(frame_w, frame_h)
    roi_x1 = max(0, min(x1, x2) - pad)
    roi_y1 = max(0, min(y1, y2) - pad)
    roi_x2 = min(frame_w, max(x1, x2) + pad)
    roi_y2 = min(frame_h, max(y1, y2) + pad)
    return (roi_x1, roi_y1, roi_x2, roi_y2)
