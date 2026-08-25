"""
Centroid-based multi-object tracker + line-crossing counter.

Deliberately simple (no Kalman filter, no re-ID embeddings): each frame,
new detection centroids are matched to existing tracks by nearest
distance. This is cheap enough to run every frame on CPU and is accurate
enough for vehicles moving through a fixed camera view at road speed.

Counting works by defining a virtual line (two points). For each track,
we remember which side of the line its centroid was on last frame; a
sign flip means the track crossed the line, and the direction of the
flip tells us which way ("in" vs "out").

Detections now carry a label + confidence (from the detector), so each
Track carries a label too, and each crossing event reports what type of
vehicle it was — this is what feeds the crossing-event log in main.py.
"""

import math
import time


def _centroid(box):
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


def _side_of_line(point, line):
    """Sign of the cross product; tells us which side of the line a point is on."""
    (x1, y1), (x2, y2) = line
    px, py = point
    cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    if cross > 0:
        return 1
    if cross < 0:
        return -1
    return 0


class Track:
    __slots__ = ("id", "centroid", "box", "label", "confidence", "side", "disappeared", "counted")

    def __init__(self, track_id, centroid, box, label, confidence, side):
        self.id = track_id
        self.centroid = centroid
        self.box = box
        self.label = label
        self.confidence = confidence
        self.side = side
        self.disappeared = 0
        self.counted = False


class LineCrossingCounter:
    def __init__(self, line, max_disappeared=15, max_match_distance=80):
        """
        line: ((x1, y1), (x2, y2)) — the virtual counting line, in frame
              pixel coordinates (matches the frame size the detector runs on).
        max_disappeared: frames a track can go unmatched before being dropped.
        max_match_distance: max pixel distance to match a detection to a
                             prior track's centroid (prevents id-swapping
                             between unrelated vehicles).
        """
        self.line = line
        self.max_disappeared = max_disappeared
        self.max_match_distance = max_match_distance

        self._tracks = {}
        self._next_id = 1

        self.count_in = 0
        self.count_out = 0
        # Per-label tallies, e.g. {"Car": {"in": 4, "out": 2}, "Truck": {...}}
        self.count_by_label = {}

    @property
    def total(self):
        return self.count_in + self.count_out

    def _record_crossing(self, direction, label):
        if direction == "in":
            self.count_in += 1
        else:
            self.count_out += 1
        bucket = self.count_by_label.setdefault(label, {"in": 0, "out": 0})
        bucket[direction] += 1

    def update(self, detections):
        """
        detections: list of (x, y, w, h, label, confidence) — the unified
                    detector output format (see detector.py).
        Returns (tracks, events):
          tracks: current list of active Track objects (for drawing)
          events: list of dicts for crossings that happened THIS call, e.g.
                  {"direction": "in", "label": "Car", "track_id": 7,
                   "confidence": 0.87, "timestamp": 1735000000.0}
                  main.py appends these to the persistent crossing log.
        """
        parsed = [((_centroid(d[:4])), d[:4], d[4], d[5]) for d in detections]
        unmatched_detections = set(range(len(parsed)))
        matched_track_ids = set()
        events = []

        for track_id, track in self._tracks.items():
            best_idx, best_dist = None, self.max_match_distance
            for idx in unmatched_detections:
                c = parsed[idx][0]
                d = math.hypot(c[0] - track.centroid[0], c[1] - track.centroid[1])
                if d < best_dist:
                    best_idx, best_dist = idx, d

            if best_idx is not None:
                centroid, box, label, confidence = parsed[best_idx]
                new_side = _side_of_line(centroid, self.line)

                if track.side != 0 and new_side != 0 and new_side != track.side and not track.counted:
                    direction = "in" if new_side > track.side else "out"
                    self._record_crossing(direction, track.label)
                    track.counted = True
                    events.append({
                        "direction": direction,
                        "label": track.label,
                        "track_id": track.id,
                        "confidence": round(track.confidence, 2),
                        "timestamp": time.time(),
                    })

                track.centroid = centroid
                track.box = box
                track.label = label
                track.confidence = confidence
                track.side = new_side if new_side != 0 else track.side
                track.disappeared = 0
                unmatched_detections.discard(best_idx)
                matched_track_ids.add(track_id)

        for idx in unmatched_detections:
            centroid, box, label, confidence = parsed[idx]
            side = _side_of_line(centroid, self.line)
            self._tracks[self._next_id] = Track(self._next_id, centroid, box, label, confidence, side)
            self._next_id += 1

        for track in self._tracks.values():
            if track.id not in matched_track_ids:
                track.disappeared += 1

        self._tracks = {
            tid: t for tid, t in self._tracks.items() if t.disappeared <= self.max_disappeared
        }

        return list(self._tracks.values()), events
