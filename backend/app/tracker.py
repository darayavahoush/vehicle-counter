"""
Centroid-based multi-object tracker + line-crossing counter.

Deliberately simple (no Kalman filter, no re-ID embeddings): each frame,
new detection centroids are matched to existing tracks by nearest
distance. This is cheap enough to run every frame on CPU and is accurate
enough for vehicles moving through a fixed camera view at road speed.

Counting works by defining a virtual line (two points). For each track,
we remember which side of the line its centroid was on last frame; a
sign flip means the track crossed the line, and the direction of the
flip tells us which way (e.g. "in" vs "out").
"""

import math


def _centroid(detection):
    # Accepts either a plain (x, y, w, h) box or a detector.Detection
    # (x, y, w, h, label, conf) — only the first four fields matter here.
    x, y, w, h = detection[0], detection[1], detection[2], detection[3]
    return (x + w / 2.0, y + h / 2.0)


def _label_of(detection):
    return detection[4] if len(detection) > 4 else "vehicle"


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
    __slots__ = ("id", "centroid", "box", "label", "side", "disappeared", "counted", "_label_votes")

    def __init__(self, track_id, centroid, box, label, side):
        self.id = track_id
        self.centroid = centroid
        self.box = box
        self.label = label
        self.side = side
        self.disappeared = 0
        self.counted = False
        self._label_votes = {label: 1}

    def _vote_label(self, label):
        # A single frame's classification can wobble (car vs truck at a
        # bad angle); keep whichever label has been seen most often over
        # this track's lifetime so the on-screen label — and the label
        # locked in at counting time — doesn't flicker.
        self._label_votes[label] = self._label_votes.get(label, 0) + 1
        self.label = max(self._label_votes, key=self._label_votes.get)


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
        self.count_by_label = {}

    @property
    def total(self):
        return self.count_in + self.count_out

    def update(self, detections_in):
        """
        detections_in: list of (x, y, w, h) or detector.Detection objects
                       for the current frame.

        Returns (tracks, events):
          tracks: current list of active Track objects (for drawing).
          events: list of dicts, one per line-crossing that happened THIS
                  frame — {"track_id", "label", "direction"} where
                  direction is "in" or "out". Reported explicitly here
                  (rather than callers diffing count_in/count_out deltas)
                  so main.py can attach a timestamp + occupancy snapshot
                  and log each crossing as its own row, without guessing
                  which crossing a delta of >1 corresponds to.
        """
        detections = [(_centroid(d), (d[0], d[1], d[2], d[3]), _label_of(d)) for d in detections_in]
        unmatched_detections = set(range(len(detections)))
        matched_track_ids = set()
        events = []

        # Greedy nearest-neighbor matching: for each existing track, find the
        # closest unmatched detection within range.
        for track_id, track in self._tracks.items():
            best_idx, best_dist = None, self.max_match_distance
            for idx in unmatched_detections:
                c, _, _ = detections[idx]
                d = math.hypot(c[0] - track.centroid[0], c[1] - track.centroid[1])
                if d < best_dist:
                    best_idx, best_dist = idx, d

            if best_idx is not None:
                centroid, box, label = detections[best_idx]
                new_side = _side_of_line(centroid, self.line)
                track._vote_label(label)

                # Crossing = side flipped (and wasn't already counted this pass).
                if track.side != 0 and new_side != 0 and new_side != track.side and not track.counted:
                    direction = "in" if new_side > track.side else "out"
                    if direction == "in":
                        self.count_in += 1
                    else:
                        self.count_out += 1
                    track.counted = True
                    self.count_by_label[track.label] = self.count_by_label.get(track.label, 0) + 1
                    events.append({"track_id": track.id, "label": track.label, "direction": direction})

                track.centroid = centroid
                track.box = box
                track.side = new_side if new_side != 0 else track.side
                track.disappeared = 0
                unmatched_detections.discard(best_idx)
                matched_track_ids.add(track_id)

        # Detections with no matching track become new tracks.
        for idx in unmatched_detections:
            centroid, box, label = detections[idx]
            side = _side_of_line(centroid, self.line)
            self._tracks[self._next_id] = Track(self._next_id, centroid, box, label, side)
            self._next_id += 1

        # Age out tracks that weren't matched this frame; drop stale ones.
        for track in self._tracks.values():
            if track.id not in matched_track_ids:
                track.disappeared += 1

        self._tracks = {
            tid: t for tid, t in self._tracks.items() if t.disappeared <= self.max_disappeared
        }

        return list(self._tracks.values()), events
