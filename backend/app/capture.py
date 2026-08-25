"""
Threaded video capture that always exposes the LATEST frame.

Why this exists: cv2.VideoCapture buffers frames internally. If your
processing loop is slower than the camera's FPS (common on CPU), the
buffer fills up and you start processing frames that are seconds old,
with the lag growing over time. This class runs capture in its own
thread, continuously grabbing frames and discarding all but the most
recent one, so processing always sees "now", not a growing backlog.
"""

import os
import threading
import time
import cv2


class LatestFrameReader:
    def __init__(self, source, reconnect_delay=2.0):
        """
        source: RTSP URL (str), video file path (str), or webcam index (int)
        reconnect_delay: seconds to wait before retrying a dropped stream
        """
        self.source = source
        self.reconnect_delay = reconnect_delay
        # A local video file "disconnecting" just means it hit EOF — loop it
        # instantly instead of treating it like a dropped camera connection.
        self._is_local_file = isinstance(source, str) and os.path.isfile(source)
        self._cap = None
        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._connected = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _open(self):
        cap = cv2.VideoCapture(self.source)
        # Keep OpenCV's own buffer as small as possible; we do our own
        # "latest frame" buffering above this layer.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _run(self):
        self._cap = self._open()
        self._connected = self._cap.isOpened()

        while self._running:
            if not self._cap or not self._cap.isOpened():
                self._connected = False
                time.sleep(self.reconnect_delay)
                self._cap = self._open()
                continue

            ok, frame = self._cap.read()
            if not ok:
                if self._is_local_file:
                    # End of file, not a dropped connection — loop instantly.
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self._connected = False
                time.sleep(self.reconnect_delay)
                self._cap.release()
                self._cap = self._open()
                continue

            self._connected = True
            with self._lock:
                self._frame = frame

    def read(self):
        """Returns the most recent frame (or None if nothing yet)."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    @property
    def connected(self):
        return self._connected

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
