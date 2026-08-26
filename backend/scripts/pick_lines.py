"""
Pick tripwire line(s) by clicking on an actual frame from your camera,
instead of guessing LINE_X1..Y2 fractions blind.

Usage:
    python backend/scripts/pick_lines.py --source rtsp://user:pass@ip:554/stream1
    python backend/scripts/pick_lines.py --source 0                     # webcam
    python backend/scripts/pick_lines.py --source demo_videos/car-detection.mp4

Controls (once the window opens):
    - Click two points to draw one tripwire line. You'll be asked in the
      terminal to name it (e.g. "northbound") — press Enter to just call
      it "main" if you only need one line.
    - Repeat for as many lines as you need (e.g. one per carriageway on
      a divided highway).
    - 'u' undoes your last click.
    - 'z' undoes your last completed line entirely.
    - 'q' or ENTER (with no pending click) finishes and prints the config.

Output: a ready-to-paste LINES_JSON line for backend/.env, plus
backend/scripts/line_preview.png so you can double check the placement
without needing to re-open the live window.

Needs a display (run this on your Mac / wherever you have a screen —
not over a headless SSH session to the Pi). If you're picking lines for
a camera the Pi will run in production, point --source at the same
RTSP URL from your Mac; the fractions this outputs don't depend on
which machine draws them, only on the frame's aspect ratio.
"""

import argparse
import json
import os
import sys

import cv2

WINDOW_NAME = "Click 2 points per tripwire  |  u=undo point  z=undo line  q/ENTER=done"


def grab_one_frame(source, retries=20):
    src = int(source) if str(source).strip().isdigit() else source
    cap = cv2.VideoCapture(src)
    frame = None
    for _ in range(retries):
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
            break
    cap.release()
    if frame is None:
        raise RuntimeError(
            f"Couldn't read a frame from source {source!r}. Check the RTSP URL/credentials, "
            f"or that a webcam is actually available at that index."
        )
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="RTSP URL, webcam index (e.g. 0), or video file path")
    args = parser.parse_args()

    print(f"Grabbing a frame from {args.source!r} ...")
    frame = grab_one_frame(args.source)
    h, w = frame.shape[:2]
    print(f"Got a {w}x{h} frame.")

    points = []        # pending clicks for the line currently being drawn
    lines = []          # completed {"name", x1, y1, x2, y2} (pixel coords for now)
    colors = [(0, 220, 255), (255, 140, 60), (120, 220, 120), (220, 100, 220)]

    def redraw():
        canvas = frame.copy()
        for i, ln in enumerate(lines):
            color = colors[i % len(colors)]
            cv2.line(canvas, (ln["x1"], ln["y1"]), (ln["x2"], ln["y2"]), color, 2)
            mx, my = (ln["x1"] + ln["x2"]) // 2, (ln["y1"] + ln["y2"]) // 2
            cv2.putText(canvas, ln["name"], (mx + 6, my - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        for p in points:
            cv2.circle(canvas, p, 5, (0, 0, 255), -1)
        if len(points) == 1:
            cv2.putText(canvas, "click the 2nd point...", (12, h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, canvas)

    def on_mouse(event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        points.append((x, y))
        if len(points) == 2:
            (x1, y1), (x2, y2) = points
            name = input(f"Name for this line (Enter for '{'main' if not lines else f'line{len(lines) + 1}'}'): ").strip()
            if not name:
                name = "main" if not lines else f"line{len(lines) + 1}"
            lines.append({"name": name, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
            points.clear()
        redraw()

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    redraw()

    print("\nClick two points to draw each tripwire line. Press 'q' or ENTER when done.\n")
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (ord("q"), 13):  # 'q' or ENTER
            if points:
                print("(finish or undo the pending line first)")
                continue
            break
        elif key == ord("u"):
            if points:
                points.pop()
                redraw()
        elif key == ord("z"):
            if lines:
                removed = lines.pop()
                print(f"Removed line '{removed['name']}'.")
                redraw()

    cv2.destroyAllWindows()

    if not lines:
        print("No lines drawn — nothing to output.")
        sys.exit(0)

    # Convert pixel coords -> fractions (0..1) of THIS frame's dimensions.
    # Fractions are what main.py actually reads, and they're independent
    # of PROCESS_WIDTH since aspect ratio is preserved on resize.
    specs = [
        {
            "name": ln["name"],
            "x1": round(ln["x1"] / w, 4),
            "y1": round(ln["y1"] / h, 4),
            "x2": round(ln["x2"] / w, 4),
            "y2": round(ln["y2"] / h, 4),
        }
        for ln in lines
    ]

    preview = frame.copy()
    for i, ln in enumerate(lines):
        color = colors[i % len(colors)]
        cv2.line(preview, (ln["x1"], ln["y1"]), (ln["x2"], ln["y2"]), color, 2)
        mx, my = (ln["x1"] + ln["x2"]) // 2, (ln["y1"] + ln["y2"]) // 2
        cv2.putText(preview, ln["name"], (mx + 6, my - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    preview_path = os.path.join(os.path.dirname(__file__), "line_preview.png")
    cv2.imwrite(preview_path, preview)

    print("\nPaste this into backend/.env:\n")
    print(f"LINES_JSON={json.dumps(specs)}")
    print(f"\nPreview image saved to {preview_path} — double check the placement there.")
    if len(specs) == 1:
        print(
            "\n(Single line — you could equivalently use LINE_X1/Y1/X2/Y2 instead of "
            "LINES_JSON, but LINES_JSON works fine either way.)"
        )


if __name__ == "__main__":
    main()
