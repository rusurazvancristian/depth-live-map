"""Distance calibration using a known A5 sheet (150x205mm) at 1.0m.

Usage:
    python3 src/calibration/calibrate_distance.py

Controls:
    SPACE  — freeze frame and start drawing
    Mouse  — click-drag to draw rectangle around paper
    ENTER  — save calibration
    ESC    — retry from live feed
    Q      — quit without saving
"""

import json
import logging
import os
import time

import cv2
import numpy as np
from picamera2 import Picamera2

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

PAPER_W_M    = 0.150
PAPER_H_M    = 0.205
KNOWN_DIST_M = 1.0
CAM_W, CAM_H = 640, 480
OUT_PATH = os.path.join(os.path.dirname(__file__), "intrinsics.json")
WINDOW   = "Calibration"


# ── Mouse state ───────────────────────────────────────────────────────────────
class _Mouse:
    def __init__(self):
        self.pt1 = None   # (x, y) drag start
        self.pt2 = None   # (x, y) current/end
        self.done = False # True when mouse released

    def reset(self):
        self.pt1 = self.pt2 = None
        self.done = False

    def callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pt1 = (x, y)
            self.pt2 = (x, y)
            self.done = False
        elif event == cv2.EVENT_MOUSEMOVE and self.pt1 is not None and not self.done:
            self.pt2 = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.pt1 is not None:
            self.pt2 = (x, y)
            self.done = True

    def roi(self):
        """Return (x1, y1, x2, y2) normalised, or None if too small."""
        if self.pt1 is None or self.pt2 is None:
            return None
        x1, y1 = min(self.pt1[0], self.pt2[0]), min(self.pt1[1], self.pt2[1])
        x2, y2 = max(self.pt1[0], self.pt2[0]), max(self.pt1[1], self.pt2[1])
        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return None
        return x1, y1, x2, y2


def _open_camera():
    cam = Picamera2()
    cfg = cam.create_preview_configuration(
        main={"size": (CAM_W, CAM_H), "format": "BGR888"},
        controls={"FrameRate": 30, "AfMode": 0, "LensPosition": 0.0},
    )
    cam.configure(cfg)
    cam.start()
    time.sleep(1.0)
    return cam


def _overlay_text(img, lines, y0=30, color=(0, 220, 255)):
    for i, line in enumerate(lines):
        cv2.putText(img, line, (10, y0 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (10, y0 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)


def _show(img):
    """Display BGR image on Wayland (imshow expects RGB)."""
    cv2.imshow(WINDOW, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def main():
    cam = _open_camera()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, CAM_W, CAM_H)

    mouse = _Mouse()
    cv2.setMouseCallback(WINDOW, mouse.callback)

    # States: LIVE → FROZEN → RESULT
    state = "LIVE"
    frozen = None
    intrinsics = None

    logger.info("Ready. Hold A5 paper at 1.0m and press SPACE.")

    try:
        while True:
            key = cv2.waitKey(1) & 0xFF

            # ── Q always quits ────────────────────────────────────────────
            if key == ord("q"):
                logger.info("Quit without saving.")
                break

            # ── LIVE: stream camera, wait for SPACE ───────────────────────
            if state == "LIVE":
                frame = cam.capture_array()
                vis = frame.copy()
                _overlay_text(vis, ["Hold A5 paper at 1.0m", "SPACE = freeze   Q = quit"])
                cx, cy = CAM_W // 2, CAM_H // 2
                cv2.line(vis, (cx-20, cy), (cx+20, cy), (0,255,0), 1)
                cv2.line(vis, (cx, cy-20), (cx, cy+20), (0,255,0), 1)
                _show(vis)

                if key == ord(" "):
                    frozen = frame.copy()
                    mouse.reset()
                    state = "FROZEN"
                    logger.info("Frozen. Draw rectangle around the paper.")

            # ── FROZEN: show frozen frame, user draws rectangle ───────────
            elif state == "FROZEN":
                vis = frozen.copy()
                _overlay_text(vis, ["Draw rectangle around paper",
                                    "ESC = retry live   Q = quit"])

                roi = mouse.roi()
                if roi:
                    x1, y1, x2, y2 = roi
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

                _show(vis)

                # Accept completed rectangle with ENTER
                if key == 13 and mouse.done:
                    roi = mouse.roi()
                    if roi is None:
                        logger.warning("Rectangle too small — draw again.")
                    else:
                        x1, y1, x2, y2 = roi
                        rw, rh = x2 - x1, y2 - y1
                        f_y = (rh * PAPER_H_M) / KNOWN_DIST_M
                        f_x = (rw * PAPER_W_M) / KNOWN_DIST_M
                        intrinsics = {
                            "focal_length_px": round(f_y, 2),
                            "f_x": round(f_x, 2),
                            "f_y": round(f_y, 2),
                            "width": CAM_W,
                            "height": CAM_H,
                            "calibration_method": "a5_paper_1m",
                            "paper_w_mm": int(PAPER_W_M * 1000),
                            "paper_h_mm": int(PAPER_H_M * 1000),
                            "known_dist_m": KNOWN_DIST_M,
                        }
                        verify = (f_y * PAPER_H_M) / rh
                        logger.info("f_y=%.1f  f_x=%.1f  verify=%.3fm", f_y, f_x, verify)
                        state = "RESULT"

                elif key == 27:  # ESC
                    mouse.reset()
                    state = "LIVE"
                    logger.info("Retrying...")

            # ── RESULT: show summary, wait for ENTER to save or ESC retry ─
            elif state == "RESULT":
                vis = frozen.copy()
                x1, y1, x2, y2 = mouse.roi()
                rw, rh = x2 - x1, y2 - y1
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

                verify = (intrinsics["f_y"] * PAPER_H_M) / rh
                err_pct = abs(verify - KNOWN_DIST_M) / KNOWN_DIST_M * 100

                _overlay_text(vis, [
                    f"ROI: {rw} x {rh} px",
                    f"f_y = {intrinsics['f_y']:.1f} px",
                    f"f_x = {intrinsics['f_x']:.1f} px",
                    f"Verify: {verify:.3f}m  (err {err_pct:.1f}%)",
                    "ENTER = save   ESC = retry   Q = quit",
                ], color=(0, 255, 100))
                _show(vis)

                if key == 13:  # ENTER — save
                    with open(OUT_PATH, "w") as f:
                        json.dump(intrinsics, f, indent=2)
                    logger.info("Saved to %s", OUT_PATH)
                    logger.info("  f_y=%.1f px  f_x=%.1f px  verify=%.3fm",
                                intrinsics["f_y"], intrinsics["f_x"], verify)
                    break

                elif key == 27:  # ESC — retry
                    mouse.reset()
                    intrinsics = None
                    state = "LIVE"
                    logger.info("Retrying...")

    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
