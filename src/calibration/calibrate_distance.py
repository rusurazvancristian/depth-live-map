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
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

if not PICAMERA2_AVAILABLE:
    logger.warning("picamera2 not found. Initialising mock Picamera2 using system webcam / synthetic generator.")

    class Picamera2:
        """Mock Picamera2 implementation using OpenCV webcam or synthetic paper generator."""
        def __init__(self) -> None:
            self.cap = cv2.VideoCapture(0)
            self.dummy_frame = None

        def configure(self, cam_config) -> None:
            pass

        def start(self) -> None:
            if not self.cap.isOpened():
                logger.warning("Could not open system webcam. Generating synthetic test frames.")
                self.dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        def capture_array(self) -> np.ndarray:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    # OpenCV reads BGR, we return RGB to match Picamera2 output
                    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Generate a dummy frame with a simulated A5 sheet of paper at 1.0m
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Center of frame is (320, 240)
            # For 1.0m distance, size of paper in pixels is determined by camera matrix.
            # E.g., paper height = 205mm, width = 150mm.
            # If f_y = 600, rh = (height_m * f_y) / distance = (0.205 * 600) / 1.0 = 123 pixels
            # rw = (width_m * f_x) / distance = (0.150 * 600) / 1.0 = 90 pixels
            x1, y1 = 320 - 45, 240 - 61
            x2, y2 = 320 + 45, 240 + 61
            cv2.rectangle(frame, (x1, y1), (x2, y2), (240, 240, 240), -1)
            cv2.putText(frame, "MOCK A5 PAPER", (x1 + 5, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (10, 10, 10), 1)
            return frame

        def stop(self) -> None:
            if self.cap.isOpened():
                self.cap.release()

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
    if PICAMERA2_AVAILABLE:
        cfg = cam.create_preview_configuration(
            main={"size": (CAM_W, CAM_H), "format": "RGB888"},
            controls={"FrameRate": 30, "AfMode": 0, "LensPosition": 0.0},
        )
        cam.configure(cfg)
    cam.start()
    time.sleep(1.0)
    return cam


def _overlay_text(img, lines, y0=30, color=(255, 220, 0)):
    for i, line in enumerate(lines):
        cv2.putText(img, line, (10, y0 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (10, y0 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)


def _show(img):
    """Display RGB image — RGB888 camera format, no conversion needed."""
    cv2.imshow(WINDOW, img)


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
            # ── Build frame to display ────────────────────────────────────
            if state == "LIVE":
                frame = cam.capture_array()
                vis = frame.copy()
                _overlay_text(vis, ["Hold A5 paper at 1.0m",
                                    "SPACE / F = freeze   Q = quit"])
                cx, cy = CAM_W // 2, CAM_H // 2
                cv2.line(vis, (cx-20, cy), (cx+20, cy), (0, 255, 0), 1)
                cv2.line(vis, (cx, cy-20), (cx, cy+20), (0, 255, 0), 1)

            elif state == "FROZEN":
                vis = frozen.copy()
                _overlay_text(vis, ["Draw rectangle around paper",
                                    "ENTER / A = accept   ESC / R = retry   Q = quit"])
                roi = mouse.roi()
                if roi:
                    x1, y1, x2, y2 = roi
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

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
                    f"Verify: {verify:.3f} m  (err {err_pct:.1f}%)",
                    "ENTER / S = save   ESC / R = retry   Q = quit",
                ], color=(100, 255, 100))

            # ── Display first, then collect key ───────────────────────────
            _show(vis)
            key = cv2.waitKey(30) & 0xFF  # 30ms: pumps GUI events reliably

            if key != 255:
                logger.info("Key pressed: %d ('%s')", key, chr(key) if 32 <= key < 127 else "special")

            # ── Global keys ───────────────────────────────────────────────
            if key in (ord("q"), ord("Q")):
                logger.info("Quit without saving.")
                break

            # ── State transitions ─────────────────────────────────────────
            if state == "LIVE":
                if key in (ord(" "), ord("f"), ord("F")):
                    frozen = frame.copy()
                    mouse.reset()
                    state = "FROZEN"
                    logger.info("Frozen. Draw rectangle around the paper.")

            elif state == "FROZEN":
                if key in (13, 10, ord("a"), ord("A")) and mouse.done:  # ENTER or 'a'
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
                        logger.info("f_y=%.1f  f_x=%.1f", f_y, f_x)
                        state = "RESULT"
                elif key in (27, ord("r"), ord("R")):  # ESC or 'r'
                    mouse.reset()
                    state = "LIVE"
                    logger.info("Retrying...")

            elif state == "RESULT":
                if key in (13, 10, ord("s"), ord("S")):  # ENTER or 's'
                    with open(OUT_PATH, "w") as f:
                        json.dump(intrinsics, f, indent=2)
                    logger.info("Saved → %s  (f_y=%.1f px)", OUT_PATH, intrinsics["f_y"])
                    break
                elif key in (27, ord("r"), ord("R")):  # ESC or 'r'
                    mouse.reset()
                    intrinsics = None
                    state = "LIVE"
                    logger.info("Retrying...")

    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
