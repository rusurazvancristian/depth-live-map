"""Distance calibration using a known A5 sheet (150×205mm) at 1.0m.

Usage:
    python3 src/calibration/calibrate_distance.py

Workflow:
    1. Hold the paper flat at exactly 1.0m from the camera.
    2. Press SPACE to freeze the frame.
    3. Draw a rectangle around the paper with the mouse.
    4. Press ENTER to accept and save, ESC to retry.
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

# ── Known calibration target ──────────────────────────────────────────────────
PAPER_W_M  = 0.150   # 150 mm
PAPER_H_M  = 0.205   # 205 mm
KNOWN_DIST_M = 1.0   # held at exactly 1 metre

CAM_W, CAM_H = 640, 480
OUT_PATH = os.path.join(os.path.dirname(__file__), "intrinsics.json")
WINDOW = "Calibration — SPACE=freeze  ENTER=save  ESC=retry"


def _open_camera() -> Picamera2:
    cam = Picamera2()
    cfg = cam.create_preview_configuration(
        main={"size": (CAM_W, CAM_H), "format": "BGR888"},
        controls={"FrameRate": 30, "AfMode": 0, "LensPosition": 0.0},
    )
    cam.configure(cfg)
    cam.start()
    time.sleep(1.0)
    return cam


def _draw_guide(frame: np.ndarray) -> np.ndarray:
    vis = frame.copy()
    h, w = vis.shape[:2]
    cv2.putText(vis, "Hold A5 paper at 1.0m — press SPACE to freeze",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2, cv2.LINE_AA)
    # Centre cross
    cx, cy = w // 2, h // 2
    cv2.line(vis, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 1)
    cv2.line(vis, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 1)
    return vis


def _select_roi(frozen: np.ndarray):
    """Let user draw rectangle; returns (x, y, w, h) or None on cancel."""
    cv2.putText(frozen, "Draw rectangle around paper — ENTER=ok  ESC=retry",
                (10, CAM_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
    roi = cv2.selectROI(WINDOW, frozen, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("ROI selector")
    x, y, rw, rh = roi
    if rw < 10 or rh < 10:
        return None
    return x, y, rw, rh


def _compute_intrinsics(roi_w_px: int, roi_h_px: int) -> dict:
    f_y = (roi_h_px * PAPER_H_M) / KNOWN_DIST_M
    f_x = (roi_w_px * PAPER_W_M) / KNOWN_DIST_M
    return {
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


def _verify_distance(intrinsics: dict, roi_h_px: int) -> float:
    """Cross-check: recalculate distance with the saved f_y."""
    return (intrinsics["f_y"] * PAPER_H_M) / roi_h_px


def main() -> None:
    cam = _open_camera()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, CAM_W, CAM_H)
    logger.info("Camera ready. Hold A5 paper at 1.0m and press SPACE.")

    try:
        while True:
            # ── Live preview ──────────────────────────────────────────────
            frame = cam.capture_array()
            cv2.imshow(WINDOW, cv2.cvtColor(_draw_guide(frame), cv2.COLOR_BGR2RGB))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("Aborted.")
                return
            if key != ord(" "):
                continue

            # ── Frame frozen — select ROI ─────────────────────────────────
            frozen = frame.copy()
            logger.info("Frame frozen. Draw rectangle around the paper.")

            roi = _select_roi(cv2.cvtColor(frozen, cv2.COLOR_BGR2RGB))
            if roi is None:
                logger.warning("Selection too small or cancelled — retrying.")
                continue

            x, y, rw, rh = roi
            intrinsics = _compute_intrinsics(rw, rh)
            verified_dist = _verify_distance(intrinsics, rh)

            # ── Result overlay ────────────────────────────────────────────
            result_vis = cv2.cvtColor(frozen, cv2.COLOR_BGR2RGB).copy()
            cv2.rectangle(result_vis, (x, y), (x + rw, y + rh), (0, 255, 0), 2)

            lines = [
                f"ROI:  {rw} x {rh} px",
                f"f_y:  {intrinsics['f_y']:.1f} px",
                f"f_x:  {intrinsics['f_x']:.1f} px",
                f"Verify dist: {verified_dist:.3f} m  (expected 1.000 m)",
                "",
                "ENTER = save    ESC = retry",
            ]
            for i, line in enumerate(lines):
                color = (0, 255, 100) if "1.0" in line else (0, 220, 255)
                cv2.putText(result_vis, line, (10, 50 + i * 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

            cv2.imshow(WINDOW, result_vis)
            logger.info("f_y=%.1f  f_x=%.1f  verify=%.3fm", intrinsics["f_y"], intrinsics["f_x"], verified_dist)

            # ── Save or retry ─────────────────────────────────────────────
            while True:
                k = cv2.waitKey(0) & 0xFF
                if k == 13:  # ENTER
                    with open(OUT_PATH, "w") as f:
                        json.dump(intrinsics, f, indent=2)
                    logger.info("Saved intrinsics to %s", OUT_PATH)
                    logger.info("  f_y = %.1f px  |  f_x = %.1f px", intrinsics["f_y"], intrinsics["f_x"])
                    return
                if k == 27:  # ESC
                    logger.info("Retrying...")
                    break

    finally:
        cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
