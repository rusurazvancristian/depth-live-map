"""
Focal length calibration + geometry verification.

Hold an A5 sheet (150 x 205 mm) at exactly 1.0 m from the camera.

Controls
--------
SPACE   freeze frame
mouse   click-drag rectangle around the paper
ENTER   compute & save
ESC     retry
V       verify mode — show live distance using saved f_y
Q       quit
"""

import json, os, time
import cv2
import numpy as np
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
    logger = logging.getLogger(__name__)
    logger.warning("picamera2 not found. Initialising mock Picamera2 using system webcam / synthetic generator.")

    class Picamera2:
        """Mock Picamera2 implementation using OpenCV webcam or synthetic paper generator."""
        def __init__(self) -> None:
            self.cap = cv2.VideoCapture(0)
            self.dummy_frame = None

        def create_preview_configuration(self, **kwargs):
            return None

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
                    # OpenCV reads BGR, which matches BGR888 requested format
                    return frame
            # Generate a dummy frame with a simulated A5 sheet of paper at 1.0m
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Center of frame is (320, 240)
            # Paper height = 205mm, width = 150mm. f_y = 408px.
            # rh = (0.205 * 408) / 1.0 = 84 px
            # rw = (0.150 * 408) / 1.0 = 61 px
            x1, y1 = 320 - 30, 240 - 42
            x2, y2 = 320 + 30, 240 + 42
            cv2.rectangle(frame, (x1, y1), (x2, y2), (240, 240, 240), -1)
            cv2.putText(frame, "MOCK A5 PAPER", (x1 + 5, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (10, 10, 10), 1)
            return frame

        def stop(self) -> None:
            if self.cap.isOpened():
                self.cap.release()

# ── Known target ──────────────────────────────────────────────────────────────
PAPER_W_M    = 0.150
PAPER_H_M    = 0.205
DIST_M       = 1.0
CAM_W, CAM_H = 640, 480
OUT          = "src/calibration/intrinsics.json"
WIN          = "Calibrate"

# ── Helpers ───────────────────────────────────────────────────────────────────
def open_cam():
    cam = Picamera2()
    if PICAMERA2_AVAILABLE:
        cam.configure(cam.create_preview_configuration(
            main={"size": (CAM_W, CAM_H), "format": "BGR888"},
            controls={"FrameRate": 30, "AfMode": 0, "LensPosition": 0.0},
        ))
    cam.start(); time.sleep(1.0)
    return cam

def show(img):
    cv2.imshow(WIN, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def put(img, lines, y0=28, col=(0, 200, 255)):
    for i, t in enumerate(lines):
        y = y0 + i * 26
        cv2.putText(img, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(img, t, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col,   1, cv2.LINE_AA)

# ── Mouse ROI ─────────────────────────────────────────────────────────────────
pt1 = pt2 = None
dragging = done = False

def on_mouse(ev, x, y, *_):
    global pt1, pt2, dragging, done
    if ev == cv2.EVENT_LBUTTONDOWN:
        pt1 = pt2 = (x, y); dragging = True; done = False
    elif ev == cv2.EVENT_MOUSEMOVE and dragging:
        pt2 = (x, y)
    elif ev == cv2.EVENT_LBUTTONUP:
        pt2 = (x, y); dragging = False; done = True

def get_roi():
    if pt1 is None or pt2 is None: return None
    x1, y1 = min(pt1[0],pt2[0]), min(pt1[1],pt2[1])
    x2, y2 = max(pt1[0],pt2[0]), max(pt1[1],pt2[1])
    return (x1,y1,x2,y2) if (x2-x1)>10 and (y2-y1)>10 else None

def reset_roi():
    global pt1, pt2, dragging, done
    pt1 = pt2 = None; dragging = done = False

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    cam = open_cam()
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, CAM_W, CAM_H)
    cv2.setMouseCallback(WIN, on_mouse)

    state  = "LIVE"     # LIVE | FROZEN | RESULT | VERIFY
    frozen = None
    fy     = None

    # Load existing f_y if available
    if os.path.exists(OUT):
        try:
            fy = json.load(open(OUT))["focal_length_px"]
            print(f"Loaded existing f_y={fy:.1f} from {OUT}")
        except Exception:
            pass

    while True:
        # ── Build vis ────────────────────────────────────────────────────
        if state == "LIVE":
            frame = cam.capture_array()
            vis = frame.copy()
            put(vis, ["Hold A5 paper at 1.0 m", "SPACE / F = freeze  V = verify  Q = quit"])
            cx,cy = CAM_W//2, CAM_H//2
            cv2.line(vis,(cx-20,cy),(cx+20,cy),(0,255,0),1)
            cv2.line(vis,(cx,cy-20),(cx,cy+20),(0,255,0),1)

        elif state == "FROZEN":
            vis = frozen.copy()
            put(vis, ["Draw rect around paper", "ENTER / A = accept  ESC / R = retry  Q = quit"])
            roi = get_roi()
            if roi:
                cv2.rectangle(vis, roi[:2], roi[2:], (0,255,0), 2)

        elif state == "RESULT":
            vis = frozen.copy()
            roi = get_roi()
            cv2.rectangle(vis, roi[:2], roi[2:], (0,255,0), 2)
            rw = roi[2]-roi[0]; rh = roi[3]-roi[1]
            fx_ = round(rw*PAPER_W_M/DIST_M, 1)
            fy_ = round(rh*PAPER_H_M/DIST_M, 1)
            ver = fy_*PAPER_H_M/rh
            err = abs(ver-DIST_M)/DIST_M*100
            put(vis, [
                f"ROI  {rw} x {rh} px",
                f"f_x = {fx_:.1f} px",
                f"f_y = {fy_:.1f} px",
                f"Verify: {ver:.3f} m  (err {err:.1f}%)",
                "ENTER / S = save  ESC / R = retry  Q = quit",
            ], col=(80,255,80))

        elif state == "VERIFY":
            frame = cam.capture_array()
            vis = frame.copy()
            if fy:
                # Draw a test bbox guide (centre 40% of frame)
                gx1,gy1 = int(CAM_W*0.3), int(CAM_H*0.2)
                gx2,gy2 = int(CAM_W*0.7), int(CAM_H*0.8)
                cv2.rectangle(vis,(gx1,gy1),(gx2,gy2),(255,100,0),1)
                roi_v = get_roi()
                if roi_v:
                    cv2.rectangle(vis, roi_v[:2], roi_v[2:], (0,255,0), 2)
                    rh_v = roi_v[3]-roi_v[1]
                    if rh_v > 5:
                        d = fy*PAPER_H_M/rh_v
                        put(vis,[f"d = {d:.3f} m  (f_y={fy:.0f})"],
                            y0=CAM_H-36, col=(80,255,80))
                put(vis,["VERIFY: draw rect on paper -> see distance",
                         "SPACE / F = recalibrate  Q = quit"], col=(0,200,255))
            else:
                put(vis,["No f_y saved yet — press ESC / R to calibrate first"])

        # ── Show + key ───────────────────────────────────────────────────
        show(vis)
        key = cv2.waitKey(30) & 0xFF

        if key != 255:
            # Safe character representation printing
            print(f"Key pressed: {key} ('{chr(key) if 32 <= key < 127 else 'special'}')")

        if key in (ord("q"), ord("Q")):
            break

        # ── Transitions ──────────────────────────────────────────────────
        if state == "LIVE":
            if key in (ord(" "), ord("f"), ord("F")):
                frozen = frame.copy(); reset_roi(); state = "FROZEN"
                print("Frozen. Draw rectangle around the paper.")
            elif key in (ord("v"), ord("V")) and fy:
                reset_roi(); state = "VERIFY"

        elif state == "FROZEN":
            if key in (13, 10, ord("a"), ord("A")) and done:          # ENTER or 'a'
                roi = get_roi()
                if roi:
                    rw = roi[2]-roi[0]; rh = roi[3]-roi[1]
                    state = "RESULT"
            elif key in (27, ord("r"), ord("R")):                 # ESC or 'r'
                reset_roi(); state = "LIVE"

        elif state == "RESULT":
            if key in (13, 10, ord("s"), ord("S")):                   # ENTER or 's'
                roi = get_roi()
                rw = roi[2]-roi[0]; rh = roi[3]-roi[1]
                fy_ = round(rh*PAPER_H_M/DIST_M, 1)
                fx_ = round(rw*PAPER_W_M/DIST_M, 1)
                data = {"focal_length_px": fy_, "f_x": fx_, "f_y": fy_,
                        "width": CAM_W, "height": CAM_H,
                        "calibration_method": "a5_paper_1m"}
                os.makedirs(os.path.dirname(OUT), exist_ok=True)
                json.dump(data, open(OUT,"w"), indent=2)
                fy = fy_
                print(f"Saved  f_y={fy_}  f_x={fx_}  →  {OUT}")
                reset_roi(); state = "VERIFY"
            elif key in (27, ord("r"), ord("R")):                 # ESC or 'r'
                reset_roi(); state = "LIVE"

        elif state == "VERIFY":
            if key in (ord(" "), ord("f"), ord("F")):
                reset_roi(); state = "LIVE"

    cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
