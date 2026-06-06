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
from picamera2 import Picamera2

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
            put(vis, ["Hold A5 paper at 1.0 m", "SPACE=freeze  V=verify  Q=quit"])
            cx,cy = CAM_W//2, CAM_H//2
            cv2.line(vis,(cx-20,cy),(cx+20,cy),(0,255,0),1)
            cv2.line(vis,(cx,cy-20),(cx,cy+20),(0,255,0),1)

        elif state == "FROZEN":
            vis = frozen.copy()
            put(vis, ["Draw rect around paper", "ENTER=accept  ESC=retry  Q=quit"])
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
                "ENTER=save  ESC=retry  Q=quit",
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
                         "SPACE=recalibrate  Q=quit"], col=(0,200,255))
            else:
                put(vis,["No f_y saved yet — press ESC to calibrate first"])

        # ── Show + key ───────────────────────────────────────────────────
        show(vis)
        key = cv2.waitKey(30) & 0xFF

        if key == ord("q"):
            break

        # ── Transitions ──────────────────────────────────────────────────
        if state == "LIVE":
            if key == ord(" "):
                frozen = frame.copy(); reset_roi(); state = "FROZEN"
            elif key == ord("v") and fy:
                reset_roi(); state = "VERIFY"

        elif state == "FROZEN":
            if key == 13 and done:          # ENTER
                roi = get_roi()
                if roi:
                    rw = roi[2]-roi[0]; rh = roi[3]-roi[1]
                    state = "RESULT"
            elif key == 27:                 # ESC
                reset_roi(); state = "LIVE"

        elif state == "RESULT":
            if key == 13:                   # ENTER — save
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
            elif key == 27:                 # ESC
                reset_roi(); state = "LIVE"

        elif state == "VERIFY":
            if key == ord(" "):
                reset_roi(); state = "LIVE"

    cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
