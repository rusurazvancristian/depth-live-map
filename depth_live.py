#!/usr/bin/env python3
"""
Live depth-estimation map — Camera Module 3 + Hailo-8 (26 TOPS)
fastdepth model: 224x224 input → 224x224 float32 depth map

Controls:
  Q / Esc  — quit
  C        — cycle colormaps
  I        — toggle invert depth
  F        — toggle fullscreen
  S        — save screenshot
"""

import os
import sys
import time
import argparse
import signal
import numpy as np
import cv2
from picamera2 import Picamera2
from hailo_platform import (
    VDevice, HEF, ConfigureParams, InferVStreams,
    InputVStreamParams, OutputVStreamParams,
    FormatType, HailoStreamInterface,
)

MODEL_PATH  = "/usr/share/hailo-models/fast_depth_h8.hef"
CAM_W, CAM_H = 640, 480
MODEL_SIZE  = 224
DISP_H      = 480
WIN_TITLE   = "Depth Live — Hailo-8 | Q=quit  C=colormap  I=invert  S=save"

COLORMAPS = [
    ("Magma",    cv2.COLORMAP_MAGMA),
    ("Turbo",    cv2.COLORMAP_TURBO),
    ("Inferno",  cv2.COLORMAP_INFERNO),
    ("Jet",      cv2.COLORMAP_JET),
    ("Hot",      cv2.COLORMAP_HOT),
    ("Viridis",  cv2.COLORMAP_VIRIDIS),
]


# ── camera ────────────────────────────────────────────────────────────────────

def init_camera() -> Picamera2:
    cam = Picamera2()
    cfg = cam.create_preview_configuration(
        main={"size": (CAM_W, CAM_H), "format": "RGB888"},
        controls={"FrameRate": 30},
    )
    cam.configure(cfg)
    cam.start()
    time.sleep(0.8)
    return cam


# ── hailo ─────────────────────────────────────────────────────────────────────

def init_hailo(model_path: str):
    hef = HEF(model_path)
    target = VDevice()
    params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_groups = target.configure(hef, params)
    ng = network_groups[0]
    ng_params = ng.create_params()

    inp_name = hef.get_input_vstream_infos()[0].name
    out_name = hef.get_output_vstream_infos()[0].name

    in_p  = InputVStreamParams.make_from_network_group(ng, quantized=False, format_type=FormatType.UINT8)
    out_p = OutputVStreamParams.make_from_network_group(ng, quantized=False, format_type=FormatType.FLOAT32)

    # hef and target must stay alive for the lifetime of ng/pipeline
    return hef, target, ng, ng_params, in_p, out_p, inp_name, out_name


# ── depth colorisation ────────────────────────────────────────────────────────

def depth_to_color(depth: np.ndarray, cmap_id: int, invert: bool) -> np.ndarray:
    """Normalize float32 depth map → BGR uint8 colourmap image."""
    d = depth.squeeze()
    lo, hi = d.min(), d.max()
    if hi > lo:
        norm = ((d - lo) / (hi - lo) * 255).astype(np.uint8)
    else:
        norm = np.full(d.shape, 128, dtype=np.uint8)
    if invert:
        norm = 255 - norm
    return cv2.applyColorMap(norm, cmap_id)   # BGR


def make_scalebar(height: int, width: int, cmap_id: int, invert: bool) -> np.ndarray:
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    grad = np.arange(height, dtype=np.uint8).reshape(-1, 1)
    if not invert:
        grad = 255 - grad
    colored = cv2.applyColorMap(grad, cmap_id)   # (H,1,3)
    bar[:] = np.repeat(colored, width, axis=1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(bar, "near", (2, 18),         font, 0.4, (255, 255, 255), 1)
    cv2.putText(bar, "far",  (2, height - 6), font, 0.4, (255, 255, 255), 1)
    return bar


def overlay_crosshair(img: np.ndarray, depth: np.ndarray) -> None:
    """Draw a crosshair at centre and print the relative depth %."""
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.drawMarker(img, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
    d = depth.squeeze()
    lo, hi = d.min(), d.max()
    dy, dx = d.shape
    px = d[int(dy * cy / h), int(dx * cx / w)]
    rel = (px - lo) / (hi - lo + 1e-8) * 100
    cv2.putText(img, f"{rel:.0f}%", (cx + 12, cy - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",    default=MODEL_PATH)
    ap.add_argument("--invert",   action="store_true", help="Invert depth (swap near/far)")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--cmap",     type=int, default=0, help="Colormap index 0-5")
    ap.add_argument("--no-camera", action="store_true", help="Hide camera pane, show depth only")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[init] Camera Module 3 …")
    cam = init_camera()

    print(f"[init] Hailo-8  model={args.model}")
    hef, target, ng, ng_params, in_p, out_p, inp_name, out_name = init_hailo(args.model)

    invert   = args.invert
    cmap_idx = args.cmap % len(COLORMAPS)
    SCALEBAR_W = 28
    DISP_W     = DISP_H  # square panels

    shot_idx = 0
    fps = 0.0
    t_prev = time.perf_counter()

    cv2.namedWindow(WIN_TITLE, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(WIN_TITLE, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # graceful exit on SIGINT
    running = [True]
    signal.signal(signal.SIGINT, lambda *_: running.__setitem__(0, False))

    print("[run]  Press Q in the window to quit.")

    with InferVStreams(ng, in_p, out_p) as pipeline:
        with ng.activate(ng_params):
            while running[0]:
                # ── capture ──────────────────────────────────────────
                frame = cam.capture_array()              # (480,640,3) uint8

                # ── preprocess ───────────────────────────────────────
                inp = cv2.resize(frame, (MODEL_SIZE, MODEL_SIZE))
                batch = np.expand_dims(inp, 0)           # (1,224,224,3)

                # ── inference ────────────────────────────────────────
                result = pipeline.infer({inp_name: batch})
                depth = result[out_name][0]              # (224,224,1) float32

                # ── colourize ────────────────────────────────────────
                cmap_name, cmap_id = COLORMAPS[cmap_idx]
                depth_bgr = depth_to_color(depth, cmap_id, invert)
                depth_disp = cv2.resize(depth_bgr, (DISP_W, DISP_H))
                overlay_crosshair(depth_disp, depth)

                # ── fps ──────────────────────────────────────────────
                t_now = time.perf_counter()
                fps = 0.85 * fps + 0.15 / max(t_now - t_prev, 1e-6)
                t_prev = t_now

                # ── annotations on depth pane ────────────────────────
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(depth_disp,
                            f"Hailo-8  {fps:.1f} FPS  [{cmap_name}]{'  INV' if invert else ''}",
                            (8, 26), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

                scalebar = make_scalebar(DISP_H, SCALEBAR_W, cmap_id, invert)

                if args.no_camera:
                    canvas = np.hstack([depth_disp, scalebar])
                else:
                    cam_disp = cv2.resize(frame, (DISP_W, DISP_H))
                    cv2.putText(cam_disp, "Camera Module 3",
                                (8, 26), font, 0.55, (0, 255, 100), 1, cv2.LINE_AA)
                    canvas = np.hstack([cam_disp, depth_disp, scalebar])

                cv2.imshow(WIN_TITLE, canvas)

                # ── keyboard ─────────────────────────────────────────
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):          # Q or Esc
                    break
                elif key == ord('c'):
                    cmap_idx = (cmap_idx + 1) % len(COLORMAPS)
                elif key == ord('i'):
                    invert = not invert
                elif key == ord('f'):
                    fs = cv2.getWindowProperty(WIN_TITLE, cv2.WND_PROP_FULLSCREEN)
                    cv2.setWindowProperty(WIN_TITLE, cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_NORMAL if fs else cv2.WINDOW_FULLSCREEN)
                elif key == ord('s'):
                    fname = f"depth_shot_{shot_idx:04d}.png"
                    cv2.imwrite(fname, canvas)
                    print(f"[save] {fname}")
                    shot_idx += 1

    cv2.destroyAllWindows()
    cam.stop()
    print("[done]")


if __name__ == "__main__":
    main()
