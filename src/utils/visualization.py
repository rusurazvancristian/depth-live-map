import cv2
import numpy as np
import math
from data_contract import FrameResult

COLORMAPS = [
    ("Magma",    cv2.COLORMAP_MAGMA),
    ("Turbo",    cv2.COLORMAP_TURBO),
    ("Inferno",  cv2.COLORMAP_INFERNO),
    ("Jet",      cv2.COLORMAP_JET),
    ("Hot",      cv2.COLORMAP_HOT),
    ("Viridis",  cv2.COLORMAP_VIRIDIS),
]

def depth_to_color(depth: np.ndarray, cmap_id: int, invert: bool) -> np.ndarray:
    """Normalize float32 depth map -> BGR uint8 colormap image."""
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
    """Creates a vertical colormap scalebar overlay."""
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


def draw_distance_overlay(frame: np.ndarray, result: FrameResult) -> np.ndarray:
    """Draw bbox, distance, and confidence on the camera frame."""
    vis = frame.copy()
    
    if result.class_id < 0:
        cv2.putText(vis, "No detection", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return vis

    x1, y1, x2, y2 = result.bbox
    color = (0, 255, 0)

    # Bounding box
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

    # Distance label
    if not math.isnan(result.final_distance_m):
        dist_text = f"{result.final_distance_m:.1f}m"
        lo, hi = result.confidence_95
        conf_text = f"[{lo:.1f}-{hi:.1f}m]"
    else:
        dist_text = "?.?m"
        conf_text = ""

    label = f"{result.class_name} {dist_text} {conf_text}"
    cv2.putText(vis, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    return vis
