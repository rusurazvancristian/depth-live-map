"""Input/Output VStream preprocessing helpers. [TRACK A]"""

import cv2
import numpy as np
from typing import Tuple, Optional


def letterbox_resize(
    frame: np.ndarray,
    target_size: int = 640,
    pad_color: int = 114,
    dst_bgr: Optional[np.ndarray] = None,
    dst_rgb: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize image with letterbox padding to maintain aspect ratio.

    Supports optional pre-allocated destination buffers to avoid memory re-allocations.

    Args:
        frame: Input BGR image, shape (H, W, 3).
        target_size: Target square dimension in pixels.
        pad_color: Grayscale value used for padding borders.
        dst_bgr: Pre-allocated destination array for BGR resize, shape (new_h, new_w, 3).
        dst_rgb: Pre-allocated destination array for RGB output, shape (target_size, target_size, 3).

    Returns:
        Tuple of (padded_rgb_image, scale_factor, (pad_w, pad_h)).
        padded_rgb_image is uint8 RGB, shape (target_size, target_size, 3).
    """
    h, w = frame.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2

    if dst_rgb is not None:
        # Fill padding borders in preallocated buffer
        dst_rgb.fill(pad_color)
        dst_slice = dst_rgb[pad_h : pad_h + new_h, pad_w : pad_w + new_w]
        
        if dst_bgr is not None and dst_bgr.shape[:2] == (new_h, new_w):
            cv2.resize(frame, (new_w, new_h), dst=dst_bgr, interpolation=cv2.INTER_LINEAR)
            cv2.cvtColor(dst_bgr, cv2.COLOR_BGR2RGB, dst=dst_slice)
        else:
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            cv2.cvtColor(resized, cv2.COLOR_BGR2RGB, dst=dst_slice)
            
        return dst_rgb, scale, (pad_w, pad_h)

    # Slow fallback path (allocates memory)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2
    padded = cv2.copyMakeBorder(
        resized,
        pad_h, target_size - new_h - pad_h,
        pad_w, target_size - new_w - pad_w,
        cv2.BORDER_CONSTANT,
        value=(pad_color, pad_color, pad_color),
    )
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    return rgb, scale, (pad_w, pad_h)


def unletterbox_bbox(
    x1n: float, y1n: float, x2n: float, y2n: float,
    scale: float,
    pad: Tuple[int, int],
    orig_w: int,
    orig_h: int,
    model_size: int = 640,
) -> Tuple[int, int, int, int]:
    """Convert normalised model-space bbox back to original frame pixel coordinates.

    Args:
        x1n, y1n, x2n, y2n: Normalised [0-1] bbox from YOLO output.
        scale: Scale factor used in letterbox_resize.
        pad: (pad_w, pad_h) padding applied in letterbox_resize.
        orig_w, orig_h: Original frame dimensions.
        model_size: Model input square size (default 640).

    Returns:
        (x1, y1, x2, y2) in original frame pixel coordinates, clipped to frame bounds.
    """
    pad_w, pad_h = pad
    x1 = int((x1n * model_size - pad_w) / scale)
    y1 = int((y1n * model_size - pad_h) / scale)
    x2 = int((x2n * model_size - pad_w) / scale)
    y2 = int((y2n * model_size - pad_h) / scale)
    x1 = max(0, min(x1, orig_w - 1))
    y1 = max(0, min(y1, orig_h - 1))
    x2 = max(0, min(x2, orig_w - 1))
    y2 = max(0, min(y2, orig_h - 1))
    return x1, y1, x2, y2


def to_nhwc_batch(rgb_image: np.ndarray) -> np.ndarray:
    """Wrap a single HWC image into a NHWC batch of size 1.

    Args:
        rgb_image: uint8 RGB array, shape (H, W, 3).

    Returns:
        uint8 NHWC array, shape (1, H, W, 3).
    """
    return np.expand_dims(rgb_image, axis=0)
