import cv2
import numpy as np

def letterbox_resize(
    frame: np.ndarray,
    target_size: int = 640,
    pad_color: int = 114,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize image with letterbox padding to maintain aspect ratio.

    Args:
        frame: Input BGR image, shape (H, W, 3).
        target_size: Target square dimension in pixels.
        pad_color: Color intensity for padded boundaries.

    Returns:
        Tuple of (padded_image, scale_factor, (pad_w, pad_h)).
    """
    h, w = frame.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2
    padded = cv2.copyMakeBorder(
        resized, pad_h, target_size - new_h - pad_h,
        pad_w, target_size - new_w - pad_w,
        cv2.BORDER_CONSTANT, value=(pad_color, pad_color, pad_color),
    )
    return padded, scale, (pad_w, pad_h)


def normalize_depth(depth_map: np.ndarray) -> np.ndarray:
    """Min-max normalize depth to [0, 1]. Handle constant maps gracefully.
    
    Args:
        depth_map: Raw relative/metric depth map array.
        
    Returns:
        Normalized depth map array of same shape.
    """
    lo, hi = depth_map.min(), depth_map.max()
    if hi - lo < 1e-6:
        return np.full_like(depth_map, 0.5)
    return (depth_map - lo) / (hi - lo)
