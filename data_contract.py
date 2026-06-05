from dataclasses import dataclass, field
import numpy as np

@dataclass
class FrameResult:
    """Immutable data contract passed through the pipeline.
    
    This is the ONLY data structure shared between Track A and Track B.
    Any change to this schema must be agreed by BOTH tracks.
    """
    # ── Inputs (from camera) ──
    frame: np.ndarray                  # (H, W, 3) uint8 BGR
    timestamp: float                   # time.perf_counter()

    # ── Engine 1: YOLO outputs ──
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)   # (x1, y1, x2, y2) pixels
    bbox_height_px: float = 0.0                        # y2 - y1
    class_id: int = -1                                 # COCO class index
    class_name: str = ""                               # human-readable label
    det_confidence: float = 0.0                        # [0, 1]

    # ── Engine 2: Geometry outputs ──
    d_geometric_m: float = float("nan")                # pinhole estimate (metres)

    # ── Engine 3: Depth Anything V2 outputs ──
    rel_depth_score: float = float("nan")              # median relative depth in bbox
    depth_variance: float = float("nan")               # spatial variance inside bbox

    # ── Engine 4: Fusion MLP outputs ──
    final_distance_m: float = float("nan")             # fused metric distance (metres)
    log_variance: float = float("nan")                 # ln(σ²) for confidence interval
    confidence_68: tuple[float, float] = (0.0, 0.0)    # ±1σ interval (metres)
    confidence_95: tuple[float, float] = (0.0, 0.0)    # ±2σ interval (metres)
