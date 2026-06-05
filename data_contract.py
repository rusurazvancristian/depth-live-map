"""Shared data contract — FrameResult dataclass.

FROZEN: Any change requires agreement from both Track A and Track B.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class FrameResult:
    """Immutable data contract passed through the 4-engine pipeline.

    Ownership:
        Engine 1 (YOLO)     — bbox, bbox_height_px, class_id, class_name, det_confidence
        Engine 2 (Geometry) — d_geometric_m
        Engine 3 (Depth)    — rel_depth_score, depth_variance
        Engine 4 (Fusion)   — final_distance_m, log_variance, confidence_68, confidence_95

    CRITICAL: Engines must NEVER modify fields owned by another engine.
    """

    # ── Inputs (from camera) ──────────────────────────────────────────────────
    frame: np.ndarray
    timestamp: float

    # ── Engine 1: YOLO outputs ───────────────────────────────────────────────
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    bbox_height_px: float = 0.0
    class_id: int = -1
    class_name: str = ""
    det_confidence: float = 0.0

    # ── Engine 2: Geometry outputs ───────────────────────────────────────────
    d_geometric_m: float = float("nan")

    # ── Engine 3: Depth Anything V2 outputs ─────────────────────────────────
    rel_depth_score: float = float("nan")
    depth_variance: float = float("nan")

    # ── Engine 4: Fusion MLP outputs ────────────────────────────────────────
    final_distance_m: float = float("nan")
    log_variance: float = float("nan")
    confidence_68: tuple[float, float] = (0.0, 0.0)
    confidence_95: tuple[float, float] = (0.0, 0.0)
