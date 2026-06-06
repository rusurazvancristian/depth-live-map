"""Shared data contract — FrameResult dataclass.

Covers the full 5-stage pipeline:
  YOLO → ByteTrack → Geometry → SCDepthV3 → KalmanDepth
Plus target lock status for the UI overlay.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class Detection:
    """A single YOLO detection in original frame coordinates.

    Attributes:
        bbox: (x1, y1, x2, y2) in pixel coordinates.
        confidence: detection confidence score.
        class_id: COCO class index.
        class_name: human-readable class name.
    """
    bbox: tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str


@dataclass
class TrackedObject:
    """A ByteTrack-assigned tracked object.

    Attributes:
        track_id: unique temporal ID assigned by ByteTrack.
        bbox: (x1, y1, x2, y2) current estimated position.
        confidence: last detection confidence.
        class_id: COCO class index.
        class_name: human-readable class name.
        bbox_height_px: height in pixels for geometric distance.
        d_geometric_m: pinhole-model metric distance.
        rel_depth_score: median normalised depth from SCDepthV3 ROI.
        kalman_distance_m: Kalman-filtered metric distance.
        kalman_variance: Kalman covariance P[0,0].
    """
    track_id: int = -1
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    confidence: float = 0.0
    class_id: int = -1
    class_name: str = ""
    bbox_height_px: float = 0.0
    d_geometric_m: float = float("nan")
    rel_depth_score: float = float("nan")
    kalman_distance_m: float = float("nan")
    kalman_variance: float = float("nan")


@dataclass
class FrameResult:
    """Top-level per-frame data contract passed through the pipeline.

    Ownership:
        Stage 1 (YOLO)      — detections
        Stage 2 (ByteTrack) — tracked_objects
        Stage 3 (Geometry)  — tracked_objects[i].d_geometric_m
        Stage 4 (Depth)     — tracked_objects[i].rel_depth_score, depth_map
        Stage 5 (Kalman)    — tracked_objects[i].kalman_distance_m/variance

    CRITICAL: Stages must NEVER modify fields owned by another stage.
    """

    # ── Inputs (from camera) ──────────────────────────────────────────────────
    frame: np.ndarray = field(default_factory=lambda: np.empty(0))
    timestamp: float = 0.0

    # ── Stage 1: YOLO outputs (raw multi-detection) ──────────────────────────
    detections: List[Detection] = field(default_factory=list)

    # ── Stage 2: ByteTrack outputs (tracked objects with IDs) ────────────────
    tracked_objects: List[TrackedObject] = field(default_factory=list)

    # ── Stage 4: Full depth map from SCDepthV3 (normalised 0-1) ──────────────
    depth_map: Optional[np.ndarray] = None

    # ── Target Lock Status ───────────────────────────────────────────────────
    target_id: int = -1               # ByteTrack ID of the locked target
    target_status: str = "IDLE"       # IDLE | LOCKED | SEARCHING | LOST
    target_distance_m: float = float("nan")
    target_is_arrived: bool = False
