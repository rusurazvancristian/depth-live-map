"""Centralised configuration — all constants, paths, tunables.

Covers: 3-model NPU pipeline (YOLO26s + SCDepthV3 + RepVGG-A0 ReID),
        ByteTrack multi-object tracking, Kalman depth fusion,
        exemplar-based target lock, and arrival trigger.
"""

from dataclasses import dataclass, field
from typing import Dict


# ── Model registry for auto-download ─────────────────────────────────────────
MODEL_REGISTRY: Dict[str, str] = {
    "yolo26m.hef": (
        "https://github.com/DanielDubinsky/yolo26_hailo/releases/latest/download/yolo26m.hef"
    ),
    "scdepthv3.hef": (
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/"
        "ModelZoo/Compiled/v2.18.0/hailo8/scdepthv3.hef"
    ),
    "repvgg_a0_person_reid_512.hef": (
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/"
        "ModelZoo/Compiled/v2.14.0/hailo8/repvgg_a0_person_reid_512.hef"
    ),
}


@dataclass(frozen=True)
class Config:
    """Immutable configuration. Loaded once at startup.

    IMPORTANT: Always run camera calibration and update focal_length_px.
    The default is an estimate for Camera Module 3 at 640x480.
    """

    # ── Camera ───────────────────────────────────────────────────────────────
    cam_width: int = 640
    cam_height: int = 480
    cam_fps: int = 30

    # ── Intrinsics (update from calibration/intrinsics.json after calibrating) ─
    focal_length_px: float = 600.0
    principal_point: tuple[float, float] = (320.0, 240.0)

    # ── Model paths ──────────────────────────────────────────────────────────
    models_dir: str = "/home/martir/Downloads"
    yolo_hef_path: str = "/home/martir/Downloads/yolo26m.hef"
    depth_hef_path: str = "/home/martir/Downloads/scdepthv3.hef"
    reid_hef_path: str = "/home/martir/Downloads/repvgg_a0_person_reid_512.hef"
    heights_json: str = "src/calibration/object_heights.json"
    intrinsics_json: str = "src/calibration/intrinsics.json"

    # ── Detection ────────────────────────────────────────────────────────────
    det_conf: float = 0.5
    depth_input_height: int = 256
    depth_input_width: int = 320

    # ── ReID (RepVGG-A0) ─────────────────────────────────────────────────────
    reid_input_height: int = 256
    reid_input_width: int = 128
    reid_embedding_dim: int = 512

    # ── ByteTrack ────────────────────────────────────────────────────────────
    track_high_thresh: float = 0.6       # 1st association threshold
    track_low_thresh: float = 0.1        # 2nd association threshold (unconfirmed)
    track_match_thresh: float = 0.8      # IoU matching threshold
    track_buffer: int = 30               # frames to keep lost tracks (1 second @ 30fps)
    track_min_hits: int = 3              # min hits before track is confirmed

    # ── Target Lock (Exemplar Matching) ──────────────────────────────────────
    target_classes: tuple = ("person", "chair")  # auto-lock on first detection of these classes
    golden_template_frames: int = 5      # consecutive stable frames to capture template
    reid_cosine_threshold: float = 0.85  # cosine similarity for re-identification
    reid_search_timeout: int = 90        # frames in SEARCHING before giving up (3s @ 30fps)

    # ── Kalman Depth Fusion ──────────────────────────────────────────────────
    kalman_process_noise: float = 0.1    # Q scaling factor
    kalman_geom_noise_coeff: float = 0.08   # R_geom = (coeff * d)^2
    kalman_depth_noise_coeff: float = 0.06  # R_depth = (coeff * d^1.5)^2
    kalman_scale_ema_alpha: float = 0.05    # EMA for geometric-depth scale alignment
    kalman_gate_chi2: float = 3.84       # chi-squared gate (95% confidence, 1 DOF)

    # ── Arrival Trigger ──────────────────────────────────────────────────────
    arrival_distance_m: float = 0.5      # distance threshold in metres
    arrival_center_tolerance: float = 0.10  # ±10% of frame dimension

    # ── Display ──────────────────────────────────────────────────────────────
    display_width: int = 1280
    display_height: int = 480
    show_depth_map: bool = True
    scalebar_width: int = 28
    default_colormap: int = 0
