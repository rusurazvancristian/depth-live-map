"""Centralised configuration — all constants, paths, tunables. [SHARED]"""

from dataclasses import dataclass


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
    yolo_hef_path: str = "/usr/share/hailo-models/yolov8s_h8.hef"
    depth_hef_path: str = "/usr/share/hailo-models/fast_depth_h8.hef"
    fusion_onnx_path: str = "models/fusion_mlp.onnx"
    fusion_norm_path: str = "models/fusion_norm.pt"
    heights_json: str = "src/calibration/object_heights.json"
    intrinsics_json: str = "src/calibration/intrinsics.json"

    # ── Detection ────────────────────────────────────────────────────────────
    det_conf: float = 0.5
    depth_input_size: int = 224

    # ── Display ──────────────────────────────────────────────────────────────
    display_width: int = 1280
    display_height: int = 480
    show_depth_map: bool = True
    scalebar_width: int = 28
    default_colormap: int = 0
