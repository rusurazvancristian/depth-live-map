from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    """Immutable configuration. Loaded once at startup."""

    # ── Camera ──
    cam_width: int = 640
    cam_height: int = 480
    cam_fps: int = 30

    # ── Intrinsics (from calibration) ──
    focal_length_px: float = 600.0      # f_y from K matrix — MUST calibrate
    principal_point: tuple[float, float] = (320.0, 240.0)

    # ── Model paths ──
    yolo_hef_path: str = "models/yolov8n.hef"
    depth_hef_path: str = "models/fast_depth_h8.hef"
    fusion_onnx_path: str = "models/fusion_mlp.onnx"
    heights_json: str = "src/calibration/object_heights.json"

    # ── Detection ──
    det_conf: float = 0.5               # Minimum YOLO confidence threshold
    depth_input_size: int = 224          # Depth model input resolution

    # ── Display ──
    display_height: int = 480
    scalebar_width: int = 28
    default_colormap: int = 0           # Index into COLORMAPS list
