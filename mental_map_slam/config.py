from dataclasses import dataclass


@dataclass(frozen=True)
class CameraConfig:
    fx: float = 554.3
    fy: float = 554.3
    cx: float = 960.0
    cy: float = 540.0
    width: int = 1920
    height: int = 1080
    fps: float = 15.0
    cam_height_m: float = 0.40   # camera height above ground (Unitree Go2)


@dataclass(frozen=True)
class SLAMConfig:
    # DepthAnything V2 metric INDOOR model (Hypersim) — scene is indoor.
    # The Outdoor (vKITTI) model overestimates indoor depth ~8x; verified via the
    # refrigerator anchor (1.6 m known height). Switch to Large for more accuracy.
    depth_model_id: str = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
    depth_fallback_id: str = "depth-anything/Depth-Anything-V2-Small-hf"
    # Refrigerator-anchored metric correction (see depth_engine). Indoor model
    # reads ~1.8x too far; 0.55 brings the fridge to its true 1.6 m height.
    depth_metric_scale: float = 0.55

    # Feature tracking
    max_features: int = 3000
    min_pnp_inliers: int = 15
    pnp_reproj_error: float = 6.0
    max_translation_m: float = 2.0   # reject pose if jump > this per step

    # Depth thresholds
    depth_min_m: float = 0.3
    depth_max_m: float = 15.0

    # BEV map: 5cm/pixel → 2000px covers 100m × 100m
    bev_res_m: float = 0.05
    bev_size_px: int = 2000
    depth_subsample: int = 8     # stride when projecting depth to BEV

    # Processing: keep every Nth frame (15 FPS → stride 3 = 5 effective FPS)
    frame_stride: int = 3


CAMERA = CameraConfig()
SLAM = SLAMConfig()
