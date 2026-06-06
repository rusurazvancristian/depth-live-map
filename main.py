"""Main orchestrator — camera loop + 4-engine pipeline + display. [TRACK A]"""

import json
import logging
import math
import time

import cv2
import numpy as np
from picamera2 import Picamera2
from hailo_platform import VDevice, HailoSchedulingAlgorithm

from config import Config
from data_contract import FrameResult
from src.engines.base_engine import BaseEngine
from src.engines.yolo_engine import YOLOEngine
from src.engines.geometry_engine import GeometryEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Lazy imports for Track B engines (not available until Track B implements them) ──
try:
    from src.engines.depth_engine import DepthEngine
    _DEPTH_AVAILABLE = True
except ImportError:
    _DEPTH_AVAILABLE = False
    logger.warning("DepthEngine not available — running without depth cue.")

try:
    from src.engines.fusion_engine import FusionEngine
    _FUSION_AVAILABLE = True
except ImportError:
    _FUSION_AVAILABLE = False
    logger.warning("FusionEngine not available — using GeometryEngine output as final distance.")


def load_focal_length(config: Config) -> float:
    """Load f_y from intrinsics.json if available, else use config default."""
    try:
        with open(config.intrinsics_json) as f:
            data = json.load(f)
        f_y = float(data["focal_length_px"])
        logger.info("Loaded f_y=%.1f from %s", f_y, config.intrinsics_json)
        return f_y
    except (FileNotFoundError, KeyError):
        logger.warning(
            "intrinsics.json not found — using default f_y=%.1f. Run calibrate_camera.py!",
            config.focal_length_px,
        )
        return config.focal_length_px


def build_pipeline(config: Config) -> list[BaseEngine]:
    """Instantiate and return all available engines in order.

    Engines 3 and 4 are conditionally loaded — if Track B hasn't implemented
    them yet, the pipeline degrades gracefully to geometry-only distance.
    """
    focal_length_px = load_focal_length(config)

    # Shared VDevice with ROUND_ROBIN scheduler — allows multiple HEF models on one chip
    vdevice_params = VDevice.create_params()
    vdevice_params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    shared_device = VDevice(vdevice_params)

    engines: list[BaseEngine] = [
        YOLOEngine(hef_path=config.yolo_hef_path, conf_threshold=config.det_conf,
                   device=shared_device, use_scheduler=True),
        GeometryEngine(focal_length_px=focal_length_px, heights_path=config.heights_json),
    ]

    if _DEPTH_AVAILABLE:
        engines.append(DepthEngine(hef_path=config.depth_hef_path,
                                   device=shared_device, use_scheduler=True))
    if _FUSION_AVAILABLE:
        engines.append(FusionEngine(onnx_path=config.fusion_onnx_path,
                                    norm_path=config.fusion_norm_path))

    logger.info("Pipeline: %s", " -> ".join(e.name for e in engines))
    return engines


def run_pipeline(engines: list[BaseEngine], result: FrameResult) -> FrameResult:
    """Chain engines sequentially. Pure function over the engine list.

    Args:
        engines: Ordered list of BaseEngine instances.
        result: Initial FrameResult with frame and timestamp populated.

    Returns:
        FrameResult after all engines have processed it.
    """
    for engine in engines:
        result = engine.process(result)
    return result


def draw_overlay(frame: np.ndarray, result: FrameResult, fps: float) -> np.ndarray:
    """Draw bounding box, distance label, and FPS on the frame.

    Args:
        frame: BGR image to draw on (copied internally).
        result: Processed FrameResult.
        fps: Current frames per second.

    Returns:
        BGR image with overlay drawn.
    """
    vis = frame.copy()
    h, w = vis.shape[:2]

    # FPS
    cv2.putText(vis, f"{fps:.1f} FPS", (w - 100, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

    if result.class_id < 0:
        cv2.putText(vis, "No detection", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 220), 2, cv2.LINE_AA)
        return vis

    x1, y1, x2, y2 = result.bbox
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Distance — prefer Fusion, fall back to Geometry
    if not math.isnan(result.final_distance_m):
        dist_m = result.final_distance_m
        lo, hi = result.confidence_95
        dist_text = f"{dist_m:.2f}m  [{lo:.1f}-{hi:.1f}]"
    elif not math.isnan(result.d_geometric_m):
        dist_text = f"{result.d_geometric_m:.2f}m (geo)"
    else:
        dist_text = "?.??m"

    label = f"{result.class_name} {result.det_confidence:.2f}  {dist_text}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
    cv2.putText(vis, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    return vis


# Slider range: 0-100 maps to LensPosition 0.0-10.0 dioptres
# 0   = infinity (~inf m)
# 10  = ~1.0 m
# 20  = ~0.5 m
# 100 = ~0.1 m (macro)
_LENS_SCALE = 10.0  # slider_val / _LENS_SCALE = dioptres

def _slider_to_dioptres(val: int) -> float:
    return val / _LENS_SCALE

def _dioptres_to_distance(d: float) -> str:
    if d < 0.01:
        return "inf"
    return f"{1.0 / d:.2f}m"


def init_camera(config: Config) -> Picamera2:
    """Initialise and start the Camera Module 3 in manual focus mode.

    Args:
        config: Config instance with cam_width, cam_height, cam_fps.

    Returns:
        Running Picamera2 instance set to manual focus.
    """
    cam = Picamera2()
    cam_config = cam.create_preview_configuration(
        main={"size": (config.cam_width, config.cam_height), "format": "BGR888"},
        controls={
            "FrameRate": config.cam_fps,
            "AfMode": 0,          # 0=manual, 1=auto, 2=continuous
            "LensPosition": 0.0,  # start at infinity
        },
    )
    cam.configure(cam_config)
    cam.start()
    time.sleep(1.0)
    logger.info("Camera started: %dx%d @ %d FPS | focus=manual", config.cam_width, config.cam_height, config.cam_fps)
    return cam


def _render_depth_panel(depth_map: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Convert float log-disparity map to a coloured BGR panel for display."""
    p2, p98 = np.percentile(depth_map, (2, 98))
    norm = np.clip((depth_map - p2) / max(p98 - p2, 1e-6), 0.0, 1.0)
    grey = (norm * 255).astype(np.uint8)
    coloured = cv2.applyColorMap(grey, cv2.COLORMAP_INFERNO)
    return cv2.resize(coloured, (target_w, target_h))


def main() -> None:
    config = Config()
    cam = init_camera(config)
    engines = build_pipeline(config)

    yolo: YOLOEngine = engines[0]
    yolo.start()

    depth_engine = next((e for e in engines if e.name == "DepthEngine"), None)
    if depth_engine is not None:
        depth_engine.start()

    has_depth = depth_engine is not None and config.show_depth_map
    win_w = config.display_width if has_depth else config.cam_width
    win_h = config.display_height

    WINDOW = "How Far?"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, win_w, win_h)

    # Focus slider: 0 (infinity) .. 100 (macro ~0.1m)
    cv2.createTrackbar("Focus (0=inf)", WINDOW, 0, 100, lambda _: None)

    fps = 0.0
    t_prev = time.perf_counter()
    last_lens_val = -1

    logger.info("Pipeline running. R=reset tracker. Q=quit.")

    try:
        while True:
            slider_val = cv2.getTrackbarPos("Focus (0=inf)", WINDOW)
            if slider_val != last_lens_val:
                dioptres = _slider_to_dioptres(slider_val)
                cam.set_controls({"LensPosition": dioptres})
                last_lens_val = slider_val
                logger.debug("LensPosition=%.2f  (~%s)", dioptres, _dioptres_to_distance(dioptres))

            frame = cam.capture_array()
            result = FrameResult(frame=frame, timestamp=time.perf_counter())
            result = run_pipeline(engines, result)

            t_now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
            t_prev = t_now

            vis = draw_overlay(frame, result, fps)

            dioptres = _slider_to_dioptres(slider_val)
            focus_text = f"Focus: {_dioptres_to_distance(dioptres)}"
            h, w = vis.shape[:2]
            cv2.putText(vis, focus_text, (w - 160, h - 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)

            if has_depth and depth_engine.depth_map is not None:
                depth_panel = _render_depth_panel(depth_engine.depth_map, win_h, config.cam_width)
                # Highlight bbox region on depth panel
                if result.class_id >= 0:
                    x1, y1, x2, y2 = result.bbox
                    dw_scale = config.cam_width / frame.shape[1]
                    dh_scale = win_h / frame.shape[0]
                    cv2.rectangle(depth_panel,
                                  (int(x1 * dw_scale), int(y1 * dh_scale)),
                                  (int(x2 * dw_scale), int(y2 * dh_scale)),
                                  (0, 255, 0), 2)
                    if not math.isnan(result.rel_depth_score):
                        cv2.putText(depth_panel, f"d={result.rel_depth_score:.2f}",
                                    (int(x1 * dw_scale) + 4, int(y1 * dh_scale) + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
                # Stack camera + depth side by side
                vis_resized = cv2.resize(vis, (config.cam_width, win_h))
                display = np.hstack([vis_resized, depth_panel])
            else:
                display = cv2.resize(vis, (win_w, win_h))

            cv2.imshow(WINDOW, cv2.cvtColor(display, cv2.COLOR_BGR2RGB))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                yolo.reset_tracker()

    finally:
        yolo.stop()
        if depth_engine is not None:
            depth_engine.stop()
        cam.stop()
        cv2.destroyAllWindows()
        logger.info("Pipeline stopped.")


if __name__ == "__main__":
    main()
