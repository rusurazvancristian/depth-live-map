"""Main orchestrator — camera loop + 4-engine pipeline + display. [TRACK A]"""

import json
import logging
import math
import time

import cv2
import numpy as np
from picamera2 import Picamera2

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

    engines: list[BaseEngine] = [
        YOLOEngine(hef_path=config.yolo_hef_path, conf_threshold=config.det_conf),
        GeometryEngine(focal_length_px=focal_length_px, heights_path=config.heights_json),
    ]

    if _DEPTH_AVAILABLE:
        engines.append(DepthEngine(hef_path=config.depth_hef_path,
                                   model_input_size=config.depth_input_size))
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


def init_camera(config: Config) -> Picamera2:
    """Initialise and start the Camera Module 3.

    Args:
        config: Config instance with cam_width, cam_height, cam_fps.

    Returns:
        Running Picamera2 instance.
    """
    cam = Picamera2()
    cam_config = cam.create_preview_configuration(
        main={"size": (config.cam_width, config.cam_height), "format": "RGB888"},
        controls={"FrameRate": config.cam_fps},
    )
    cam.configure(cam_config)
    cam.start()
    time.sleep(1.0)  # warm up
    logger.info("Camera started: %dx%d @ %d FPS", config.cam_width, config.cam_height, config.cam_fps)
    return cam


def main() -> None:
    config = Config()
    cam = init_camera(config)
    engines = build_pipeline(config)

    # Start YOLO session (holds the Hailo pipeline open across frames)
    yolo: YOLOEngine = engines[0]
    yolo.start()

    cv2.namedWindow("How Far?", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("How Far?", config.display_width, config.display_height)

    fps = 0.0
    t_prev = time.perf_counter()

    logger.info("Pipeline running. Press Q to quit.")

    try:
        while True:
            frame = cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2BGR)
            result = FrameResult(frame=frame, timestamp=time.perf_counter())
            result = run_pipeline(engines, result)

            t_now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
            t_prev = t_now

            vis = draw_overlay(frame, result, fps)
            cv2.imshow("How Far?", vis)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        yolo.stop()
        cam.stop()
        cv2.destroyAllWindows()
        logger.info("Pipeline stopped.")


if __name__ == "__main__":
    main()
