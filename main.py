"""Main orchestrator — model downloads, multiplexer session, camera loop, tracking pipeline, and live HUD display."""

import json
import logging
import math
import time
import os
import cv2
import numpy as np

# Config and data contract
from config import Config, MODEL_REGISTRY
from data_contract import FrameResult, TrackedObject
from src.setup.model_downloader import ensure_models
from src.hailo_inference.hef_loader import HailoMultiplexer
from src.engines import (
    YOLOEngine,
    GeometryEngine,
    DepthEngine,
    KalmanDepthEngine,
    ReIDEngine,
)
from src.tracking import ByteTracker, TargetLock
from src.utils.visualization import draw_hud

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    logger.warning("picamera2 not found. Initialising mock Picamera2 using system webcam / synthetic generator.")

    class Picamera2:
        """Mock Picamera2 implementation using OpenCV webcam or synthetic animated frame generator."""

        def __init__(self) -> None:
            self.cap = cv2.VideoCapture(0)
            self.dummy_frame = None

        def configure(self, cam_config) -> None:
            pass

        def start(self) -> None:
            if not self.cap.isOpened():
                logger.warning("Could not open system webcam. Generating synthetic test frames.")
                self.dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        def capture_array(self) -> np.ndarray:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    # Picamera2 outputs RGB, OpenCV reads BGR
                    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Generate synthetic target frame (simulated human walking)
            frame = self.dummy_frame.copy()
            t = time.time()
            # Walk left/right
            x = int(320 + 150 * math.sin(t * 0.8))
            y = int(240 + 20 * math.cos(t * 1.6))
            
            # Draw room grid lines
            for i in range(0, 640, 80):
                cv2.line(frame, (i, 0), (i, 480), (35, 35, 35), 1)
            for j in range(0, 480, 60):
                cv2.line(frame, (0, j), (640, j), (35, 35, 35), 1)
                
            # Simulated person target (red block)
            cv2.rectangle(frame, (x - 45, y - 90), (x + 45, y + 90), (0, 0, 180), -1)
            cv2.circle(frame, (x, y - 110), 30, (0, 0, 180), -1)
            
            # Synthetic helper text
            cv2.putText(
                frame,
                "DEMO MODE: SYNTHETIC TARGET GENERATOR",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                lineType=cv2.LINE_AA,
            )
            return frame

        def stop(self) -> None:
            if self.cap.isOpened():
                self.cap.release()


def load_focal_length(config: Config) -> float:
    """Load f_y from intrinsics.json if calibrated, otherwise scale or fallback to config default."""
    try:
        with open(config.intrinsics_json) as f:
            data = json.load(f)
        f_y = float(data["focal_length_px"])
        
        calib_w = data.get("width")
        calib_h = data.get("height")
        if calib_w and calib_h and (calib_w != config.cam_width or calib_h != config.cam_height):
            # Scale f_y proportionally with frame height change
            scale = config.cam_height / calib_h
            f_y_scaled = f_y * scale
            logger.info(
                "Scaled loaded f_y from %.1f (calibrated at %dx%d) to %.1f (running at %dx%d)",
                f_y, calib_w, calib_h, f_y_scaled, config.cam_width, config.cam_height
            )
            return f_y_scaled

        logger.info("Loaded f_y=%.1f from %s", f_y, config.intrinsics_json)
        return f_y
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        logger.warning(
            "Could not load calibration intrinsics (%s) — using default config focal length %.1f.",
            e,
            config.focal_length_px,
        )
        return config.focal_length_px


def init_camera(config: Config) -> Picamera2:
    """Initialize Raspberry Pi 5 Camera Module 3 or fallback mock camera."""
    cam = Picamera2()
    if PICAMERA2_AVAILABLE:
        cam_config = cam.create_preview_configuration(
            main={"size": (config.cam_width, config.cam_height), "format": "BGR888"},
            controls={
                "FrameRate": config.cam_fps,
                "AfMode": 0,         # manual focus
                "LensPosition": 0.0, # start at infinity
            },
        )
        cam.configure(cam_config)
    cam.start()
    time.sleep(1.0)
    logger.info("Camera online: %dx%d @ %d FPS | focus=manual", config.cam_width, config.cam_height, config.cam_fps)
    return cam


def main() -> None:
    config = Config()

    # 1. Verify / download model HEF files
    logger.info("Ensuring all model files exist in %s...", config.models_dir)
    try:
        ensure_models(config.models_dir, MODEL_REGISTRY)
    except Exception as e:
        logger.critical("Failed to verify/download HEFs: %s", e)
        return

    # Determine focal length
    focal_length_px = load_focal_length(config)

    # Dictionary mapping model names to HEF files for multiplexer
    model_paths = {
        "yolo": config.yolo_hef_path,
        "depth": config.depth_hef_path,
        "reid": config.reid_hef_path,
    }

    # 2. Setup NPU Multiplexer Context
    logger.info("Initializing HailoMultiplexer with models: %s", list(model_paths.keys()))
    with HailoMultiplexer(model_paths) as multiplexer:
        
        # 3. Instantiate pipeline engines
        yolo_engine = YOLOEngine(
            multiplexer,
            model_name="yolo",
            conf_threshold=config.det_conf,
        )
        geometry_engine = GeometryEngine(
            focal_length_px=focal_length_px,
            heights_path=config.heights_json,
        )
        depth_engine = DepthEngine(
            multiplexer,
            model_name="depth",
            input_h=config.depth_input_height,
            input_w=config.depth_input_width,
        )
        kalman_depth_engine = KalmanDepthEngine(
            q_scale=config.kalman_process_noise,
            geom_coeff=config.kalman_geom_noise_coeff,
            depth_coeff=config.kalman_depth_noise_coeff,
            scale_alpha=config.kalman_scale_ema_alpha,
            gate_chi2=config.kalman_gate_chi2,
        )
        reid_engine = ReIDEngine(
            multiplexer,
            model_name="reid",
            input_h=config.reid_input_height,
            input_w=config.reid_input_width,
            embedding_dim=config.reid_embedding_dim,
        )

        # 4. Instantiate ByteTracker and TargetLock state machine
        byte_tracker = ByteTracker(
            high_thresh=config.track_high_thresh,
            low_thresh=config.track_low_thresh,
            match_thresh=config.track_match_thresh,
            buffer=config.track_buffer,
            min_hits=config.track_min_hits,
        )
        
        target_lock = TargetLock(
            target_class=config.target_class_name,
            stable_frames=config.golden_template_frames,
            cosine_thresh=config.reid_cosine_threshold,
            search_timeout=config.reid_search_timeout,
        )

        # Start Camera
        cam = init_camera(config)

        # Open Window
        window_name = "Depth Live Tracker HUD — Hailo-8"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, config.display_width, config.display_height)

        fps = 0.0
        t_prev = time.perf_counter()
        
        logger.info("HUD Pipeline running. Press [T] to manually lock on target, [Q] to quit.")

        reid_vectors: Dict[int, np.ndarray] = {}

        try:
            while True:
                # Capture frame (BGR888 — no conversion needed)
                frame_bgr = cam.capture_array()
                
                timestamp = time.perf_counter()
                
                # Create result container
                result = FrameResult(frame=frame_bgr, timestamp=timestamp)

                # ── STAGE 1: YOLO Object Detection ──
                result = yolo_engine.process(result)

                # ── STAGE 2: ByteTrack Multi-Object Association ──
                if result.detections:
                    dets = np.array(
                        [list(d.bbox) + [d.confidence] for d in result.detections],
                        dtype=np.float32
                    )
                    cids = np.array([d.class_id for d in result.detections], dtype=np.int32)
                else:
                    dets = np.empty((0, 5), dtype=np.float32)
                    cids = np.empty((0,), dtype=np.int32)
                
                result.tracked_objects = byte_tracker.update(dets, cids)

                # ── STAGE 3: Geometry-based Distance Calculation ──
                result = geometry_engine.process(result)

                # ── STAGE 4: SCDepthV3 Depth Estimation ──
                result = depth_engine.process(result)

                # ── STAGE 5: Kalman Depth Fusion ──
                result = kalman_depth_engine.process(result)

                # ── STAGE 6: ReID Feature Extraction for target lock matching ──
                # Gather bboxes for tracked objects of target class
                target_bboxes = [
                    (obj.track_id, obj.bbox)
                    for obj in result.tracked_objects
                    if obj.class_name == config.target_class_name
                ]
                
                # Extract embeddings in batch
                current_embeddings = reid_engine.extract_batch(frame_bgr, target_bboxes)
                for tid, emb in current_embeddings.items():
                    reid_vectors[tid] = emb

                # ── STAGE 7: Target Lock State Machine ──
                target_lock.update(result.tracked_objects, reid_vectors)
                
                # Populate target details back to FrameResult
                result.target_id = target_lock.target_id
                result.target_status = target_lock.status
                
                # Identify matched target object and verify arrival trigger
                target_obj = next(
                    (obj for obj in result.tracked_objects if obj.track_id == result.target_id),
                    None
                )
                
                if target_obj is not None:
                    result.target_distance_m = target_obj.kalman_distance_m
                    
                    # Verify arrival conditions: target distance <= 0.5m and bbox center centered
                    x1, y1, x2, y2 = target_obj.bbox
                    tx_center = (x1 + x2) / 2.0
                    ty_center = (y1 + y2) / 2.0
                    
                    # Centered definition: within tolerance from center
                    frame_w, frame_h = config.cam_width, config.cam_height
                    dx = abs(tx_center - frame_w / 2.0)
                    dy = abs(ty_center - frame_h / 2.0)
                    
                    is_centered = (dx <= config.arrival_center_tolerance * frame_w) and (
                        dy <= config.arrival_center_tolerance * frame_h
                    )
                    
                    is_near = (
                        not math.isnan(target_obj.kalman_distance_m)
                        and target_obj.kalman_distance_m <= config.arrival_distance_m
                    )
                    
                    result.target_is_arrived = is_centered and is_near
                else:
                    result.target_distance_m = float("nan")
                    result.target_is_arrived = False

                # Garbage collect stale ReID vectors of tracks no longer present
                active_ids = {obj.track_id for obj in result.tracked_objects}
                for stale_id in list(reid_vectors.keys()):
                    if stale_id not in active_ids:
                        del reid_vectors[stale_id]

                # Calculate FPS
                t_now = time.perf_counter()
                fps = 0.9 * fps + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
                t_prev = t_now

                # 8. Draw side-by-side HUD display
                hud_frame = draw_hud(
                    result,
                    cmap_name="Turbo",
                    invert_depth=False,
                    display_w=config.display_width,
                    display_h=config.display_height,
                )
                
                # Overlay current FPS on the HUD
                cv2.putText(
                    hud_frame,
                    f"FPS: {fps:.1f}",
                    (config.display_width - 110, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    lineType=cv2.LINE_AA,
                )

                cv2.imshow(window_name, hud_frame)

                # 9. Key press actions
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("User requested exit.")
                    break
                elif key == ord("t"):
                    # Manual lock override on first target class detection
                    lockable_objs = [
                        obj for obj in result.tracked_objects
                        if obj.class_name == config.target_class_name
                    ]
                    if lockable_objs:
                        candidate = lockable_objs[0]
                        candidate_id = candidate.track_id
                        candidate_emb = reid_vectors.get(candidate_id)
                        if candidate_emb is not None:
                            target_lock.manual_lock(candidate_id, candidate_emb)
                            logger.info("Manual override triggered lock on target ID %d", candidate_id)
                        else:
                            logger.warning("Manual override failed: ReID vector unavailable for ID %d", candidate_id)
                    else:
                        logger.warning("Manual override failed: no target class object in frame")

        finally:
            cam.stop()
            cv2.destroyAllWindows()
            logger.info("System clean exit.")


if __name__ == "__main__":
    main()
