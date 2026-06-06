"""Module for orchestrating the Hailo Depth Live pipeline in a background thread."""

import os
import sys
import time
import json
import math
import logging
import threading
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

# Resolve workspace root to ensure absolute paths for resources and module imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import Config, MODEL_REGISTRY
from data_contract import FrameResult, TrackedObject
from src.setup.model_downloader import ensure_models
from src.hailo_inference.hef_loader import HailoMultiplexer, HAILO_AVAILABLE
from src.engines import (
    YOLOEngine,
    GeometryEngine,
    DepthEngine,
    KalmanDepthEngine,
    ReIDEngine,
)
from src.tracking import ByteTracker, TargetLock
from src.utils.visualization import draw_hud, COLORMAPS

logger = logging.getLogger(__name__)

# Constants for BBox smoother matching main.py
_BBOX_ALPHA = 0.35      # EMA smoothing — lower = smoother, higher = more responsive
_COAST_FRAMES = 20      # frames to keep last bbox when target temporarily disappears

# Fallback Camera Mock
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
                    # Keep as BGR format since OpenCV captures in BGR and our pipeline handles it
                    return frame
                else:
                    logger.warning("Webcam opened but failed to read frames. Releasing webcam and falling back to synthetic generator.")
                    self.cap.release()

            # Generate synthetic target frame (simulated human walking + A5 paper sheet)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            t = time.time()
            
            # Draw room grid lines
            for i in range(0, 640, 80):
                cv2.line(frame, (i, 0), (i, 480), (35, 35, 35), 1)
            for j in range(0, 480, 60):
                cv2.line(frame, (0, j), (640, j), (35, 35, 35), 1)
                
            # 1. Simulated person target (red block)
            x = int(320 + 150 * math.sin(t * 0.8))
            y = int(240 + 20 * math.cos(t * 1.6))
            cv2.rectangle(frame, (x - 45, y - 90), (x + 45, y + 90), (0, 0, 180), -1)
            cv2.circle(frame, (x, y - 110), 30, (0, 0, 180), -1)
            
            # 2. Simulated A5 paper sheet (white rectangle) at distance 1.0m
            # Height = 205mm, width = 150mm. f_y = 408px.
            # At 1.0m, rh = 84px, rw = 61px.
            px1, py1 = 320 - 30, 240 - 42
            px2, py2 = 320 + 30, 240 + 42
            cv2.rectangle(frame, (px1, py1), (px2, py2), (240, 240, 240), -1)
            cv2.putText(frame, "MOCK A5 PAPER", (px1 - 15, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            
            # Synthetic helper text
            cv2.putText(
                frame,
                "DEMO MODE: SYNTHETIC TARGET & CALIBRATION",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (0, 255, 255),
                1,
                lineType=cv2.LINE_AA,
            )
            return frame

        def stop(self) -> None:
            if self.cap.isOpened():
                self.cap.release()


class _BBoxSmoother:
    """EMA bbox smoother with coasting for the locked target (matching main.py)."""

    def __init__(self):
        self._bbox_f = None          # float [x1,y1,x2,y2]
        self._coast  = 0             # frames since last live detection
        self._last_dist  = float("nan")
        self._last_class = ""

    def update(self, obj: TrackedObject) -> Tuple[int, int, int, int]:
        """Feed a live TrackedObject; returns smoothed (x1,y1,x2,y2) int tuple."""
        new = np.array(obj.bbox, dtype=float)
        if self._bbox_f is None:
            self._bbox_f = new
        else:
            self._bbox_f = _BBOX_ALPHA * new + (1.0 - _BBOX_ALPHA) * self._bbox_f
        self._coast      = 0
        self._last_dist  = obj.kalman_distance_m
        self._last_class = obj.class_name
        return self._smoothed()

    def coast(self) -> Optional[Tuple[int, int, int, int]]:
        """Called when target_obj is None; returns coasted bbox or None if expired."""
        if self._bbox_f is None:
            return None
        self._coast += 1
        if self._coast > _COAST_FRAMES:
            return None
        return self._smoothed()

    def reset(self) -> None:
        self._bbox_f = None
        self._coast  = 0
        self._last_dist  = float("nan")
        self._last_class = ""

    def _smoothed(self) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = self._bbox_f
        return int(x1), int(y1), int(x2), int(y2)

    @property
    def is_coasting(self) -> bool:
        return self._coast > 0

    @property
    def coasting_frames(self) -> int:
        return self._coast


class MutableConfig:
    """A mutable configuration wrapper initialized from the central read-only Config class."""

    def __init__(self) -> None:
        base = Config()
        # Copy all properties dynamically to allow runtime modifications
        for field_name in base.__dataclass_fields__:
            setattr(self, field_name, getattr(base, field_name))


class LivePipelineManager:
    """Manages the background pipeline thread, live configs, target locking, and calibration."""

    def __init__(self) -> None:
        # Initialize configuration and focal length
        self.config = MutableConfig()
        self._current_focal_length_px = self.load_focal_length()

        # Calibration state machine
        self.calibration_state = "idle"  # "idle" | "frozen" | "verifying"
        self._calibration_rect: Optional[Tuple[int, int, int, int]] = None
        self._frozen_frame: Optional[np.ndarray] = None
        self._frozen_depth: Optional[np.ndarray] = None
        self._freeze_next_frame = False

        # Display / Streaming attributes
        self.colormap_name = "Turbo"
        self._latest_hud_jpeg: Optional[bytes] = None
        self._latest_depth_jpeg: Optional[bytes] = None

        # Target override commands
        self._pending_lock = False
        self._pending_lock_track_id: Optional[int] = None
        self._pending_unlock = False

        # Target status properties
        self._target_id = -1
        self._target_status = "IDLE"
        self._target_distance_m = float("nan")
        self._target_center_offset = (0.0, 0.0)
        self._target_is_arrived = False

        # Multi-threading orchestrator
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Performance metrics
        self.fps = 0.0

    def load_focal_length(self) -> float:
        """Load f_y from intrinsics.json if calibrated, otherwise scale or fallback to config default."""
        try:
            intrinsics_path = os.path.join(ROOT_DIR, self.config.intrinsics_json)
            if os.path.exists(intrinsics_path):
                with open(intrinsics_path) as f:
                    data = json.load(f)
                f_y = float(data["focal_length_px"])
                
                calib_w = data.get("width")
                calib_h = data.get("height")
                if calib_w and calib_h and (calib_w != self.config.cam_width or calib_h != self.config.cam_height):
                    # Scale f_y proportionally with frame height change
                    scale = self.config.cam_height / calib_h
                    f_y_scaled = f_y * scale
                    logger.info(
                        "Scaled loaded f_y from %.1f (calibrated at %dx%d) to %.1f (running at %dx%d)",
                        f_y, calib_w, calib_h, f_y_scaled, self.config.cam_width, self.config.cam_height
                    )
                    return f_y_scaled

                logger.info("Loaded f_y=%.1f from %s", f_y, intrinsics_path)
                return f_y
        except Exception as e:
            logger.warning("Error reading calibration intrinsics (%s) — using default config focal length.", e)
            
        return self.config.focal_length_px

    def start(self) -> None:
        """Start the background pipeline thread."""
        with self._lock:
            if self._running:
                logger.warning("LivePipelineManager is already running.")
                return
            self._running = True
            self._thread = threading.Thread(target=self._pipeline_loop, daemon=True)
            self._thread.start()
            logger.info("LivePipelineManager background thread started.")

    def stop(self) -> None:
        """Stop the background pipeline thread and release resources."""
        thread_to_join = None
        with self._lock:
            if not self._running:
                logger.warning("LivePipelineManager is not running.")
                return
            self._running = False
            thread_to_join = self._thread
            logger.info("Stopping LivePipelineManager background thread...")
            
        if thread_to_join:
            thread_to_join.join(timeout=5.0)
            logger.info("LivePipelineManager background thread stopped.")

    def is_running(self) -> bool:
        """Check if the manager thread is currently running."""
        with self._lock:
            return self._running

    # ── Target Lock API ──────────────────────────────────────────────────────

    def lock_target(self, track_id: Optional[int] = None) -> None:
        """Lock target manually on a track ID or auto-lock on first candidate.
        
        Args:
            track_id: The ByteTrack track ID to lock onto. If None, locks onto
                      the first available object of target class.
        """
        with self._lock:
            self._pending_lock_track_id = track_id
            self._pending_lock = True
            logger.info("LivePipelineManager: Lock target request queued (track_id=%s)", track_id)

    def unlock_target(self) -> None:
        """Reset the target lock, transitioning to searching for new targets."""
        with self._lock:
            self._pending_unlock = True
            logger.info("LivePipelineManager: Unlock target request queued")

    # ── Calibration API ──────────────────────────────────────────────────────

    def start_calibration(self) -> None:
        """Freeze the current frame and prepare for calibration region selection."""
        with self._lock:
            self._freeze_next_frame = True
            self._calibration_rect = None
            self._frozen_frame = None
            self._frozen_depth = None
            self.calibration_state = "frozen"
            logger.info("LivePipelineManager: Calibration initiated, freezing next frame.")

    def set_calibration_rect(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Receive the selection coordinates of the paper sheet on the frozen frame.
        
        Args:
            x1, y1: Top-left coordinate.
            x2, y2: Bottom-right coordinate.
        """
        with self._lock:
            self._calibration_rect = (x1, y1, x2, y2)
            logger.debug("LivePipelineManager: Set calibration rect to (%d, %d, %d, %d)", x1, y1, x2, y2)

    def calculate_and_save_focal_length(self) -> float:
        """Calculate focal length from paper box and save to src/calibration/intrinsics.json.
        
        Calculates using the pinhole model: f = (pixel_size * distance) / real_size.
        """
        with self._lock:
            if self._calibration_rect is None:
                raise ValueError("No calibration rectangle has been selected.")
            
            x1, y1, x2, y2 = self._calibration_rect
            y_diff = abs(y2 - y1)
            x_diff = abs(x2 - x1)
            
            if y_diff == 0 or x_diff == 0:
                raise ValueError("Calibration rectangle dimensions must be non-zero.")
                
            PAPER_H_M = 0.205  # Height of paper: 205mm
            PAPER_W_M = 0.150  # Width of paper: 150mm
            DIST_M = 1.0       # Distance: 1.0m
            
            # Compute focal length in pixels for height and width
            f_y = (y_diff * DIST_M) / PAPER_H_M
            f_x = (x_diff * DIST_M) / PAPER_W_M
            
            # Save configuration
            intrinsics_path = os.path.join(ROOT_DIR, self.config.intrinsics_json)
            os.makedirs(os.path.dirname(intrinsics_path), exist_ok=True)
            
            width = self.config.cam_width
            height = self.config.cam_height
            
            data = {
                "focal_length_px": round(f_y, 1),
                "f_x": round(f_x, 1),
                "f_y": round(f_y, 1),
                "cx": width / 2.0,
                "cy": height / 2.0,
                "width": width,
                "height": height,
                "calibration_method": "a5_paper_1m",
                "notes": "Web UI calibrated via A5 paper at 1.0m"
            }
            
            try:
                with open(intrinsics_path, "w") as f:
                    json.dump(data, f, indent=2)
                logger.info("Saved calibrated focal length f_y=%.1f px to %s", f_y, intrinsics_path)
            except Exception as exc:
                logger.error("Failed to write intrinsics file: %s", exc)
                raise
                
            self._current_focal_length_px = f_y
            self.calibration_state = "idle"
            self._calibration_rect = None
            self._frozen_frame = None
            self._frozen_depth = None
            return f_y

    def verify_calibration(self, enable: bool = True) -> None:
        """Switch verify mode on/off to estimate distance using the new calibration focal length.
        
        Args:
            enable: True to switch to 'verifying' state, False to return to 'idle'.
        """
        with self._lock:
            if enable:
                self.calibration_state = "verifying"
                logger.info("LivePipelineManager: Switched to verifying state.")
            else:
                self.calibration_state = "idle"
                logger.info("LivePipelineManager: Switched to idle state.")

    # ── Config and Stream Getters ────────────────────────────────────────────

    def update_config(self, **kwargs) -> None:
        """Update configuration properties live in a thread-safe way."""
        with self._lock:
            for key, val in kwargs.items():
                if hasattr(self.config, key):
                    if key == "target_classes":
                        if isinstance(val, str):
                            val = (val,)
                        else:
                            val = tuple(val)
                    else:
                        default_val = getattr(Config(), key)
                        if default_val is not None:
                            try:
                                val = type(default_val)(val)
                            except (ValueError, TypeError):
                                pass
                    setattr(self.config, key, val)
                    logger.info("LivePipelineManager: Updated config %s = %s", key, val)
                else:
                    logger.warning("LivePipelineManager: Config has no property %s", key)

    def get_status(self) -> Dict[str, Any]:
        """Return the current pipeline status metrics and configurations."""
        with self._lock:
            config_dict = {
                "det_conf": self.config.det_conf,
                "reid_cosine_threshold": self.config.reid_cosine_threshold,
                "kalman_process_noise": self.config.kalman_process_noise,
                "arrival_distance_m": self.config.arrival_distance_m,
                "target_classes": list(self.config.target_classes),
            }
            return {
                "running": self._running,
                "fps": round(self.fps, 1),
                "calibration_state": self.calibration_state,
                "focal_length_px": round(self._current_focal_length_px, 1),
                "config": config_dict,
                "target_id": self._target_id,
                "target_status": self._target_status,
                "target_distance_m": self._target_distance_m if not math.isnan(self._target_distance_m) else None,
                "target_center_offset": self._target_center_offset,
                "target_is_arrived": self._target_is_arrived,
            }

    def get_latest_hud_jpeg(self) -> Optional[bytes]:
        """Return the latest encoded BGR HUD frame JPEG bytes."""
        with self._lock:
            return self._latest_hud_jpeg

    def get_latest_depth_jpeg(self) -> Optional[bytes]:
        """Return the latest encoded colorized depth map JPEG bytes."""
        with self._lock:
            return self._latest_depth_jpeg

    # ── Pipeline Loop and Helpers ────────────────────────────────────────────

    def _init_camera(self) -> Picamera2:
        """Initialize Camera Module or fallback mock camera."""
        cam = Picamera2()
        if PICAMERA2_AVAILABLE:
            cam_config = cam.create_preview_configuration(
                main={"size": (self.config.cam_width, self.config.cam_height), "format": "BGR888"},
                controls={
                    "FrameRate": self.config.cam_fps,
                    "AfMode": 0,
                    "LensPosition": 0.0,
                },
            )
            cam.configure(cam_config)
        cam.start()
        time.sleep(1.0)  # Camera warm up
        logger.info("Camera online: %dx%d @ %d FPS", self.config.cam_width, self.config.cam_height, self.config.cam_fps)
        return cam

    def _apply_live_config(self, yolo_engine, geometry_engine, kalman_depth_engine, target_lock) -> None:
        """Read and apply configurations to the respective running engines under lock."""
        with self._lock:
            yolo_engine._conf_thr = self.config.det_conf
            target_lock.cosine_thresh = self.config.reid_cosine_threshold
            
            kalman_depth_engine._q_scale = self.config.kalman_process_noise
            for tracker in kalman_depth_engine._trackers.values():
                tracker.q_scale = self.config.kalman_process_noise
                
            target_lock.target_classes = (
                (self.config.target_classes,)
                if isinstance(self.config.target_classes, str)
                else tuple(self.config.target_classes)
            )
            geometry_engine._focal_length_px = self._current_focal_length_px

    def _colorize_depth(self, depth_map: np.ndarray) -> np.ndarray:
        """Generate a colorized depth map using the active colormap."""
        if depth_map is None:
            return np.zeros((self.config.cam_height, self.config.cam_width, 3), dtype=np.uint8)
            
        p2 = np.percentile(depth_map, 2)
        p98 = np.percentile(depth_map, 98)
        depth_clipped = np.clip(depth_map, p2, max(p98, p2 + 1e-6))
        depth_norm = ((depth_clipped - p2) / (p98 - p2) * 255).astype(np.uint8)
        depth_norm = 255 - depth_norm  # Invert: disparity high=close -> close=dark, far=bright
        
        cmap_id = COLORMAPS.get(self.colormap_name, cv2.COLORMAP_TURBO)
        depth_color = cv2.applyColorMap(depth_norm, cmap_id)
        
        # Resize to camera dimensions for consistent output size
        depth_color_resized = cv2.resize(
            depth_color,
            (self.config.cam_width, self.config.cam_height),
            interpolation=cv2.INTER_LINEAR
        )
        return depth_color_resized

    def _render_frozen_hud_and_depth(self) -> None:
        """Render and encode HUD and depth map from the frozen frame."""
        hud_frame = np.zeros((self.config.display_height, self.config.display_width, 3), dtype=np.uint8)
        half_w = self.config.display_width // 2
        
        # Read frozen frame details under lock
        with self._lock:
            frozen_frame = self._frozen_frame.copy() if self._frozen_frame is not None else None
            rect = self._calibration_rect
            frozen_depth = self._frozen_depth
            
        if frozen_frame is None:
            return

        left_panel = cv2.resize(frozen_frame, (half_w, self.config.display_height))
        
        # Draw current calibration rect selection if any
        if rect:
            x1, y1, x2, y2 = rect
            scale_x = half_w / self.config.cam_width
            scale_y = self.config.display_height / self.config.cam_height
            hx1, hy1 = int(round(x1 * scale_x)) , int(round(y1 * scale_y))
            hx2, hy2 = int(round(x2 * scale_x)), int(round(y2 * scale_y))
            cv2.rectangle(left_panel, (hx1, hy1), (hx2, hy2), (0, 255, 0), 2)
            
        # Draw translucent panel status
        panel_h = 95
        panel_w = 260
        status_panel = left_panel[10 : 10 + panel_h, 10 : 10 + panel_w].copy()
        overlay = np.zeros_like(status_panel)
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (20, 20, 20), cv2.FILLED)
        cv2.addWeighted(status_panel, 0.4, overlay, 0.6, 0, dst=left_panel[10 : 10 + panel_h, 10 : 10 + panel_w])
        cv2.rectangle(left_panel, (10, 10), (10 + panel_w, 10 + panel_h), (60, 60, 60), 1, lineType=cv2.LINE_AA)
        
        cv2.putText(left_panel, "CALIBRATION: FROZEN", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(left_panel, "Draw rect around paper", (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        if rect:
            x1, y1, x2, y2 = rect
            rw, rh = abs(x2 - x1), abs(y2 - y1)
            cv2.putText(left_panel, f"ROI: {rw}x{rh} px", (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        else:
            cv2.putText(left_panel, "No selection", (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
            
        hud_frame[:, :half_w] = left_panel
        
        # Colorize and place frozen depth map on right half
        if frozen_depth is not None:
            depth_color = self._colorize_depth(frozen_depth)
            right_panel = cv2.resize(depth_color, (half_w, self.config.display_height))
            hud_frame[:, half_w:] = right_panel
        else:
            hud_frame[:, half_w:] = 0
            
        # Encode
        _, hud_jpg = cv2.imencode('.jpg', hud_frame)
        with self._lock:
            self._latest_hud_jpeg = hud_jpg.tobytes()
            
        if frozen_depth is not None:
            depth_color = self._colorize_depth(frozen_depth)
            _, depth_jpg = cv2.imencode('.jpg', depth_color)
            with self._lock:
                self._latest_depth_jpeg = depth_jpg.tobytes()
        else:
            with self._lock:
                self._latest_depth_jpeg = None

    def _pipeline_loop(self) -> None:
        """Primary background thread loop running the camera capture and inference pipeline."""
        # Download/verify models if Hailo NPU is available
        if HAILO_AVAILABLE:
            logger.info("Ensuring all model HEF files exist in %s...", self.config.models_dir)
            try:
                ensure_models(self.config.models_dir, MODEL_REGISTRY)
            except Exception as e:
                logger.warning("Could not download/verify HEFs: %s", e)

        model_paths = {
            "yolo": self.config.yolo_hef_path,
            "depth": self.config.depth_hef_path,
            "reid": self.config.reid_hef_path,
        }

        logger.info("Initializing HailoMultiplexer...")
        try:
            multiplexer_ctx = HailoMultiplexer(model_paths)
        except Exception as exc:
            logger.error("Failed to initialize HailoMultiplexer: %s", exc)
            with self._lock:
                self._running = False
            return

        with multiplexer_ctx as multiplexer:
            # Instantiate pipeline engines
            yolo_engine = YOLOEngine(
                multiplexer,
                model_name="yolo",
                conf_threshold=self.config.det_conf,
            )
            geometry_engine = GeometryEngine(
                focal_length_px=self._current_focal_length_px,
                heights_path=os.path.join(ROOT_DIR, self.config.heights_json),
            )
            # Retrieve shape from multiplexer to avoid config discrepancies and prevent crashes (e.g. in mock mode)
            depth_shape = multiplexer.get_input_shape("depth")
            depth_h, depth_w = (depth_shape[1], depth_shape[2]) if len(depth_shape) == 4 else (depth_shape[0], depth_shape[1])
            
            reid_shape = multiplexer.get_input_shape("reid")
            reid_h, reid_w = (reid_shape[1], reid_shape[2]) if len(reid_shape) == 4 else (reid_shape[0], reid_shape[1])

            depth_engine = DepthEngine(
                multiplexer,
                model_name="depth",
                input_h=depth_h,
                input_w=depth_w,
            )
            kalman_depth_engine = KalmanDepthEngine(
                q_scale=self.config.kalman_process_noise,
                geom_coeff=self.config.kalman_geom_noise_coeff,
                depth_coeff=self.config.kalman_depth_noise_coeff,
                scale_alpha=self.config.kalman_scale_ema_alpha,
                gate_chi2=self.config.kalman_gate_chi2,
            )
            reid_engine = ReIDEngine(
                multiplexer,
                model_name="reid",
                input_h=reid_h,
                input_w=reid_w,
                embedding_dim=self.config.reid_embedding_dim,
            )

            byte_tracker = ByteTracker(
                high_thresh=self.config.track_high_thresh,
                low_thresh=self.config.track_low_thresh,
                match_thresh=self.config.track_match_thresh,
                buffer=self.config.track_buffer,
                min_hits=self.config.track_min_hits,
            )
            
            target_lock = TargetLock(
                target_class=self.config.target_classes,
                stable_frames=self.config.golden_template_frames,
                cosine_thresh=self.config.reid_cosine_threshold,
                search_timeout=self.config.reid_search_timeout,
            )

            cam = self._init_camera()
            fps = 0.0
            t_prev = time.perf_counter()
            
            reid_vectors: Dict[int, np.ndarray] = {}
            smoother = _BBoxSmoother()

            try:
                while True:
                    # Thread shutdown check
                    with self._lock:
                        if not self._running:
                            break
                        calib_state = self.calibration_state

                    # Handle frozen calibration state
                    if calib_state == "frozen":
                        with self._lock:
                            is_freeze_req = self._freeze_next_frame
                        
                        if is_freeze_req:
                            frame_bgr = cam.capture_array()
                            with self._lock:
                                self._frozen_frame = frame_bgr.copy()
                            
                            temp_result = FrameResult(frame=frame_bgr, timestamp=time.perf_counter())
                            try:
                                temp_result = depth_engine.process(temp_result)
                                with self._lock:
                                    self._frozen_depth = temp_result.depth_map
                            except Exception as exc:
                                logger.error("Failed to run depth engine on frozen frame: %s", exc)
                                with self._lock:
                                    self._frozen_depth = None
                                    
                            with self._lock:
                                self._freeze_next_frame = False

                        # If we have a frozen frame, render HUD and colorized depth map
                        with self._lock:
                            has_frozen = self._frozen_frame is not None
                        if has_frozen:
                            self._render_frozen_hud_and_depth()
                            
                        time.sleep(0.033)
                        continue

                    # Normal streaming / pipeline loop
                    frame_bgr = cam.capture_array()
                    timestamp = time.perf_counter()
                    
                    result = FrameResult(frame=frame_bgr, timestamp=timestamp)

                    # Apply live configuration changes
                    self._apply_live_config(yolo_engine, geometry_engine, kalman_depth_engine, target_lock)

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
                    target_bboxes = [
                        (obj.track_id, obj.bbox)
                        for obj in result.tracked_objects
                        if obj.class_name in self.config.target_classes
                    ]
                    
                    current_embeddings = reid_engine.extract_batch(frame_bgr, target_bboxes)
                    for tid, emb in current_embeddings.items():
                        reid_vectors[tid] = emb

                    # ── STAGE 7: Target Lock State Machine ──
                    with self._lock:
                        if self._pending_unlock:
                            target_lock._handle_lost()
                            smoother.reset()
                            self._pending_unlock = False
                            logger.info("LivePipelineManager: Reset target lock.")

                        if self._pending_lock:
                            track_id = self._pending_lock_track_id
                            if track_id is not None:
                                candidate = next((obj for obj in result.tracked_objects if obj.track_id == track_id), None)
                                if candidate is not None:
                                    candidate_emb = reid_vectors.get(track_id)
                                    if candidate_emb is not None:
                                        target_lock.manual_lock(track_id, candidate_emb)
                                        smoother.reset()
                                        logger.info("LivePipelineManager: Lock track ID %d success", track_id)
                                    else:
                                        logger.warning("LivePipelineManager: Lock failed, no ReID vector for ID %d", track_id)
                                else:
                                    logger.warning("LivePipelineManager: Lock failed, track ID %d not found", track_id)
                            else:
                                lockable_objs = [
                                    obj for obj in result.tracked_objects
                                    if obj.class_name in self.config.target_classes
                                ]
                                if lockable_objs:
                                    candidate = lockable_objs[0]
                                    candidate_emb = reid_vectors.get(candidate.track_id)
                                    if candidate_emb is not None:
                                        target_lock.manual_lock(candidate.track_id, candidate_emb)
                                        smoother.reset()
                                        logger.info("LivePipelineManager: Locked on first visible candidate ID %d", candidate.track_id)
                                    else:
                                        logger.warning("LivePipelineManager: Lock failed, no ReID vector for ID %d", candidate.track_id)
                                else:
                                    logger.warning("LivePipelineManager: Lock failed, no target class objects visible")
                            self._pending_lock = False
                            self._pending_lock_track_id = None

                    target_lock.update(result.tracked_objects, reid_vectors)
                    
                    result.target_id = target_lock.target_id
                    result.target_status = target_lock.status
                    
                    target_obj = next(
                        (obj for obj in result.tracked_objects if obj.track_id == result.target_id),
                        None
                    )

                    if target_obj is not None:
                        target_obj.bbox = smoother.update(target_obj)
                        result.target_distance_m = target_obj.kalman_distance_m
                    elif target_lock.status in ("SEARCHING",):
                        coasted_bbox = smoother.coast()
                        if coasted_bbox is not None:
                            ghost = TrackedObject(
                                track_id=result.target_id,
                                bbox=coasted_bbox,
                                class_name=smoother._last_class,
                                kalman_distance_m=smoother._last_dist,
                            )
                            result.tracked_objects.append(ghost)
                            target_obj = ghost
                            result.target_distance_m = smoother._last_dist
                        else:
                            result.target_distance_m = float("nan")
                    else:
                        smoother.reset()
                        result.target_distance_m = float("nan")

                    if target_obj is not None:
                        x1, y1, x2, y2 = target_obj.bbox
                        frame_w, frame_h = self.config.cam_width, self.config.cam_height
                        dx = abs((x1 + x2) / 2.0 - frame_w / 2.0)
                        dy = abs((y1 + y2) / 2.0 - frame_h / 2.0)
                        is_centered = (dx <= self.config.arrival_center_tolerance * frame_w) and (
                            dy <= self.config.arrival_center_tolerance * frame_h
                        )
                        is_near = (
                            not math.isnan(target_obj.kalman_distance_m)
                            and target_obj.kalman_distance_m <= self.config.arrival_distance_m
                        )
                        result.target_is_arrived = is_centered and is_near
                    else:
                        result.target_is_arrived = False

                    # Clean up stale ReID vectors
                    active_ids = {obj.track_id for obj in result.tracked_objects}
                    for stale_id in list(reid_vectors.keys()):
                        if stale_id not in active_ids:
                            del reid_vectors[stale_id]

                    # Measure FPS
                    t_now = time.perf_counter()
                    fps = 0.9 * fps + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
                    t_prev = t_now
                    with self._lock:
                        self.fps = fps
                        self._target_id = result.target_id
                        self._target_status = result.target_status
                        self._target_distance_m = result.target_distance_m
                        self._target_is_arrived = result.target_is_arrived
                        if target_obj is not None:
                            x1, y1, x2, y2 = target_obj.bbox
                            frame_w, frame_h = self.config.cam_width, self.config.cam_height
                            dx = (x1 + x2) / 2.0 - frame_w / 2.0
                            dy = (y1 + y2) / 2.0 - frame_h / 2.0
                            self._target_center_offset = (round(dx, 1), round(dy, 1))
                        else:
                            self._target_center_offset = (0.0, 0.0)

                    # ── Draw side-by-side HUD display ──
                    hud_frame = draw_hud(
                        result,
                        cmap_name=self.colormap_name,
                        invert_depth=False,
                        display_w=self.config.display_width,
                        display_h=self.config.display_height,
                    )
                    
                    # Overlay FPS
                    cv2.putText(
                        hud_frame,
                        f"FPS: {fps:.1f}",
                        (self.config.display_width - 110, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        lineType=cv2.LINE_AA,
                    )

                    # Overlay Verify Calibration details if verifying
                    if calib_state == "verifying":
                        half_w = self.config.display_width // 2
                        cv2.putText(
                            hud_frame,
                            "CALIBRATION: VERIFY MODE",
                            (half_w - 150, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                            lineType=cv2.LINE_AA,
                        )
                        
                        with self._lock:
                            rect = self._calibration_rect
                        
                        if rect:
                            vx1, vy1, vx2, vy2 = rect
                            scale_x = half_w / self.config.cam_width
                            scale_y = self.config.display_height / self.config.cam_height
                            hx1, hy1 = int(round(vx1 * scale_x)), int(round(vy1 * scale_y))
                            hx2, hy2 = int(round(vx2 * scale_x)), int(round(vy2 * scale_y))
                            cv2.rectangle(hud_frame, (hx1, hy1), (hx2, hy2), (0, 255, 0), 2)
                            
                            vy_diff = abs(vy2 - vy1)
                            if vy_diff > 5:
                                PAPER_H_M = 0.205
                                d_est = (self._current_focal_length_px * PAPER_H_M) / vy_diff
                                label = f"d = {d_est:.3f} m (f_y={self._current_focal_length_px:.1f})"
                                cv2.putText(
                                    hud_frame,
                                    label,
                                    (hx1, max(hy1 - 10, 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 255, 0),
                                    1,
                                    lineType=cv2.LINE_AA,
                                )

                    # Encode JPEG bytes for streaming
                    _, hud_jpg = cv2.imencode('.jpg', hud_frame)
                    with self._lock:
                        self._latest_hud_jpeg = hud_jpg.tobytes()

                    # Colorize and encode depth map
                    if result.depth_map is not None:
                        depth_color = self._colorize_depth(result.depth_map)
                        _, depth_jpg = cv2.imencode('.jpg', depth_color)
                        with self._lock:
                            self._latest_depth_jpeg = depth_jpg.tobytes()
                    else:
                        with self._lock:
                            self._latest_depth_jpeg = None

            finally:
                cam.stop()
                logger.info("Camera stopped. LivePipelineManager background loop exiting.")
