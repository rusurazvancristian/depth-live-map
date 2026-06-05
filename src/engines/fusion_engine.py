"""Engine 4 — Distance fusion Multi-Layer Perceptron (ONNX) + 2-state EKF on Pi CPU. [TRACK B]"""

import numpy as np
import logging
import os
import json
import math
from typing import Optional, Tuple, Dict, Any

from src.engines.base_engine import BaseEngine
from data_contract import FrameResult

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("onnxruntime not found. FusionEngine will run in EKF fallback mode without MLP.")


class ObjectTracker:
    """2-state Extended Kalman Filter per tracked object: x = [distance, velocity]
    
    Heteroscedastic measurement noise: auto graceful degradation at range.
    """
    dt: float
    x: Optional[np.ndarray]
    P: Optional[np.ndarray]
    scale_factor: Optional[float]
    Q: np.ndarray
    F: np.ndarray
    H: np.ndarray

    def __init__(self, dt: float = 1/30.0) -> None:
        self.dt = dt
        self.x = None     # [d, d_dot]
        self.P = None     # 2×2 covariance
        self.scale_factor = None  # Running calibration scale factor for relative depth
        
        self.Q = np.array([[self.dt**4/4, self.dt**3/2],
                           [self.dt**3/2, self.dt**2  ]]) * 0.1
        self.Q += np.eye(2) * 1e-6

        self.F = np.array([[1, self.dt],
                           [0, 1      ]])
        self.H = np.array([[1, 0]])    # observe distance only

    def init(self, d0: float) -> None:
        self.x = np.array([np.clip(d0, 0.1, 100.0), 0.0])
        self.P = np.diag([1.0, 0.5])

    @staticmethod
    def R_bbox(d: float) -> float:
        d_clipped = max(0.1, d)
        return max(1e-4, (0.08 * d_clipped)**2)

    @staticmethod
    def R_depth(d: float) -> float:
        d_clipped = max(0.1, d)
        return max(1e-4, (0.06 * d_clipped**1.5)**2)

    def predict(self, dt: float) -> None:
        self.dt = dt
        self.F[0, 1] = self.dt
        self.Q = np.array([[self.dt**4/4, self.dt**3/2],
                           [self.dt**3/2, self.dt**2  ]]) * 0.1
        self.Q += np.eye(2) * 1e-6
        
        if self.x is not None:
            self.x = self.F @ self.x
            self.x[0] = np.clip(self.x[0], 0.1, 100.0)
            self.P = self.F @ self.P @ self.F.T + self.Q
            self.P = 0.5 * (self.P + self.P.T)
            self.P[0, 0] = max(1e-6, self.P[0, 0])
            self.P[1, 1] = max(1e-6, self.P[1, 1])

    def update(self, z: float, R: float) -> bool:
        if self.x is None:
            self.init(z)
            return True
            
        R = max(1e-4, R)
        S = (self.H @ self.P @ self.H.T + R).item()
        if S <= 1e-9:
            return False
            
        innovation = (z - self.H @ self.x).item()
        gamma = innovation**2 / S
        if gamma > 3.84:
            return False
        
        K = self.P @ self.H.T / S
        self.x = self.x + K.flatten() * innovation
        self.x[0] = np.clip(self.x[0], 0.1, 100.0)
        
        I_KH = np.eye(2) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + R * (K @ K.T)
        self.P = 0.5 * (self.P + self.P.T)
        self.P[0, 0] = max(1e-6, self.P[0, 0])
        self.P[1, 1] = max(1e-6, self.P[1, 1])
        return True

    @property
    def distance(self) -> float:
        return float(self.x[0]) if self.x is not None else float('nan')


class FusionEngine(BaseEngine):
    """ONNX-based distance fusion MLP + 2-state EKF running on Pi CPU."""
    onnx_path: str
    _session: Optional["ort.InferenceSession"]
    _tracker: ObjectTracker
    _last_timestamp: Optional[float]
    _last_class_id: int
    mean: np.ndarray
    std: np.ndarray

    def __init__(self, onnx_path: str, norm_path: Optional[str] = None) -> None:
        self.onnx_path = onnx_path
        self._session = None
        self._tracker = ObjectTracker()
        self._last_timestamp = None
        self._last_class_id = -1
        
        self.mean = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.std = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        
        if norm_path is None:
            norm_path = os.path.splitext(onnx_path)[0] + "_norm.json"
            
        self._load_norm_params(norm_path)

        if ONNX_AVAILABLE:
            try:
                self._session = ort.InferenceSession(
                    self.onnx_path,
                    providers=["CPUExecutionProvider"]
                )
                logger.info(f"Loaded Fusion MLP ONNX model from {self.onnx_path}")
            except Exception as e:
                logger.error(f"Failed to load ONNX session: {e}")
                self._session = None

    def _load_norm_params(self, path: str) -> None:
        if path.endswith(".pt"):
            try:
                import torch
                if os.path.exists(path):
                    data = torch.load(path, map_location="cpu")
                    if isinstance(data, dict) and "mean" in data and "std" in data:
                        m = data["mean"]
                        self.mean = m.cpu().numpy() if hasattr(m, "cpu") else np.array(m, dtype=np.float32)
                        s = data["std"]
                        self.std = s.cpu().numpy() if hasattr(s, "cpu") else np.array(s, dtype=np.float32)
                        logger.info(f"Loaded normalization parameters from PyTorch file: {path}")
                        return
            except Exception as e:
                logger.warning(f"Failed to load PyTorch norm file {path}: {e}")
            
            json_path = os.path.splitext(path)[0] + ".json"
            if os.path.exists(json_path):
                path = json_path
            else:
                json_path_alt = os.path.splitext(path)[0] + "_norm.json"
                if os.path.exists(json_path_alt):
                    path = json_path_alt

        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                self.mean = np.array(data["mean"], dtype=np.float32)
                self.std = np.array(data["std"], dtype=np.float32)
                logger.info(f"Loaded normalization parameters from JSON file: {path}")
            except Exception as e:
                logger.warning(f"Error reading normalization config {path}, using defaults: {e}")
        else:
            logger.warning(f"Normalization config {path} not found. Using defaults.")

    def process(self, result: FrameResult) -> FrameResult:
        t_now = result.timestamp
        if self._last_timestamp is None:
            dt = 1/30.0
        else:
            dt = max(0.001, t_now - self._last_timestamp)
        self._last_timestamp = t_now

        if dt > 1.5:
            self._tracker.x = None
            self._tracker.P = None
            self._tracker.scale_factor = None

        current_class_id = result.class_id
        if current_class_id != self._last_class_id:
            self._tracker.x = None
            self._tracker.P = None
            self._tracker.scale_factor = None
            self._last_class_id = current_class_id

        d_geom = result.d_geometric_m
        rel_depth = result.rel_depth_score
        
        # Replace NaNs with sentinel values (-1.0) for the MLP features
        feat_geom = -1.0 if math.isnan(d_geom) else float(d_geom)
        feat_depth = -1.0 if math.isnan(rel_depth) else float(rel_depth)
        feat_class = -1.0 if result.class_id < 0 else float(result.class_id)

        if feat_geom == -1.0 and feat_depth == -1.0:
            if self._tracker.x is not None:
                self._tracker.predict(dt)
                self._write_tracker_outputs(result)
            else:
                result.final_distance_m = float("nan")
                result.log_variance = float("nan")
                result.confidence_68 = (0.0, 0.0)
                result.confidence_95 = (0.0, 0.0)
            return result

        try:
            dist_mlp = float("nan")
            log_var_mlp = float("nan")
            
            if ONNX_AVAILABLE and self._session is not None and feat_geom != -1.0 and feat_depth != -1.0:
                try:
                    raw_features = np.array([feat_geom, feat_depth, feat_class], dtype=np.float32)
                    norm_features = (raw_features - self.mean) / (self.std + 1e-8)
                    input_batch = np.expand_dims(norm_features, axis=0)

                    input_name = self._session.get_inputs()[0].name
                    pred = self._session.run(None, {input_name: input_batch})[0]
                    dist_mlp = float(pred[0, 0])
                    log_var_mlp = float(pred[0, 1])
                except Exception as e:
                    logger.warning(f"ONNX MLP execution failed: {e}")

            if self._tracker.x is not None:
                self._tracker.predict(dt)

            if not math.isnan(dist_mlp):
                R = float(np.exp(log_var_mlp))
                R = float(np.clip(R, 1e-4, 25.0))
                self._tracker.update(dist_mlp, R)
            else:
                if feat_geom != -1.0:
                    d_est = feat_geom if self._tracker.x is None else self._tracker.distance
                    R_geom = self._tracker.R_bbox(d_est)
                    self._tracker.update(feat_geom, R_geom)
                    
                if feat_depth != -1.0 and self._tracker.x is not None:
                    if feat_geom != -1.0:
                        safe_geom = max(0.1, feat_geom)
                        safe_depth = max(1e-3, feat_depth)
                        instantaneous_scale = np.clip(safe_geom / safe_depth, 0.1, 10.0)
                        
                        if self._tracker.scale_factor is None:
                            self._tracker.scale_factor = instantaneous_scale
                        else:
                            alpha = 0.05
                            self._tracker.scale_factor = (1.0 - alpha) * self._tracker.scale_factor + alpha * instantaneous_scale
                    
                    if self._tracker.scale_factor is not None:
                        d_depth = feat_depth * self._tracker.scale_factor
                        R_depth = self._tracker.R_depth(self._tracker.distance)
                        self._tracker.update(d_depth, R_depth)

            self._write_tracker_outputs(result)

        except Exception as e:
            logger.error(f"Error in FusionEngine: {e}")
            result.final_distance_m = float("nan")
            result.log_variance = float("nan")
            result.confidence_68 = (0.0, 0.0)
            result.confidence_95 = (0.0, 0.0)

        return result

    def _write_tracker_outputs(self, result: FrameResult) -> None:
        if self._tracker.x is not None:
            dist = float(self._tracker.distance)
            var = float(self._tracker.P[0, 0])
            sigma = float(np.sqrt(max(1e-6, var)))
            dist = float(np.clip(dist, 0.1, 100.0))
            
            result.final_distance_m = dist
            result.log_variance = float(np.log(max(1e-8, var)))
            result.confidence_68 = (float(np.clip(dist - sigma, 0.1, 100.0)), float(np.clip(dist + sigma, 0.1, 100.0)))
            result.confidence_95 = (float(np.clip(dist - 2 * sigma, 0.1, 100.0)), float(np.clip(dist + 2 * sigma, 0.1, 100.0)))
        else:
            result.final_distance_m = float("nan")
            result.log_variance = float("nan")
            result.confidence_68 = (0.0, 0.0)
            result.confidence_95 = (0.0, 0.0)
