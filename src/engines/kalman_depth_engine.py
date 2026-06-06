"""Engine — Kalman-filtered metric depth estimation."""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from data_contract import FrameResult, TrackedObject
from src.engines.base_engine import BaseEngine

logger = logging.getLogger(__name__)


class _DepthTracker:
    """Kalman filter tracking metric depth for a single track ID."""

    def __init__(
        self,
        q_scale: float = 0.1,
        geom_coeff: float = 0.08,
        depth_coeff: float = 0.06,
        scale_alpha: float = 0.05,
        gate_chi2: float = 3.84,
    ) -> None:
        self.q_scale = q_scale
        self.geom_coeff = geom_coeff
        self.depth_coeff = depth_coeff
        self.scale_alpha = scale_alpha
        self.gate_chi2 = gate_chi2

        # State x = [distance, velocity]^T
        self.x = np.zeros((2, 1), dtype=np.float32)
        # Covariance matrix P
        self.P = np.eye(2, dtype=np.float32) * 10.0
        
        # Scale factor for Depth Anything V2: d = scale / disparity
        # At ~1m, disparity ≈ 2.0 → scale ≈ 2.0 (auto-calibrated from geometry)
        self.scale_factor: float = 2.0
        
        self.initialized = False
        self.absence_counter = 0

        # State transition F
        self.F = np.eye(2, dtype=np.float32)
        # Observation matrix H
        self.H = np.array([[1.0, 0.0]], dtype=np.float32)

    def predict(self, dt: float) -> None:
        """Perform Kalman predict step."""
        if not self.initialized:
            return

        # Update F matrix with dt
        self.F[0, 1] = dt

        # Process noise Q
        Q = np.array([
            [dt**4 / 4.0, dt**3 / 2.0],
            [dt**3 / 2.0, dt**2]
        ], dtype=np.float32) * self.q_scale + np.eye(2, dtype=np.float32) * 1e-6

        # Propagate state and covariance
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + Q

    def update_geometric(self, z: float, rel_depth: Optional[float] = None) -> bool:
        """Update state using geometric distance measurement.

        Also updates the scale_factor via EMA if rel_depth is available.
        """
        if np.isnan(z) or z <= 0.1:
            return False

        # Update scale: disparity model → d = scale / disparity
        # so scale = d_geo * disparity_at_bbox
        if rel_depth is not None and not np.isnan(rel_depth) and rel_depth > 1e-4:
            s_inst = z * rel_depth
            if not self.initialized:
                self.scale_factor = s_inst
            else:
                self.scale_factor = (
                    (1.0 - self.scale_alpha) * self.scale_factor
                    + self.scale_alpha * s_inst
                )

        if not self.initialized:
            self.x[0, 0] = z
            self.x[1, 0] = 0.0
            r_init = (self.geom_coeff * z) ** 2
            self.P = np.array([[r_init, 0.0], [0.0, 1.0]], dtype=np.float32)
            self.initialized = True
            return True

        # Measurement noise R
        R = (self.geom_coeff * self.x[0, 0]) ** 2

        # Innovation
        y = z - self.x[0, 0]
        S = self.P[0, 0] + R

        # Chi-squared gating
        if (y**2) / S > self.gate_chi2:
            logger.debug("Geometric measurement gated out (innovation=%.3f, S=%.3f)", y, S)
            return False

        # Kalman Gain
        K = self.P[:, 0:1] / S

        # State update
        self.x = self.x + K * y

        # Joseph form covariance update for numerical stability: P = (I-KH)P(I-KH)' + KRK'
        I_KH = np.eye(2, dtype=np.float32) - np.dot(K, self.H)
        self.P = np.dot(np.dot(I_KH, self.P), I_KH.T) + np.dot(K, K.T) * R

        # Clip distance to [0.1, 100.0]
        self.x[0, 0] = np.clip(self.x[0, 0], 0.1, 100.0)
        return True

    def update_depth(self, rel_depth: float) -> bool:
        """Update state using disparity measurement (scaled to metric).

        Depth Anything V2: disparity model, d = scale / disparity.
        """
        if np.isnan(rel_depth) or rel_depth <= 0.0:
            return False

        # Convert disparity to metric distance: d = scale / disparity
        z = self.scale_factor / rel_depth

        if not self.initialized:
            self.x[0, 0] = z
            self.x[1, 0] = 0.0
            r_init = (self.depth_coeff * (z**1.5)) ** 2
            self.P = np.array([[r_init, 0.0], [0.0, 1.0]], dtype=np.float32)
            self.initialized = True
            return True

        # Measurement noise R
        d_est = self.x[0, 0]
        R = (self.depth_coeff * (d_est**1.5)) ** 2

        # Innovation
        y = z - self.x[0, 0]
        S = self.P[0, 0] + R

        # Chi-squared gating
        if (y**2) / S > self.gate_chi2:
            logger.debug("Depth measurement gated out (innovation=%.3f, S=%.3f)", y, S)
            return False

        # Kalman Gain
        K = self.P[:, 0:1] / S

        # State update
        self.x = self.x + K * y

        # Joseph form covariance update
        I_KH = np.eye(2, dtype=np.float32) - np.dot(K, self.H)
        self.P = np.dot(np.dot(I_KH, self.P), I_KH.T) + np.dot(K, K.T) * R

        # Clip distance
        self.x[0, 0] = np.clip(self.x[0, 0], 0.1, 100.0)
        return True

    @property
    def distance(self) -> float:
        return float(self.x[0, 0]) if self.initialized else float("nan")

    @property
    def variance(self) -> float:
        return float(self.P[0, 0]) if self.initialized else float("nan")


class KalmanDepthEngine(BaseEngine):
    """Fuses geometric distance and monocular relative depth using a Kalman filter."""

    def __init__(
        self,
        q_scale: float = 0.1,
        geom_coeff: float = 0.08,
        depth_coeff: float = 0.06,
        scale_alpha: float = 0.05,
        gate_chi2: float = 3.84,
    ) -> None:
        """Initialize KalmanDepthEngine.

        Parameters match standard fusion hyperparameters.
        """
        self._q_scale = q_scale
        self._geom_coeff = geom_coeff
        self._depth_coeff = depth_coeff
        self._scale_alpha = scale_alpha
        self._gate_chi2 = gate_chi2

        # Dict mapping track ID -> _DepthTracker
        self._trackers: Dict[int, _DepthTracker] = {}
        # Keep track of timestamps to compute dt
        self._last_timestamp: Optional[float] = None

        logger.info(
            "KalmanDepthEngine ready | q_scale=%.2f | geom_coeff=%.2f | depth_coeff=%.2f",
            q_scale, geom_coeff, depth_coeff
        )

    def process(self, result: FrameResult) -> FrameResult:
        """Fuse geometric and relative depth estimates for tracked objects.

        Args:
            result: FrameResult containing frame, timestamp, tracked_objects, and depth_map.

        Returns:
            FrameResult with kalman_distance_m, kalman_variance, and rel_depth_score updated.
        """
        timestamp = result.timestamp
        depth_map = result.depth_map

        # Calculate dt (fallback to 1/30s if first frame or non-monotonic)
        dt = 1.0 / 30.0
        if self._last_timestamp is not None and timestamp > self._last_timestamp:
            dt = timestamp - self._last_timestamp
        self._last_timestamp = timestamp

        # Track active IDs in this frame
        active_ids = set()

        for obj in result.tracked_objects:
            track_id = obj.track_id
            active_ids.add(track_id)

            # 1. Get or create tracker
            if track_id not in self._trackers:
                self._trackers[track_id] = _DepthTracker(
                    q_scale=self._q_scale,
                    geom_coeff=self._geom_coeff,
                    depth_coeff=self._depth_coeff,
                    scale_alpha=self._scale_alpha,
                    gate_chi2=self._gate_chi2,
                )
            tracker = self._trackers[track_id]
            tracker.absence_counter = 0

            # 2. Kalman predict step
            tracker.predict(dt)

            # 3. Extract relative depth median from SCDepthV3 depth map ROI
            rel_depth = float("nan")
            if depth_map is not None and obj.bbox != (0, 0, 0, 0):
                rel_depth = self._extract_roi_median(
                    depth_map, obj.bbox, result.frame.shape[:2]
                )
                obj.rel_depth_score = rel_depth

            # 4. Measurement update — Geometric anchor
            geom_valid = not np.isnan(obj.d_geometric_m)
            if geom_valid:
                tracker.update_geometric(obj.d_geometric_m, rel_depth)

            # 5. Measurement update — Monocular relative depth
            if not np.isnan(rel_depth):
                tracker.update_depth(rel_depth)

            # 6. Populate filtered outputs
            if tracker.initialized:
                obj.kalman_distance_m = tracker.distance
                obj.kalman_variance = tracker.variance

        # 7. Garbage collect trackers that haven't been seen for 60 frames
        dead_ids = []
        for tid, tracker in self._trackers.items():
            if tid not in active_ids:
                tracker.absence_counter += 1
                if tracker.absence_counter >= 60:
                    dead_ids.append(tid)

        for tid in dead_ids:
            del self._trackers[tid]

        return result

    def _extract_roi_median(
        self, depth_map: np.ndarray, bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int]
    ) -> float:
        """Map bounding box to depth map coordinates and extract eroded median."""
        orig_h, orig_w = frame_shape
        depth_h, depth_w = depth_map.shape

        x1, y1, x2, y2 = bbox

        # Map to depth map scale
        scale_x = depth_w / orig_w
        scale_y = depth_h / orig_h

        x1_map = int(round(x1 * scale_x))
        y1_map = int(round(y1 * scale_y))
        x2_map = int(round(x2 * scale_x))
        y2_map = int(round(y2 * scale_y))

        # Check bounds
        x1_map = max(0, min(x1_map, depth_w - 1))
        y1_map = max(0, min(y1_map, depth_h - 1))
        x2_map = max(0, min(x2_map, depth_w - 1))
        y2_map = max(0, min(y2_map, depth_h - 1))

        bw = x2_map - x1_map
        bh = y2_map - y1_map

        if bw <= 2 or bh <= 2:
            return float("nan")

        # Eroded windowed-median ROI sampling (20% inner margin)
        margin_x = int(bw * 0.2)
        margin_y = int(bh * 0.2)

        x1_roi = max(0, x1_map + margin_x)
        x2_roi = min(depth_w, x2_map - margin_x)
        y1_roi = max(0, y1_map + margin_y)
        y2_roi = min(depth_h, y2_map - margin_y)

        if x2_roi > x1_roi and y2_roi > y1_roi:
            roi = depth_map[y1_roi:y2_roi, x1_roi:x2_roi]
            return float(np.median(roi))
        else:
            # Fallback to full mapped bounding box if ROI is empty
            roi = depth_map[y1_map:y2_map, x1_map:x2_map]
            return float(np.median(roi)) if roi.size > 0 else float("nan")
