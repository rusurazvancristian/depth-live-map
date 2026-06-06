"""Pure-Python ByteTrack multi-object tracker."""

import logging
from typing import List, Tuple, Dict
import numpy as np
from scipy.optimize import linear_sum_assignment
from data_contract import TrackedObject

logger = logging.getLogger(__name__)

COCO_CLASSES: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


class KalmanBoxTracker:
    """State tracker for a single bounding box using a Kalman Filter."""

    count = 0

    def __init__(self, bbox: np.ndarray) -> None:
        """Initialize tracker with bbox [x1, y1, x2, y2]."""
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count

        # State vector: [x_center, y_center, aspect_ratio, height, vx, vy, va, vh]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x_c = bbox[0] + w / 2.0
        y_c = bbox[1] + h / 2.0
        a = w / (h + 1e-6)

        self.x = np.array([[x_c], [y_c], [a], [h], [0], [0], [0], [0]], dtype=np.float32)

        # Transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)

        # Measurement matrix (observes position, aspect ratio, height)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)

        # Covariance matrix P
        self.P = np.eye(8, dtype=np.float32)
        self.P[4:, 4:] *= 1000.0  # High initial uncertainty for velocity
        self.P *= 10.0

        # Process noise Q
        self.Q = np.eye(8, dtype=np.float32)
        self.Q[4:, 4:] *= 0.01

        # Measurement noise R
        self.R = np.eye(4, dtype=np.float32)
        self.R[2, 2] *= 10.0  # Allow aspect ratio to change with less trust

    def predict(self) -> np.ndarray:
        """Predict the next state of the bounding box."""
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.get_state()

    def update(self, bbox: np.ndarray) -> None:
        """Update tracker with measurement bbox [x1, y1, x2, y2]."""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x_c = bbox[0] + w / 2.0
        y_c = bbox[1] + h / 2.0
        a = w / (h + 1e-6)
        z = np.array([[x_c], [y_c], [a], [h]], dtype=np.float32)

        # Standard Kalman update equations
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = np.dot(np.eye(8, dtype=np.float32) - np.dot(K, self.H), self.P)

    def get_state(self) -> np.ndarray:
        """Convert state vector [x_c, y_c, a, h] back to bbox [x1, y1, x2, y2]."""
        x_c = self.x[0, 0]
        y_c = self.x[1, 0]
        a = self.x[2, 0]
        h = self.x[3, 0]
        w = a * h
        x1 = x_c - w / 2.0
        y1 = y_c - h / 2.0
        x2 = x_c + w / 2.0
        y2 = y_c + h / 2.0
        return np.array([x1, y1, x2, y2], dtype=np.float32)


class STrack:
    """Single track representation in ByteTracker."""

    def __init__(self, bbox: np.ndarray, score: float, class_id: int) -> None:
        self.tracker = KalmanBoxTracker(bbox)
        self.track_id = self.tracker.id
        self.score = score
        self.class_id = class_id
        
        # 1: Tracked, 2: Lost
        self.state = 1
        self.hits = 1
        self.time_since_update = 0

    def predict(self) -> np.ndarray:
        """Predict track state."""
        self.time_since_update += 1
        return self.tracker.predict()

    def update(self, bbox: np.ndarray, score: float) -> None:
        """Update track with new detection."""
        self.tracker.update(bbox)
        self.score = score
        self.state = 1
        self.hits += 1
        self.time_since_update = 0

    def mark_lost(self) -> None:
        """Mark track as lost."""
        self.state = 2

    @property
    def bbox(self) -> np.ndarray:
        return self.tracker.get_state()


class ByteTracker:
    """ByteTrack multi-object tracker."""

    def __init__(
        self,
        high_thresh: float = 0.6,
        low_thresh: float = 0.1,
        match_thresh: float = 0.8,
        buffer: int = 30,
        min_hits: int = 3,
    ) -> None:
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.max_lost_frames = buffer
        self.min_hits = min_hits

        self.tracked_tracks: List[STrack] = []
        self.lost_tracks: List[STrack] = []

    def update(self, detections: np.ndarray, class_ids: np.ndarray) -> List[TrackedObject]:
        """Update tracker with detections and class_ids.

        Args:
            detections: shape (N, 5) — [x1, y1, x2, y2, confidence]
            class_ids: shape (N,) — class ID integers

        Returns:
            List of TrackedObject from data_contract.
        """
        # 1. Split detections into high-score and low-score
        high_dets: List[Tuple[np.ndarray, float, int]] = []
        low_dets: List[Tuple[np.ndarray, float, int]] = []

        for det, class_id in zip(detections, class_ids):
            bbox = det[:4]
            score = float(det[4])
            if score >= self.high_thresh:
                high_dets.append((bbox, score, class_id))
            elif score >= self.low_thresh:
                low_dets.append((bbox, score, class_id))

        # 2. Predict positions of existing tracks
        for track in self.tracked_tracks:
            track.predict()
        for track in self.lost_tracks:
            track.predict()

        # Combine tracked and lost tracks for first association
        pool_tracks = self.tracked_tracks + self.lost_tracks

        # 3. First Association: High-score detections vs All Active Tracks
        pool_bboxes = np.array([t.bbox for t in pool_tracks]) if pool_tracks else np.empty((0, 4))
        high_bboxes = np.array([d[0] for d in high_dets]) if high_dets else np.empty((0, 4))

        cost_matrix = 1.0 - self._iou_batch(high_bboxes, pool_bboxes)
        
        # Hungarian matching
        matches, unmatched_dets, unmatched_tracks = self._linear_assignment(
            cost_matrix, thresh=self.match_thresh
        )

        activated_tracks: List[STrack] = []
        for det_idx, track_idx in matches:
            track = pool_tracks[track_idx]
            bbox, score, class_id = high_dets[det_idx]
            track.update(bbox, score)
            activated_tracks.append(track)

        # 4. Second Association: Low-score detections vs remaining active (Tracked only)
        # We don't associate low-score detections with lost tracks.
        remaining_tracked_indices = [
            i for i in unmatched_tracks if pool_tracks[i].state == 1
        ]
        remaining_tracked = [pool_tracks[i] for i in remaining_tracked_indices]

        rem_tracked_bboxes = np.array([t.bbox for t in remaining_tracked]) if remaining_tracked else np.empty((0, 4))
        low_bboxes = np.array([d[0] for d in low_dets]) if low_dets else np.empty((0, 4))

        cost_matrix_low = 1.0 - self._iou_batch(low_bboxes, rem_tracked_bboxes)
        
        matches_low, unmatched_dets_low, unmatched_tracks_low = self._linear_assignment(
            cost_matrix_low, thresh=0.5  # lower IoU threshold for low confidence
        )

        for det_idx, track_idx in matches_low:
            track = remaining_tracked[track_idx]
            bbox, score, class_id = low_dets[det_idx]
            track.update(bbox, score)
            activated_tracks.append(track)

        # Any tracked track that is unmatched in both steps is marked lost
        lost_from_tracked = [
            remaining_tracked[i] for i in unmatched_tracks_low
        ]
        for track in lost_from_tracked:
            track.mark_lost()

        # Unmatched lost tracks remain lost
        unmatched_lost = [
            pool_tracks[i] for i in unmatched_tracks if pool_tracks[i].state == 2
        ]

        # 5. Initialize new tracks from unmatched high-score detections
        for det_idx in unmatched_dets:
            bbox, score, class_id = high_dets[det_idx]
            track = STrack(bbox, score, class_id)
            activated_tracks.append(track)

        # Update lists: filter out lost or removed tracks
        self.tracked_tracks = [t for t in activated_tracks if t.state == 1]
        self.lost_tracks = []

        # Keep lost tracks if they haven't exceeded buffer time
        for t in lost_from_tracked + unmatched_lost:
            if t.time_since_update <= self.max_lost_frames:
                self.lost_tracks.append(t)

        # Generate outputs for confirmed/tracked tracks
        tracked_objects: List[TrackedObject] = []
        for track in self.tracked_tracks:
            if track.hits >= self.min_hits:
                bbox_tuple = (
                    int(round(track.bbox[0])),
                    int(round(track.bbox[1])),
                    int(round(track.bbox[2])),
                    int(round(track.bbox[3]))
                )
                height = float(bbox_tuple[3] - bbox_tuple[1])
                class_name = (
                    COCO_CLASSES[track.class_id]
                    if track.class_id < len(COCO_CLASSES)
                    else str(track.class_id)
                )
                
                tracked_objects.append(
                    TrackedObject(
                        track_id=track.track_id,
                        bbox=bbox_tuple,
                        confidence=track.score,
                        class_id=track.class_id,
                        class_name=class_name,
                        bbox_height_px=height,
                    )
                )

        return tracked_objects

    def _iou_batch(self, bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
        """Compute IoU matrix between test and ground truth bounding boxes."""
        if bb_test.size == 0 or bb_gt.size == 0:
            return np.empty((bb_test.shape[0], bb_gt.shape[0]))

        bb_test = np.expand_dims(bb_test, 1)  # (N, 1, 4)
        bb_gt = np.expand_dims(bb_gt, 0)     # (1, M, 4)

        xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
        yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
        xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
        yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        wh = w * h

        area_test = (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
        area_gt = (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])

        iou = wh / (area_test + area_gt - wh + 1e-16)
        return iou

    def _linear_assignment(
        self, cost_matrix: np.ndarray, thresh: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve bipartite matching using Hungarian algorithm."""
        if cost_matrix.size == 0:
            return (
                np.empty((0, 2), dtype=int),
                np.arange(cost_matrix.shape[0]),
                np.arange(cost_matrix.shape[1]),
            )

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches = []
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= thresh:
                matches.append([r, c])

        matches = np.array(matches, dtype=int)
        if matches.size == 0:
            matches = np.empty((0, 2), dtype=int)

        unmatched_a = np.setdiff1d(
            np.arange(cost_matrix.shape[0]),
            matches[:, 0] if matches.size > 0 else [],
        )
        unmatched_b = np.setdiff1d(
            np.arange(cost_matrix.shape[1]),
            matches[:, 1] if matches.size > 0 else [],
        )
        return matches, unmatched_a, unmatched_b
