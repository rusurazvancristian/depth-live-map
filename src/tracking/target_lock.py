"""Target Lock module — manages tracking state machine and ReID golden template matching."""

import logging
from typing import List, Dict, Optional
import numpy as np
from data_contract import TrackedObject

logger = logging.getLogger(__name__)


class TargetLock:
    """Manages lock state machine for a specific class using ReID embeddings."""

    def __init__(
        self,
        target_class = "person",
        stable_frames: int = 5,
        cosine_thresh: float = 0.85,
        search_timeout: int = 90,
    ) -> None:
        """Initialize TargetLock.

        Args:
            target_class: Class name or tuple/list of class names to target.
            stable_frames: Number of consecutive frames the same track must be present to lock.
            cosine_thresh: Threshold for cosine similarity matching.
            search_timeout: Number of frames allowed to search for a lost target.
        """
        self.target_classes = (
            (target_class,) if isinstance(target_class, str) else tuple(target_class)
        )
        self.stable_frames = stable_frames
        self.cosine_thresh = cosine_thresh
        self.search_timeout = search_timeout

        self.target_id: int = -1
        self.status: str = "IDLE"  # IDLE | LOCKED | SEARCHING | LOST
        self.golden_template: Optional[np.ndarray] = None

        self._stability_counter: int = 0
        self._search_counter: int = 0
        self._candidate_id: int = -1

    def update(
        self, tracked_objects: List[TrackedObject], reid_vectors: Dict[int, np.ndarray]
    ) -> None:
        """Update the target lock state machine.

        Args:
            tracked_objects: List of active TrackedObjects.
            reid_vectors: Dict mapping track ID to 512-d ReID embedding.
        """
        if self.status == "IDLE":
            self._handle_idle(tracked_objects, reid_vectors)
        elif self.status == "LOCKED":
            self._handle_locked(tracked_objects)
        elif self.status == "SEARCHING":
            self._handle_searching(tracked_objects, reid_vectors)
        elif self.status == "LOST":
            self._handle_lost()
            self._handle_idle(tracked_objects, reid_vectors)

    def manual_lock(self, track_id: int, reid_vector: np.ndarray) -> None:
        """Manually force the tracker to lock on a specific track ID.

        Args:
            track_id: The ByteTrack track ID to lock.
            reid_vector: The 512-d L2-normalized embedding.
        """
        logger.info("Manual override: forcing lock on track ID %d", track_id)
        self.target_id = track_id
        self.status = "LOCKED"
        self.golden_template = reid_vector.copy()
        self._stability_counter = 0
        self._search_counter = 0
        self._candidate_id = -1

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two L2-normalized vectors.

        Returns:
            The cosine similarity (dot product since vectors are L2-normalized).
        """
        return float(np.dot(a, b))

    def _handle_idle(
        self, tracked_objects: List[TrackedObject], reid_vectors: Dict[int, np.ndarray]
    ) -> None:
        """Scan for candidate of target class and check stability."""
        # Find first object matching target class
        target_objs = [obj for obj in tracked_objects if obj.class_name in self.target_classes]

        if not target_objs:
            self._candidate_id = -1
            self._stability_counter = 0
            return

        # Pick the first one
        cand = target_objs[0]
        cand_id = cand.track_id

        if cand_id == self._candidate_id:
            self._stability_counter += 1
        else:
            self._candidate_id = cand_id
            self._stability_counter = 1

        logger.debug(
            "IDLE: Candidate ID=%d, stability count=%d/%d",
            self._candidate_id, self._stability_counter, self.stable_frames
        )

        if self._stability_counter >= self.stable_frames:
            # Check if ReID vector is available
            if self._candidate_id in reid_vectors:
                self.target_id = self._candidate_id
                self.golden_template = reid_vectors[self.target_id].copy()
                self.status = "LOCKED"
                self._stability_counter = 0
                logger.info(
                    "Transitioned to LOCKED on track ID %d. Golden template captured.",
                    self.target_id
                )
            else:
                logger.debug("Candidate ID=%d stable but ReID vector not yet available", self._candidate_id)

    def _handle_locked(self, tracked_objects: List[TrackedObject]) -> None:
        """Check if locked target is still present."""
        target_present = any(obj.track_id == self.target_id for obj in tracked_objects)

        if not target_present:
            logger.info("Locked target ID %d lost. Transitioning to SEARCHING.", self.target_id)
            self.status = "SEARCHING"
            self._search_counter = 0

    def _handle_searching(
        self, tracked_objects: List[TrackedObject], reid_vectors: Dict[int, np.ndarray]
    ) -> None:
        """Search for target template among current detections."""
        self._search_counter += 1
        
        if self._search_counter > self.search_timeout:
            logger.warning("Search timeout reached. Target ID %d is LOST.", self.target_id)
            self.status = "LOST"
            return

        logger.debug("SEARCHING: frame %d/%d", self._search_counter, self.search_timeout)

        # Compare ReID templates of all current target_class detections
        best_score = -1.0
        best_id = -1
        best_vector = None

        for obj in tracked_objects:
            if obj.class_name not in self.target_classes:
                continue
            
            obj_id = obj.track_id
            if obj_id in reid_vectors and self.golden_template is not None:
                sim = self.cosine_similarity(self.golden_template, reid_vectors[obj_id])
                if sim > best_score:
                    best_score = sim
                    best_id = obj_id
                    best_vector = reid_vectors[obj_id]

        if best_score >= self.cosine_thresh and best_id != -1 and best_vector is not None:
            logger.info(
                "Target recovered! Re-locked onto track ID %d (was %d, ReID match similarity=%.3f)",
                best_id, self.target_id, best_score
            )
            self.target_id = best_id
            self.golden_template = best_vector.copy()  # Update template with latest appearance
            self.status = "LOCKED"
            self._search_counter = 0

    def _handle_lost(self) -> None:
        """Reset lock state parameters."""
        self.target_id = -1
        self.status = "IDLE"
        self.golden_template = None
        self._stability_counter = 0
        self._search_counter = 0
        self._candidate_id = -1
