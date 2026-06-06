"""Engine 2 — Geometric distance estimator via pinhole camera model."""

import json
import logging
from typing import Dict
import numpy as np
from data_contract import FrameResult
from src.engines.base_engine import BaseEngine

logger = logging.getLogger(__name__)

_DISTANCE_MIN_M: float = 0.1
_DISTANCE_MAX_M: float = 100.0


class GeometryEngine(BaseEngine):
    """Pinhole-model metric distance estimator for all tracked objects.

    Reads:  tracked_objects[i].bbox_height_px, class_name
    Writes: tracked_objects[i].d_geometric_m
    """

    def __init__(self, focal_length_px: float, heights_path: str) -> None:
        """Initialize Geometry engine.

        Args:
            focal_length_px: Pinhole camera focal length in pixels.
            heights_path: Path to json file mapping class name to real height.
        """
        try:
            with open(heights_path) as f:
                raw = json.load(f)
            self._default_h = float(raw.get("_default", 0.50))
            self._heights = {
                k: float(v) for k, v in raw.items()
                if not k.startswith("_")
            }
        except Exception as exc:
            logger.error("Failed to load object heights from %s: %s", heights_path, exc)
            self._default_h = 1.70  # Default to person height
            self._heights = {"person": 1.70}

        self._focal_length_px = focal_length_px
        logger.info(
            "GeometryEngine ready | focal_length=%.1f px | default_height=%.2f m | %d classes loaded",
            focal_length_px, self._default_h, len(self._heights),
        )

    def process(self, result: FrameResult) -> FrameResult:
        """Estimate metric distance for all tracked objects using pinhole camera model.

        Args:
            result: FrameResult containing tracked_objects list.

        Returns:
            FrameResult with d_geometric_m populated for each tracked object.
        """
        for obj in result.tracked_objects:
            try:
                if obj.bbox_height_px < 1.0:
                    obj.d_geometric_m = float("nan")
                    continue

                real_h = self._heights.get(obj.class_name, self._default_h)
                d = (real_h * self._focal_length_px) / obj.bbox_height_px
                obj.d_geometric_m = float(np.clip(d, _DISTANCE_MIN_M, _DISTANCE_MAX_M))

            except Exception as exc:
                logger.debug("Failed to calculate distance for track %d: %s", obj.track_id, exc)
                obj.d_geometric_m = float("nan")

        return result
