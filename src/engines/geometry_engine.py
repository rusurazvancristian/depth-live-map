"""Engine 2 — Geometric distance estimator via pinhole camera model. [TRACK A]"""

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
    """Pinhole-model metric distance estimator.

    Reads:  bbox_height_px, class_name
    Writes: d_geometric_m
    """
    _default_h: float
    _heights: Dict[str, float]
    _focal_length_px: float

    def __init__(self, focal_length_px: float, heights_path: str) -> None:
        with open(heights_path) as f:
            raw = json.load(f)
        self._default_h = float(raw.get("_default", 0.50))
        self._heights = {
            k: float(v) for k, v in raw.items()
            if not k.startswith("_")
        }
        self._focal_length_px = focal_length_px
        logger.info(
            "GeometryEngine ready | f_y=%.1f px | %d class heights loaded",
            focal_length_px, len(self._heights),
        )

    def process(self, result: FrameResult) -> FrameResult:
        """Estimate metric distance from bbox height and known object size.

        Args:
            result: FrameResult with bbox_height_px and class_name populated.

        Returns:
            FrameResult with d_geometric_m set.
        """
        try:
            if result.bbox_height_px < 1.0:
                result.d_geometric_m = float("nan")
                return result

            real_h = self._heights.get(result.class_name, self._default_h)
            d = (real_h * self._focal_length_px) / result.bbox_height_px
            result.d_geometric_m = float(np.clip(d, _DISTANCE_MIN_M, _DISTANCE_MAX_M))

        except Exception as exc:
            logger.warning("GeometryEngine failed: %s", exc)
            result.d_geometric_m = float("nan")

        return result
