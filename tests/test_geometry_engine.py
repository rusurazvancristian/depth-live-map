"""Unit tests for GeometryEngine with tracked objects."""

import math
import numpy as np
import pytest

from data_contract import FrameResult, TrackedObject
from src.engines.geometry_engine import GeometryEngine

HEIGHTS_PATH = "src/calibration/object_heights.json"
F_Y = 600.0


def _make_result(bbox_h: float, class_name: str) -> FrameResult:
    obj = TrackedObject(
        track_id=1,
        bbox_height_px=bbox_h,
        class_name=class_name,
    )
    return FrameResult(
        frame=np.zeros((480, 640, 3), dtype=np.uint8),
        timestamp=0.0,
        tracked_objects=[obj],
    )


def test_known_distance_person():
    """Person (1.70m) at 200px, f_y=600 -> d = 1.70 * 600 / 200 = 5.1m."""
    engine = GeometryEngine(focal_length_px=F_Y, heights_path=HEIGHTS_PATH)
    result = engine.process(_make_result(200.0, "person"))
    assert abs(result.tracked_objects[0].d_geometric_m - 5.1) < 0.01


def test_known_distance_chair():
    """Chair (0.85m) at 100px, f_y=600 -> d = 0.85 * 600 / 100 = 5.1m."""
    engine = GeometryEngine(focal_length_px=F_Y, heights_path=HEIGHTS_PATH)
    result = engine.process(_make_result(100.0, "chair"))
    assert abs(result.tracked_objects[0].d_geometric_m - 5.1) < 0.01


def test_zero_bbox_returns_nan():
    engine = GeometryEngine(focal_length_px=F_Y, heights_path=HEIGHTS_PATH)
    result = engine.process(_make_result(0.0, "person"))
    assert math.isnan(result.tracked_objects[0].d_geometric_m)


def test_unknown_class_uses_default():
    """Unknown class should use _default height (0.50m), not crash."""
    engine = GeometryEngine(focal_length_px=F_Y, heights_path=HEIGHTS_PATH)
    result = engine.process(_make_result(100.0, "unknown_object_xyz"))
    expected = 0.50 * F_Y / 100.0
    assert abs(result.tracked_objects[0].d_geometric_m - expected) < 0.01


def test_result_clipped_to_max():
    """Tiny but valid bbox (1.5px) should clip at 100m, not return infinity.
    1.70 * 600 / 1.5 = 680m -> clipped to 100m.
    """
    engine = GeometryEngine(focal_length_px=F_Y, heights_path=HEIGHTS_PATH)
    result = engine.process(_make_result(1.5, "person"))
    assert result.tracked_objects[0].d_geometric_m == 100.0


def test_result_clipped_to_min():
    """Massive bbox (object fills frame) should clip at 0.1m."""
    engine = GeometryEngine(focal_length_px=F_Y, heights_path=HEIGHTS_PATH)
    result = engine.process(_make_result(10000.0, "person"))
    assert result.tracked_objects[0].d_geometric_m >= 0.1
