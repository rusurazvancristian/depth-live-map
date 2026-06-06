import pytest
import numpy as np
import math
from unittest.mock import MagicMock

from data_contract import FrameResult, TrackedObject
from src.engines.base_engine import BaseEngine
from src.engines.depth_engine import DepthEngine
from src.engines.kalman_depth_engine import KalmanDepthEngine, _DepthTracker


def test_depth_tracker_initialization():
    """Verify that _DepthTracker starts uninitialized and sets correct defaults."""
    tracker = _DepthTracker(q_scale=0.1, geom_coeff=0.08, depth_coeff=0.06)
    assert not tracker.initialized
    assert math.isnan(tracker.distance)
    assert math.isnan(tracker.variance)
    assert tracker.scale_factor == 3.0


def test_depth_tracker_update_geometric():
    """Verify that _DepthTracker initializes on first geometric update and runs updates."""
    tracker = _DepthTracker(q_scale=0.1, geom_coeff=0.08, depth_coeff=0.06)
    
    # 1. First update initializes the filter
    success = tracker.update_geometric(5.0)
    assert success
    assert tracker.initialized
    assert tracker.distance == 5.0
    assert tracker.variance == pytest.approx((0.08 * 5.0) ** 2)

    # 2. Subsequent updates update the state
    success2 = tracker.update_geometric(4.8)
    assert success2
    assert tracker.distance < 5.0  # Should pull closer to 4.8


def test_depth_tracker_scale_factor_ema():
    """Verify that scale factor updates via EMA when geometric and relative depth are both present."""
    tracker = _DepthTracker(scale_alpha=0.5)
    
    # First update sets initial scale factor
    tracker.update_geometric(5.0, rel_depth=0.5)
    assert tracker.scale_factor == 10.0  # 5.0 / 0.5

    # Second update changes it via EMA (alpha=0.5)
    # New inst = 6.0 / 0.5 = 12.0
    # Expected scale_factor = 0.5 * 10.0 + 0.5 * 12.0 = 11.0
    tracker.update_geometric(6.0, rel_depth=0.5)
    assert tracker.scale_factor == pytest.approx(11.0)


def test_depth_tracker_gating():
    """Verify that chi-squared gating rejects wildly inaccurate outliers."""
    tracker = _DepthTracker(gate_chi2=3.84)
    tracker.update_geometric(2.0)
    
    # Huge outlier (100 meters) should get rejected
    success = tracker.update_geometric(100.0)
    assert not success
    assert tracker.distance == pytest.approx(2.0)


def test_kalman_depth_engine_process():
    """Verify that KalmanDepthEngine correctly processes FrameResult objects."""
    engine = KalmanDepthEngine(q_scale=0.1, geom_coeff=0.08, depth_coeff=0.06)
    
    # Create mock inputs
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Mock depth map: 256x320 with constant value 0.5
    depth_map = np.full((256, 320), 0.5, dtype=np.float32)
    
    obj = TrackedObject(
        track_id=1,
        bbox=(160, 120, 480, 360),  # centered bounding box
        class_name="person",
        bbox_height_px=240.0,
        d_geometric_m=3.0
    )
    
    result = FrameResult(
        frame=frame,
        timestamp=1.0,
        tracked_objects=[obj],
        depth_map=depth_map
    )
    
    # Process
    out = engine.process(result)
    
    # Assert
    assert len(out.tracked_objects) == 1
    processed_obj = out.tracked_objects[0]
    
    assert processed_obj.rel_depth_score == pytest.approx(0.5)
    assert not math.isnan(processed_obj.kalman_distance_m)
    assert not math.isnan(processed_obj.kalman_variance)
    assert processed_obj.kalman_distance_m > 0.0


def test_depth_engine_process_with_mock_mux():
    """Verify DepthEngine processes frames and stores depth_map correctly."""
    # Create mock multiplexer
    mock_mux = MagicMock()
    mock_mux.get_input_shape.return_value = (1, 256, 320, 3)
    mock_mux.get_output_shape.return_value = (1, 256, 320, 1)
    
    # Mock inference output: all ones shape (1, 256, 320, 1)
    mock_output = np.ones((1, 256, 320, 1), dtype=np.float32)
    mock_mux.infer.return_value = mock_output
    
    engine = DepthEngine(mock_mux)
    
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = FrameResult(frame=frame)
    
    out = engine.process(result)
    
    assert out.depth_map is not None
    assert out.depth_map.shape == (256, 320)
    # Check normalization: since all elements are 1.0, min-max should fallback to 0.5
    assert np.all(out.depth_map == 0.5)
