import numpy as np
import math
from data_contract import FrameResult, Detection, TrackedObject


def test_frame_result_structure():
    """Validates that FrameResult has all expected fields with correct default values."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = FrameResult(frame=frame, timestamp=1.234)

    # Verify inputs
    assert np.array_equal(res.frame, frame)
    assert res.timestamp == 1.234

    # Verify Stage outputs defaults
    assert isinstance(res.detections, list)
    assert len(res.detections) == 0
    assert isinstance(res.tracked_objects, list)
    assert len(res.tracked_objects) == 0
    assert res.depth_map is None

    # Verify Target Lock defaults
    assert res.target_id == -1
    assert res.target_status == "IDLE"
    assert math.isnan(res.target_distance_m)
    assert not res.target_is_arrived


def test_detection_structure():
    """Validates Detection dataclass structure."""
    det = Detection(
        bbox=(10, 20, 100, 200),
        confidence=0.92,
        class_id=0,
        class_name="person"
    )
    assert det.bbox == (10, 20, 100, 200)
    assert det.confidence == 0.92
    assert det.class_id == 0
    assert det.class_name == "person"


def test_tracked_object_structure():
    """Validates TrackedObject dataclass structure."""
    obj = TrackedObject(
        track_id=4,
        bbox=(10, 20, 100, 200),
        confidence=0.88,
        class_id=0,
        class_name="person",
        bbox_height_px=180.0
    )
    assert obj.track_id == 4
    assert obj.bbox == (10, 20, 100, 200)
    assert obj.confidence == 0.88
    assert obj.class_id == 0
    assert obj.class_name == "person"
    assert obj.bbox_height_px == 180.0
    assert math.isnan(obj.d_geometric_m)
    assert math.isnan(obj.rel_depth_score)
    assert math.isnan(obj.kalman_distance_m)
    assert math.isnan(obj.kalman_variance)
