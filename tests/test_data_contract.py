import numpy as np
import math
from data_contract import FrameResult

def test_frame_result_structure():
    """Validates that FrameResult has all expected fields with correct default values."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = FrameResult(frame=frame, timestamp=1.234)
    
    # Verify input values
    assert np.array_equal(res.frame, frame)
    assert res.timestamp == 1.234
    
    # Verify YOLO defaults
    assert res.bbox == (0, 0, 0, 0)
    assert res.bbox_height_px == 0.0
    assert res.class_id == -1
    assert res.class_name == ""
    assert res.det_confidence == 0.0
    
    # Verify Engine outputs default to NaN or specific empty states
    assert math.isnan(res.d_geometric_m)
    assert math.isnan(res.rel_depth_score)
    assert math.isnan(res.depth_variance)
    assert math.isnan(res.final_distance_m)
    assert math.isnan(res.log_variance)
    
    # Verify confidence intervals defaults
    assert res.confidence_68 == (0.0, 0.0)
    assert res.confidence_95 == (0.0, 0.0)
