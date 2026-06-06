from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class FrameState:
    """Single processed keyframe with all derived data."""
    frame_id: int
    timestamp: float
    image: np.ndarray                           # H×W×3 BGR
    depth_map: Optional[np.ndarray] = None     # H×W float32, meters
    keypoints: Optional[list] = None           # list[cv2.KeyPoint]
    descriptors: Optional[np.ndarray] = None   # N×32 uint8 ORB
    pose_c2w: Optional[np.ndarray] = None      # 4×4 float64, camera→world SE3


@dataclass
class MapState:
    """Accumulated bird's-eye-view map."""
    bev_count: np.ndarray                       # H×W int32, observation count
    bev_height_acc: np.ndarray                  # H×W float64, sum of world-Y per cell
    trajectory: list = field(default_factory=list)  # list[np.ndarray] 3D cam positions
    frame_count: int = 0
