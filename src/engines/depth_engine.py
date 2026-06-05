"""Engine 3 — Monocular relative depth estimation via SCDepthV3 on Hailo NPU. [TRACK B]"""

import logging
from typing import Any, Optional, TYPE_CHECKING
import numpy as np
import cv2

if TYPE_CHECKING:
    from hailo_platform import VDevice

try:
    from hailo_platform import VDevice as ActualVDevice
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("hailo_platform not found. DepthEngine will run in mock/fallback mode.")

from src.engines.base_engine import BaseEngine
from data_contract import FrameResult
from src.hailo_inference.hef_loader import HEFModel

logger = logging.getLogger(__name__)


class DepthEngine(BaseEngine):
    """Monocular relative depth estimation engine using SCDepthV3 on Hailo NPU.
    
    Reads:  frame, bbox
    Writes: rel_depth_score, depth_variance
    """
    hef_path: str
    model_input_height: int
    model_input_width: int
    vdevice: Optional['VDevice']
    _model: Optional[HEFModel]
    _pipeline: Any
    _pipeline_ctx: Any
    _input_name: str
    _output_name: str
    
    # Pre-allocated BGR batch buffer for zero-allocation reuse
    _batch_bgr: Optional[np.ndarray]

    def __init__(
        self,
        hef_path: str,
        model_input_height: int = 256,
        model_input_width: int = 320,
        vdevice: Optional['VDevice'] = None,
    ) -> None:
        """Initializes the DepthEngine.
        
        Args:
            hef_path: Path to the pre-compiled SCDepthV3 .hef file.
            model_input_height: Expected input height of the model (default 256).
            model_input_width: Expected input width of the model (default 320).
            vdevice: Shared VDevice instance. If None, a new one will be created.
        """
        self.hef_path = hef_path
        self.model_input_height = model_input_height
        self.model_input_width = model_input_width
        self.vdevice = vdevice
        self._model = None
        self._pipeline = None
        self._pipeline_ctx = None
        self._batch_bgr = None

        if HAILO_AVAILABLE:
            try:
                self._model = HEFModel(hef_path, device=vdevice)
                self._input_name = self._model.input_name
                self._output_name = self._model.output_name
                logger.info(
                    "DepthEngine Hailo initialized: %s | input=%s | output=%s",
                    self.hef_path, self._input_name, self._output_name
                )
            except Exception as e:
                logger.error(f"Failed to initialize Hailo NPU for DepthEngine: {e}")
                self._model = None
        else:
            logger.info("DepthEngine initialized in mock mode.")

    def start(self) -> None:
        """Initialize session and activate NPU network group once before the frame loop."""
        if HAILO_AVAILABLE and self._model is not None:
            try:
                self._pipeline_ctx = self._model.session()
                self._pipeline = self._pipeline_ctx.__enter__()
                logger.info("DepthEngine session started successfully.")
            except Exception as e:
                logger.error(f"Failed to start DepthEngine NPU session: {e}")
                self._pipeline = None
                self._pipeline_ctx = None

    def stop(self) -> None:
        """Release session and deactivate NPU network group after the frame loop."""
        if self._pipeline_ctx is not None:
            try:
                self._pipeline_ctx.__exit__(None, None, None)
                logger.info("DepthEngine session stopped successfully.")
            except Exception as e:
                logger.error(f"Error stopping DepthEngine session: {e}")
            finally:
                self._pipeline = None
                self._pipeline_ctx = None

    def process(self, result: FrameResult) -> FrameResult:
        """Extracts relative depth score and variance from the frame within the bbox.
        
        Args:
            result: The current FrameResult object.
            
        Returns:
            The modified FrameResult object with depth fields populated.
        """
        # Ensure we have a valid bbox detection
        if result.bbox == (0, 0, 0, 0) or result.bbox_height_px < 1.0:
            result.rel_depth_score = float("nan")
            result.depth_variance = float("nan")
            return result

        try:
            frame_bgr = result.frame
            orig_h, orig_w = frame_bgr.shape[:2]

            # 1. Run NPU inference or fallback to mock
            if HAILO_AVAILABLE and self._pipeline is not None:
                # Lazy allocate batch buffer if not done
                if self._batch_bgr is None:
                    self._batch_bgr = np.empty((1, self.model_input_height, self.model_input_width, 3), dtype=np.uint8)
                
                # Resize directly to preallocated BGR batch slice
                cv2.resize(frame_bgr, (self.model_input_width, self.model_input_height), 
                           dst=self._batch_bgr[0], interpolation=cv2.INTER_LINEAR)
                
                # Run inference using the pre-activated pipeline context
                raw_outputs = self._pipeline.infer({self._input_name: self._batch_bgr})
                raw_depth_map = raw_outputs[self._output_name][0]
                
                # Reshape to expected (H, W)
                depth_map = raw_depth_map.reshape((self.model_input_height, self.model_input_width))
            else:
                # Dry-run fallback: generate a synthetic depth map (vertical gradient + center object depth)
                depth_map = np.linspace(0.8, 0.2, self.model_input_height)[:, None]
                depth_map = np.repeat(depth_map, self.model_input_width, axis=1).astype(np.float32)
                # Mock a closer object in the center area
                center_y1, center_y2 = int(self.model_input_height * 0.2), int(self.model_input_height * 0.8)
                center_x1, center_x2 = int(self.model_input_width * 0.2), int(self.model_input_width * 0.8)
                depth_map[center_y1:center_y2, center_x1:center_x2] = 0.15

            # 2. Min-max normalize depth map to [0, 1]
            lo, hi = depth_map.min(), depth_map.max()
            if hi - lo < 1e-6:
                norm_depth = np.full_like(depth_map, 0.5)
            else:
                norm_depth = (depth_map - lo) / (hi - lo)

            # 3. Map the original bbox to the depth map coordinates
            x1, y1, x2, y2 = result.bbox
            scale_y = self.model_input_height / orig_h
            scale_x = self.model_input_width / orig_w
            
            x1_map = int(x1 * scale_x)
            y1_map = int(y1 * scale_y)
            x2_map = int(x2 * scale_x)
            y2_map = int(y2 * scale_y)

            # 4. Perform eroded windowed-median ROI sampling (20% inner margin)
            bw = x2_map - x1_map
            bh = y2_map - y1_map
            margin_x = int(bw * 0.2)
            margin_y = int(bh * 0.2)

            y1_roi = max(0, y1_map + margin_y)
            y2_roi = min(self.model_input_height, y2_map - margin_y)
            x1_roi = max(0, x1_map + margin_x)
            x2_roi = min(self.model_input_width, x2_map - margin_x)

            # 5. Extract statistics
            if y2_roi > y1_roi and x2_roi > x1_roi:
                roi = norm_depth[y1_roi:y2_roi, x1_roi:x2_roi]
                result.rel_depth_score = float(np.median(roi))
                result.depth_variance = float(np.var(roi))
            else:
                result.rel_depth_score = float("nan")
                result.depth_variance = float("nan")

        except Exception as e:
            logger.error(f"Error in DepthEngine: {e}")
            result.rel_depth_score = float("nan")
            result.depth_variance = float("nan")

        return result

    def __del__(self) -> None:
        self.stop()
