"""Engine 3 — Monocular relative depth estimation via SCDepthV3 on Hailo NPU. [TRACK B]"""

import logging
import numpy as np
import cv2

try:
    from hailo_platform import (
        VDevice, HEF, ConfigureParams,
        InputVStreamParams, OutputVStreamParams,
        FormatType, HailoStreamInterface, InferVStreams,
    )
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("hailo_platform not found. DepthEngine will run in mock/fallback mode.")

from src.engines.base_engine import BaseEngine
from data_contract import FrameResult

logger = logging.getLogger(__name__)


class DepthEngine(BaseEngine):
    """Monocular relative depth estimation engine using SCDepthV3 on Hailo NPU.
    
    Reads:  frame, bbox
    Writes: rel_depth_score, depth_variance
    """

    def __init__(
        self,
        hef_path: str,
        model_input_height: int = 256,
        model_input_width: int = 320,
        vdevice: 'VDevice' = None,
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
        self._pipeline = None
        self._owns_device = False

        if HAILO_AVAILABLE:
            try:
                self._init_hailo()
            except Exception as e:
                logger.error(f"Failed to initialize Hailo NPU for DepthEngine: {e}")
                self._ng = None
        else:
            logger.info("DepthEngine initialized in mock mode.")

    def _init_hailo(self) -> None:
        """Initializes the Hailo NPU session, loading HEF and configuring streams."""
        self._hef = HEF(self.hef_path)
        self._owns_device = self.vdevice is None
        if self.vdevice is None:
            self.vdevice = VDevice()
            
        self._configure_params = ConfigureParams.create_from_hef(
            self._hef, 
            interface=HailoStreamInterface.PCIe
        )
        self._network_groups = self.vdevice.configure(self._hef, self._configure_params)
        self._ng = self._network_groups[0]
        self._ng_params = self._ng.create_params()
        
        # Get vstream names
        self._input_name = self._hef.get_input_vstream_infos()[0].name
        self._output_name = self._hef.get_output_vstream_infos()[0].name

        logger.info(
            "DepthEngine Hailo initialized: %s | input=%s | output=%s",
            self.hef_path, self._input_name, self._output_name
        )

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
            if HAILO_AVAILABLE and hasattr(self, '_ng') and self._ng is not None:
                # Resize to model input size (SCDepth expects HxW e.g. 256x320)
                frame_resized = cv2.resize(frame_bgr, (self.model_input_width, self.model_input_height))
                batch = np.expand_dims(frame_resized, axis=0)  # (1, H, W, 3) BGR uint8
                
                in_p = InputVStreamParams.make_from_network_group(self._ng, quantized=False, format_type=FormatType.UINT8)
                out_p = OutputVStreamParams.make_from_network_group(self._ng, quantized=False, format_type=FormatType.FLOAT32)
                
                with InferVStreams(self._ng, in_p, out_p) as pipeline:
                    with self._ng.activate(self._ng_params):
                        infer_results = pipeline.infer({self._input_name: batch})
                        # Raw output depth map shape is typically (1, H, W, 1) or flat
                        raw_depth_map = infer_results[self._output_name][0]
                
                # Reshape to expected (H, W)
                depth_map = raw_depth_map.reshape((self.model_input_height, self.model_input_width))
            else:
                # Dry-run fallback: generate a synthetic depth map (vertical gradient + center object depth)
                # Simple vertical gradient simulating depth for dry-run
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
        if HAILO_AVAILABLE and self._owns_device and hasattr(self, "_device") and self._device is not None:
            try:
                del self._device
            except Exception:
                pass
