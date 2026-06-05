import numpy as np
import cv2
import logging
import math
from typing import Tuple

from src.engines.base_engine import BaseEngine
from data_contract import FrameResult

logger = logging.getLogger(__name__)

# Try importing hailo_platform. If not present (e.g. during offline testing or Colab dev), 
# we degrade gracefully.
try:
    from hailo_platform import InferVStreams, ConfigureParams, VDevice, HEF
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    logger.warning("hailo_platform not found. DepthEngine will run in dry-run/fallback mode.")


class DepthEngine(BaseEngine):
    """Monocular relative depth estimation engine using Depth Anything V2 on Hailo NPU.
    
    Reads: frame, bbox
    Writes: rel_depth_score, depth_variance
    """

    def __init__(self, hef_path: str, model_input_size: int = 224, vdevice: 'VDevice' = None) -> None:
        """Initializes the DepthEngine.
        
        Args:
            hef_path: Path to the pre-compiled Depth Anything V2 .hef file.
            model_input_size: Expected input size (square) of the model.
            vdevice: Shared VDevice instance. If None, a new one will be created.
        """
        self.hef_path = hef_path
        self.model_input_size = model_input_size
        self.vdevice = vdevice
        self._pipeline = None
        self._ng_activated = False

        if HAILO_AVAILABLE:
            try:
                self._init_hailo()
            except Exception as e:
                logger.error(f"Failed to initialize Hailo NPU for DepthEngine: {e}")
                self._pipeline = None

    def _init_hailo(self) -> None:
        """Initializes the Hailo NPU session, loading HEF and configuring streams."""
        self._hef = HEF(self.hef_path)
        if self.vdevice is None:
            self.vdevice = VDevice()
            
        self._configure_params = ConfigureParams.create_from_hef(
            self._hef, 
            interface=self.vdevice.get_interface() if hasattr(self.vdevice, 'get_interface') else 1
        )
        self._network_groups = self.vdevice.configure(self._hef, self._configure_params)
        self._ng = self._network_groups[0]
        self._ng_params = self._ng.create_params()
        
        # Get vstream names
        self._input_name = self._hef.get_input_vstream_infos()[0].name
        self._output_name = self._hef.get_output_vstream_infos()[0].name
        
        # We will activate the network group and create InferVStreams during inference/processing.
        # However, to be thread-safe/stateless per-frame, we manage this lifecycle.
        # The orchestrator will typically handle the main context, but we prepare local access.
        
    def _expand_bbox(
        self,
        bbox: tuple[int, int, int, int],
        frame_shape: tuple[int, int],
        margin: float = 0.1,
    ) -> tuple[int, int, int, int]:
        """Expand bbox by margin fraction, clipped to frame bounds."""
        x1, y1, x2, y2 = bbox
        h, w = frame_shape[:2]
        bw, bh = x2 - x1, y2 - y1
        mx, my = int(bw * margin), int(bh * margin)
        return (
            max(0, x1 - mx),
            max(0, y1 - my),
            min(w, x2 + mx),
            min(h, y2 + my),
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
            # Crop region of interest with 10% context margin
            x1, y1, x2, y2 = self._expand_bbox(result.bbox, result.frame.shape, margin=0.1)
            crop = result.frame[y1:y2, x1:x2]
            
            if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
                result.rel_depth_score = float("nan")
                result.depth_variance = float("nan")
                return result

            # Run NPU inference if active and available
            if HAILO_AVAILABLE and hasattr(self, '_ng'):
                # Resize to model input size
                crop_resized = cv2.resize(crop, (self.model_input_size, self.model_input_size))
                batch = np.expand_dims(crop_resized, axis=0)  # (1, 224, 224, 3) BGR uint8
                
                # Import required components locally to ensure they load in runtime context
                from hailo_platform import InferVStreams, InputVStreamParams, OutputVStreamParams, FormatType
                
                in_p = InputVStreamParams.make_from_network_group(self._ng, quantized=False, format_type=FormatType.UINT8)
                out_p = OutputVStreamParams.make_from_network_group(self._ng, quantized=False, format_type=FormatType.FLOAT32)
                
                with InferVStreams(self._ng, in_p, out_p) as pipeline:
                    with self._ng.activate(self._ng_params):
                        infer_results = pipeline.infer({self._input_name: batch})
                        raw_depth_map = infer_results[self._output_name][0]  # (224, 224, 1) float32
                        
                # Resize depth map back to crop coordinates
                depth_map = cv2.resize(raw_depth_map, (crop.shape[1], crop.shape[0]))
            else:
                # Dry-run fallback: generate a synthetic depth map (for testing/validation)
                logger.debug("Running DepthEngine dry-run fallback.")
                # Simple vertical gradient simulating depth for dry-run
                depth_map = np.linspace(0.1, 0.9, crop.shape[0])[:, None]
                depth_map = np.repeat(depth_map, crop.shape[1], axis=1).astype(np.float32)

            # Min-max normalize depth map to [0, 1]
            lo, hi = depth_map.min(), depth_map.max()
            if hi - lo < 1e-6:
                norm_depth = np.full_like(depth_map, 0.5)
            else:
                norm_depth = (depth_map - lo) / (hi - lo)

            # Compute statistics
            result.rel_depth_score = float(np.median(norm_depth))
            result.depth_variance = float(np.var(norm_depth))

        except Exception as e:
            logger.error(f"Error in DepthEngine: {e}")
            result.rel_depth_score = float("nan")
            result.depth_variance = float("nan")

        return result
