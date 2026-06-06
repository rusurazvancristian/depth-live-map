"""Engine — Monocular relative depth estimation via SCDepthV3 on Hailo NPU."""

import logging
from typing import Optional
import cv2
import numpy as np
from data_contract import FrameResult
from src.engines.base_engine import BaseEngine

logger = logging.getLogger(__name__)


class DepthEngine(BaseEngine):
    """Monocular relative depth estimation engine using SCDepthV3 on Hailo NPU.

    Reads:  frame
    Writes: depth_map
    """

    def __init__(
        self,
        multiplexer,
        model_name: str = "depth",
        input_h: int = 256,
        input_w: int = 320,
    ) -> None:
        """Initialize Depth engine.

        Args:
            multiplexer: HailoMultiplexer instance managing the VDevice.
            model_name: Key used when loading the depth model in the multiplexer.
            input_h: Expected input height.
            input_w: Expected input width.
        """
        self._mux = multiplexer
        self._model_name = model_name
        self._input_h = input_h
        self._input_w = input_w

        # Pre-allocate single batch buffer: shape (1, 256, 320, 3) uint8
        self._batch_buffer = np.empty((1, input_h, input_w, 3), dtype=np.uint8)

        logger.info(
            "DepthEngine ready | model=%s | input=%dx%d",
            model_name, input_w, input_h
        )

    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Process a raw BGR frame and return a normalized monocular depth map.

        Args:
            frame: Raw BGR frame.

        Returns:
            Normalized depth map (256, 320) in range [0, 1], or None on error.
        """
        try:
            # Resize full frame to (input_w, input_h) -> width x height for cv2.resize
            resized = cv2.resize(frame, (self._input_w, self._input_h))

            # Convert BGR to RGB
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            # Copy into pre-allocated input batch buffer
            self._batch_buffer[0] = rgb

            # Run inference
            raw_output = self._mux.infer(self._model_name, self._batch_buffer)

            # Reshape output to (256, 320)
            depth_map = raw_output.reshape((self._input_h, self._input_w))

            # Min-max normalize to [0, 1]
            lo, hi = depth_map.min(), depth_map.max()
            if hi - lo < 1e-6:
                norm_depth = np.full_like(depth_map, 0.5)
            else:
                norm_depth = (depth_map - lo) / (hi - lo)

            return norm_depth

        except Exception as exc:
            logger.error("DepthEngine failed to process frame: %s", exc, exc_info=True)
            return None

    def process(self, result: FrameResult) -> FrameResult:
        """Process FrameResult through DepthEngine.

        Extracts the full depth map and writes it to result.depth_map.
        """
        result.depth_map = self.process_frame(result.frame)
        return result
