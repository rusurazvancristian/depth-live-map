"""Engine — RepVGG-A0 Person ReID feature extraction."""

import logging
from typing import List, Tuple, Dict
import cv2
import numpy as np
from data_contract import FrameResult
from src.engines.base_engine import BaseEngine

logger = logging.getLogger(__name__)


class ReIDEngine(BaseEngine):
    """RepVGG-A0 Person ReID feature extraction engine running on Hailo NPU.

    Reads:  frame, tracked_objects bboxes
    Writes: ReID embeddings for person detections
    """

    def __init__(
        self,
        multiplexer,
        model_name: str = "reid",
        input_h: int = 256,
        input_w: int = 128,
        embedding_dim: int = 512,
    ) -> None:
        """Initialize ReID engine.

        Args:
            multiplexer: HailoMultiplexer instance managing the VDevice.
            model_name: Key used when loading the ReID model in the multiplexer.
            input_h: Expected input height.
            input_w: Expected input width.
            embedding_dim: Dimension of the output embedding.
        """
        self._mux = multiplexer
        self._model_name = model_name
        self._input_h = input_h
        self._input_w = input_w
        self._embedding_dim = embedding_dim

        # Validate I/O shapes against HEF metadata
        in_shape = self._mux.get_input_shape(model_name)
        out_shape = self._mux.get_output_shape(model_name)

        # Normalize shapes (remove batch dimension if present)
        norm_in = in_shape[1:] if len(in_shape) == 4 else in_shape
        norm_out = out_shape[1:] if len(out_shape) == 2 else out_shape

        if norm_in != (input_h, input_w, 3):
            raise ValueError(
                f"ReID model input shape mismatch! Expected (1, {input_h}, {input_w}, 3) "
                f"or ({input_h}, {input_w}, 3), got {in_shape}"
            )

        if norm_out != (embedding_dim,):
            raise ValueError(
                f"ReID model output shape mismatch! Expected (1, {embedding_dim}) "
                f"or ({embedding_dim},), got {out_shape}"
            )

        # Pre-allocate single crop input buffer: shape (1, 256, 128, 3) uint8
        self._crop_buffer = np.empty((1, input_h, input_w, 3), dtype=np.uint8)

        logger.info(
            "ReIDEngine ready | input=%dx%d | embedding=%d",
            input_w, input_h, embedding_dim
        )

    def extract(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract L2-normalized 512-d ReID embedding from a bounding box region.

        Args:
            frame: Raw BGR frame (numpy array).
            bbox: (x1, y1, x2, y2) in pixel coordinates.

        Returns:
            512-d float32 L2-normalized ReID embedding vector.
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        # Clip crop coordinates to frame boundaries
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        # Check for valid crop region
        if x2 <= x1 or y2 <= y1:
            return np.zeros(self._embedding_dim, dtype=np.float32)

        # Crop the ROI from frame
        crop = frame[y1:y2, x1:x2]

        # Resize crop to (input_w, input_h) -> width x height for cv2.resize
        resized_crop = cv2.resize(crop, (self._input_w, self._input_h))

        # Convert BGR to RGB
        rgb_crop = cv2.cvtColor(resized_crop, cv2.COLOR_BGR2RGB)

        # Copy into pre-allocated input batch buffer
        self._crop_buffer[0] = rgb_crop

        # Run inference
        raw_output = self._mux.infer(self._model_name, self._crop_buffer)

        # Reshape to (512,)
        embedding = raw_output.reshape(self._embedding_dim)

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm
        else:
            embedding = np.zeros_like(embedding)

        return embedding

    def extract_batch(self, frame: np.ndarray, bboxes: List[Tuple[int, Tuple[int, int, int, int]]]) -> Dict[int, np.ndarray]:
        """Extract ReID embeddings for a list of track bboxes.

        Args:
            frame: Raw BGR frame.
            bboxes: List of (track_id, bbox_tuple) where bbox_tuple is (x1, y1, x2, y2).

        Returns:
            Dict mapping track_id -> 512-d L2-normalized embedding.
        """
        embeddings: Dict[int, np.ndarray] = {}
        for track_id, bbox in bboxes:
            embeddings[track_id] = self.extract(frame, bbox)
        return embeddings

    def process(self, result: FrameResult) -> FrameResult:
        """BaseEngine process contract.

        ReID embeddings are extracted in batch by the orchestrator using extract_batch.
        This process method serves as a no-op to satisfy the BaseEngine abstract interface.
        """
        return result
