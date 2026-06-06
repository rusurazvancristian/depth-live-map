"""Engine 3 — SCDepthV3 monocular depth on Hailo-8 NPU. [TRACK B / implemented by Track A]"""

import logging

import cv2
import numpy as np

from data_contract import FrameResult
from src.engines.base_engine import BaseEngine
from src.hailo_inference.hef_loader import HEFModel
from src.hailo_inference.stream_utils import to_nhwc_batch

logger = logging.getLogger(__name__)

# scdepthv3 fixed input size: H=256, W=320
_IN_H, _IN_W = 256, 320


class DepthEngine(BaseEngine):
    """SCDepthV3 monocular depth estimator on Hailo-8 NPU.

    Input:  (256, 320, 3) RGB UINT8
    Output: (256, 320, 1) float32 log-disparity — higher = closer.

    Reads:  frame, bbox, class_id
    Writes: rel_depth_score, depth_variance
            depth_map (extra attribute for visualization, not in data contract)

    Args:
        hef_path: Path to scdepthv3.hef.
        device: Shared VDevice, or None to create a new one.
    """

    def __init__(self, hef_path: str, device=None, use_scheduler: bool = False) -> None:
        self._model = HEFModel(hef_path, device=device, use_scheduler=use_scheduler)
        self._pipeline_ctx = None
        self._pipeline = None
        self.depth_map: np.ndarray | None = None  # latest (H,W) for display
        logger.info("DepthEngine ready | model=%s | input=(%d,%d)", hef_path, _IN_H, _IN_W)

    def start(self) -> None:
        self._pipeline_ctx = self._model.session()
        self._pipeline = self._pipeline_ctx.__enter__()

    def stop(self) -> None:
        if self._pipeline_ctx is not None:
            self._pipeline_ctx.__exit__(None, None, None)
            self._pipeline = None

    def process(self, result: FrameResult) -> FrameResult:
        """Run depth inference on result.frame.

        Writes rel_depth_score and depth_variance for the detected bbox region.
        If no valid detection (class_id < 0), scores are NaN.
        """
        try:
            frame_bgr = result.frame
            orig_h, orig_w = frame_bgr.shape[:2]

            resized = cv2.resize(frame_bgr, (_IN_W, _IN_H))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            batch = to_nhwc_batch(rgb)

            raw = self._model.infer(self._pipeline, batch)
            # raw shape: (1, 256, 320, 1) float32
            depth = raw[0, :, :, 0]   # (H, W) log-disparity, higher = closer
            self.depth_map = depth

            if result.class_id < 0 or result.bbox == (0, 0, 0, 0):
                result.rel_depth_score = float("nan")
                result.depth_variance = float("nan")
                return result

            # Project bbox from original frame coords to depth map coords
            x1, y1, x2, y2 = result.bbox
            dx1 = int(np.clip(x1 / orig_w * _IN_W, 0, _IN_W - 1))
            dy1 = int(np.clip(y1 / orig_h * _IN_H, 0, _IN_H - 1))
            dx2 = int(np.clip(x2 / orig_w * _IN_W, 1, _IN_W))
            dy2 = int(np.clip(y2 / orig_h * _IN_H, 1, _IN_H))

            roi = depth[dy1:dy2, dx1:dx2]
            if roi.size == 0:
                result.rel_depth_score = float("nan")
                result.depth_variance = float("nan")
                return result

            result.rel_depth_score = float(np.median(roi))
            result.depth_variance = float(np.var(roi))

        except Exception as exc:
            logger.warning("DepthEngine failed: %s", exc)
            result.rel_depth_score = float("nan")
            result.depth_variance = float("nan")
            self.depth_map = None

        return result
