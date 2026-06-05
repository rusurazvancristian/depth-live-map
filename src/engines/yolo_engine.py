"""Engine 1 — YOLO object detection on Hailo-8 NPU. [TRACK A]"""

import logging
from typing import Any, Tuple, List, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from hailo_platform import VDevice

from data_contract import FrameResult
from src.engines.base_engine import BaseEngine
from src.hailo_inference.hef_loader import HEFModel
from src.hailo_inference.stream_utils import letterbox_resize, unletterbox_bbox

logger = logging.getLogger(__name__)

COCO_CLASSES: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


class YOLOEngine(BaseEngine):
    """YOLOv8/YOLO26 object detector running on Hailo-8 NPU.

    Supports both traditional Hailo NMS output and NMS-free output shapes.

    Reads:  frame
    Writes: bbox, bbox_height_px, class_id, class_name, det_confidence
    """

    _model: HEFModel
    _conf_thr: float
    _input_size: int
    _pipeline: Any
    _pipeline_ctx: Any

    # Pre-allocated destination buffers for zero-allocation reuse
    _dst_bgr: Optional[np.ndarray]
    _batch_rgb: Optional[np.ndarray]
    _orig_h: Optional[int]
    _orig_w: Optional[int]

    def __init__(
        self,
        hef_path: str,
        conf_threshold: float = 0.5,
        device: Optional['VDevice'] = None,
    ) -> None:
        self._model = HEFModel(hef_path, device=device)
        self._conf_thr = conf_threshold
        self._input_size = self._model.input_shape[0]  # square side, e.g. 640
        self._pipeline = None
        self._pipeline_ctx = None
        
        # Initialize buffer tracking
        self._dst_bgr = None
        self._batch_rgb = None
        self._orig_h = None
        self._orig_w = None

        logger.info(
            "YOLOEngine ready | model=%s | conf_thr=%.2f | input=%d",
            hef_path, conf_threshold, self._input_size,
        )

    def start(self) -> None:
        """Open inference session. Call once before the frame loop."""
        self._pipeline_ctx = self._model.session()
        self._pipeline = self._pipeline_ctx.__enter__()

    def stop(self) -> None:
        """Close inference session."""
        if self._pipeline_ctx is not None:
            self._pipeline_ctx.__exit__(None, None, None)
            self._pipeline = None

    def process(self, result: FrameResult) -> FrameResult:
        """Run YOLO detection on result.frame.

        Args:
            result: FrameResult with frame populated (BGR, any resolution).

        Returns:
            FrameResult with bbox, class_id, class_name, det_confidence,
            bbox_height_px populated. If no detection: class_id = -1.
        """
        try:
            frame_bgr = result.frame
            orig_h, orig_w = frame_bgr.shape[:2]

            # Lazy allocate pre-allocated buffers on resolution change
            if self._batch_rgb is None or self._orig_h != orig_h or self._orig_w != orig_w:
                self._orig_h = orig_h
                self._orig_w = orig_w
                scale = self._input_size / max(orig_h, orig_w)
                new_w, new_h = int(orig_w * scale), int(orig_h * scale)
                self._dst_bgr = np.empty((new_h, new_w, 3), dtype=np.uint8)
                self._batch_rgb = np.full((1, self._input_size, self._input_size, 3), 114, dtype=np.uint8)

            # Preprocess using pre-allocated buffers
            rgb, scale, pad = letterbox_resize(
                frame_bgr,
                self._input_size,
                dst_bgr=self._dst_bgr,
                dst_rgb=self._batch_rgb[0]
            )
            batch = self._batch_rgb

            raw = self._model.infer(self._pipeline, batch)
            # raw[0] can be either a list (traditional NMS) or an array (NMS-free YOLO26/v10)

            best_score: float = -1.0
            best: Optional[Tuple[int, np.ndarray]] = None

            if isinstance(raw[0], np.ndarray):
                # NMS-free YOLO26 / YOLOv10 format: typically shape (1, N, 6) or (N, 6)
                arr = raw[0]
                if arr.ndim == 3:
                    arr = arr[0]  # Shape: (N, 6)
                # Auto-transpose if shape is (6, N)
                if arr.shape[0] == 6 and arr.shape[1] != 6:
                    arr = arr.T

                for det in arr:
                    if len(det) < 6:
                        continue
                    score = float(det[4])
                    if score >= self._conf_thr and score > best_score:
                        class_id = int(det[5])
                        best_score = score
                        best = (class_id, det[:4])
            else:
                # Traditional Hailo NMS format: list of 80 arrays
                for class_id, dets in enumerate(raw[0]):
                    if dets is None or len(dets) == 0:
                        continue
                    dets = np.asarray(dets)
                    for det in dets:
                        score = float(det[4])
                        if score >= self._conf_thr and score > best_score:
                            best_score = score
                            best = (class_id, det[:4])

            if best is None:
                result.class_id = -1
                result.class_name = ""
                result.det_confidence = 0.0
                result.bbox = (0, 0, 0, 0)
                result.bbox_height_px = 0.0
                return result

            class_id, det = best
            # Hailo NMS output coordinates are in y1, x1, y2, x2 (yxyx) format
            y1n, x1n, y2n, x2n = det[0], det[1], det[2], det[3]
            x1, y1, x2, y2 = unletterbox_bbox(
                x1n, y1n, x2n, y2n,
                scale, pad, orig_w, orig_h, self._input_size,
            )

            result.class_id = class_id
            result.class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else str(class_id)
            result.det_confidence = best_score
            result.bbox = (x1, y1, x2, y2)
            result.bbox_height_px = float(y2 - y1)

        except Exception as exc:
            logger.warning("YOLOEngine failed: %s", exc)
            result.class_id = -1
            result.class_name = ""
            result.det_confidence = 0.0
            result.bbox = (0, 0, 0, 0)
            result.bbox_height_px = 0.0

        return result
