"""Engine 1 — YOLO object detection on Hailo-8 NPU. [TRACK A]"""

import logging

import cv2
import numpy as np

from data_contract import FrameResult
from src.engines.base_engine import BaseEngine
from src.hailo_inference.hef_loader import HEFModel
from src.hailo_inference.stream_utils import to_nhwc_batch

logger = logging.getLogger(__name__)

COCO_CLASSES = [
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
    """YOLOv8 object detector running on Hailo-8 NPU.

    Output format from yolov8*_h8.hef NMS postprocess:
        result[out_name][batch_idx] = list of 80 arrays, one per class.
        Each array shape: (N, 5) with [x1_norm, y1_norm, x2_norm, y2_norm, score].

    Reads:  frame
    Writes: bbox, bbox_height_px, class_id, class_name, det_confidence

    Args:
        hef_path: Path to compiled yolov8*_h8.hef.
        conf_threshold: Minimum detection confidence [0, 1].
        device: Shared VDevice, or None to create a new one.
    """

    def __init__(
        self,
        hef_path: str,
        conf_threshold: float = 0.5,
        device=None,
    ) -> None:
        self._model = HEFModel(hef_path, device=device)
        self._conf_thr = conf_threshold
        self._input_size = self._model.input_shape[0]  # square side, e.g. 640
        self._pipeline = None
        self._active = None
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

            # Simple resize (no letterbox) — model outputs coords normalised to
            # the resized 640x640 space; direct x_n*orig_w mapping is correct.
            resized = cv2.resize(frame_bgr, (self._input_size, self._input_size))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            batch = to_nhwc_batch(rgb)

            raw = self._model.infer(self._pipeline, batch)
            # raw[0] = list[80] of ndarray (N, 5): [x1n, y1n, x2n, y2n, score]

            best_score = -1.0
            best = None

            for class_id, dets in enumerate(raw[0]):
                if dets is None or len(dets) == 0:
                    continue
                dets = np.asarray(dets)
                for det in dets:
                    score = float(det[4])
                    if score >= self._conf_thr and score > best_score:
                        best_score = score
                        best = (class_id, det)

            if best is None:
                result.class_id = -1
                result.class_name = ""
                result.det_confidence = 0.0
                result.bbox = (0, 0, 0, 0)
                result.bbox_height_px = 0.0
                return result

            class_id, det = best
            x1 = int(np.clip(det[0] * orig_w, 0, orig_w - 1))
            y1 = int(np.clip(det[1] * orig_h, 0, orig_h - 1))
            x2 = int(np.clip(det[2] * orig_w, 0, orig_w - 1))
            y2 = int(np.clip(det[3] * orig_h, 0, orig_h - 1))

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
