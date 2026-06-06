"""Engine 1 — YOLO26s multi-object detection on Hailo-8 NPU.

Returns ALL detections above confidence threshold (not just best),
compatible with downstream ByteTrack multi-object tracker.
"""

import logging
from typing import List, Optional, TYPE_CHECKING

import cv2
import numpy as np

from data_contract import Detection, FrameResult
from src.engines.base_engine import BaseEngine
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
    """YOLO26s object detector running on Hailo-8 NPU via HailoMultiplexer.

    Returns ALL detections above confidence threshold for ByteTrack consumption.

    Reads:  frame
    Writes: detections (list of Detection objects)
    """

    def __init__(
        self,
        multiplexer,
        model_name: str = "yolo",
        conf_threshold: float = 0.5,
    ) -> None:
        """Initialise YOLO engine.

        Args:
            multiplexer: HailoMultiplexer instance managing the VDevice.
            model_name: Key used when loading the YOLO model in the multiplexer.
            conf_threshold: Minimum detection confidence.
        """
        self._mux = multiplexer
        self._model_name = model_name
        self._conf_thr = conf_threshold

        input_shape = self._mux.get_input_shape(model_name)
        # Input shape is (batch, H, W, C) or (H, W, C)
        if len(input_shape) == 4:
            self._input_size = input_shape[1]  # square side, e.g. 640
        else:
            self._input_size = input_shape[0]

        # Pre-allocated destination buffers for zero-allocation reuse
        self._dst_bgr: Optional[np.ndarray] = None
        self._batch_rgb: Optional[np.ndarray] = None
        self._orig_h: Optional[int] = None
        self._orig_w: Optional[int] = None

        logger.info(
            "YOLOEngine ready | model=%s | conf_thr=%.2f | input=%d",
            model_name, conf_threshold, self._input_size,
        )

    def process(self, result: FrameResult) -> FrameResult:
        """Run YOLO detection on result.frame, returning ALL detections.

        Args:
            result: FrameResult with frame populated (BGR, any resolution).

        Returns:
            FrameResult with detections list populated. Empty list if no detections.
        """
        result.detections = []

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
                self._batch_rgb = np.full(
                    (1, self._input_size, self._input_size, 3), 114, dtype=np.uint8
                )

            # Preprocess using pre-allocated buffers
            rgb, scale, pad = letterbox_resize(
                frame_bgr,
                self._input_size,
                dst_bgr=self._dst_bgr,
                dst_rgb=self._batch_rgb[0],
            )
            batch = self._batch_rgb

            # Run NPU inference
            raw = self._mux.infer(self._model_name, batch)

            # Parse detections from raw output
            detections = self._parse_detections(raw, scale, pad, orig_w, orig_h)
            result.detections = detections

        except Exception as exc:
            logger.warning("YOLOEngine failed: %s", exc, exc_info=True)
            result.detections = []

        return result

    def _parse_detections(
        self,
        raw,
        scale: float,
        pad: tuple,
        orig_w: int,
        orig_h: int,
    ) -> List[Detection]:
        """Parse raw YOLO output into Detection objects.

        Supports both NMS-free (YOLO26/v10) and traditional Hailo NMS formats.

        Args:
            raw: Raw inference output from multiplexer.
            scale: Letterbox scale factor.
            pad: (pad_w, pad_h) from letterbox.
            orig_w: Original frame width.
            orig_h: Original frame height.

        Returns:
            List of Detection objects above confidence threshold.
        """
        detections: List[Detection] = []

        if isinstance(raw, np.ndarray):
            # NMS-free YOLO26 / YOLOv10 format
            arr = raw
            if arr.ndim == 3:
                arr = arr[0]  # (1, N, 6) -> (N, 6)
            # Auto-transpose if shape is (6, N)
            if arr.ndim == 2 and arr.shape[0] == 6 and arr.shape[1] != 6:
                arr = arr.T

            for det in arr:
                if len(det) < 6:
                    continue
                score = float(det[4])
                if score < self._conf_thr:
                    continue

                class_id = int(det[5])
                # YOLO26 NMS-free: coordinates are in xyxy format
                y1n, x1n, y2n, x2n = det[0], det[1], det[2], det[3]
                x1, y1, x2, y2 = unletterbox_bbox(
                    x1n, y1n, x2n, y2n,
                    scale, pad, orig_w, orig_h, self._input_size,
                )

                class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else str(class_id)
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=score,
                    class_id=class_id,
                    class_name=class_name,
                ))

        elif isinstance(raw, list):
            # Check if it's a list of arrays (traditional NMS) or a nested structure
            first = raw[0] if len(raw) > 0 else None
            if isinstance(first, list):
                # Traditional Hailo NMS: list[batch] -> list[class] -> np.ndarray
                class_arrays = first
                for class_id, dets in enumerate(class_arrays):
                    if dets is None or len(dets) == 0:
                        continue
                    dets = np.asarray(dets, dtype=np.float32)
                    for det in dets:
                        score = float(det[4])
                        if score < self._conf_thr:
                            continue

                        y1n, x1n, y2n, x2n = det[0], det[1], det[2], det[3]
                        x1, y1, x2, y2 = unletterbox_bbox(
                            x1n, y1n, x2n, y2n,
                            scale, pad, orig_w, orig_h, self._input_size,
                        )

                        class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else str(class_id)
                        detections.append(Detection(
                            bbox=(x1, y1, x2, y2),
                            confidence=score,
                            class_id=class_id,
                            class_name=class_name,
                        ))

        return detections
