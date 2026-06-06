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

# Raw model output scales: (stride, box_tensor_suffix, cls_tensor_suffix)
_RAW_SCALES = [
    (8,  "conv71", "conv74"),
    (16, "conv87", "conv90"),
    (32, "conv101", "conv104"),
]

_ALPHA = 0.35     # EMA blend: higher = more responsive, lower = smoother
_MAX_LOST = 15    # frames before track is dropped (~0.5s @ 30fps)
_IOU_LOCK_THR = 0.25  # minimum IoU to continue tracking same object


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class YOLOEngine(BaseEngine):
    """YOLOv8 object detector running on Hailo-8 NPU.

    Supports two output formats auto-detected at init:
    - NMS format (yolov8*_h8.hef): single output, list of 80 class arrays.
    - Raw format (yolo26m.hef): 6 raw tensors, decoded with sigmoid + grid.

    Reads:  frame
    Writes: bbox, bbox_height_px, class_id, class_name, det_confidence

    Args:
        hef_path: Path to compiled .hef model.
        conf_threshold: Minimum detection confidence [0, 1].
        device: Shared VDevice, or None to create a new one.
    """

    def __init__(
        self,
        hef_path: str,
        conf_threshold: float = 0.5,
        device=None,
        use_scheduler: bool = False,
    ) -> None:
        self._model = HEFModel(hef_path, device=device, use_scheduler=use_scheduler)
        self._conf_thr = conf_threshold
        self._input_size = self._model.input_shape[0]  # square side, e.g. 640
        self._pipeline_ctx = None
        self._pipeline = None

        # Auto-detect format: raw if 6 output tensors, NMS if 1
        self._raw_mode = len(self._model.output_names) > 1
        # Resolve tensor name prefix for raw models (e.g. "yolo26m/")
        if self._raw_mode:
            prefix = self._model.output_names[0].rsplit("/", 1)[0] + "/"
            self._scale_keys = [
                (stride, f"{prefix}{b}", f"{prefix}{c}")
                for stride, b, c in _RAW_SCALES
            ]

        # Tracker state: EMA-smoothed bbox + persistence over lost frames
        self._track_bbox_f: np.ndarray | None = None  # float [x1,y1,x2,y2]
        self._track_class_id: int = -1
        self._track_class_name: str = ""
        self._track_score: float = 0.0
        self._lost_frames: int = 0

        logger.info(
            "YOLOEngine ready | model=%s | mode=%s | conf_thr=%.2f | input=%d",
            hef_path, "raw" if self._raw_mode else "nms", conf_threshold, self._input_size,
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

    def reset_tracker(self) -> None:
        """Drop the current track — next detection starts fresh."""
        self._track_bbox_f = None
        self._track_class_id = -1
        self._track_class_name = ""
        self._track_score = 0.0
        self._lost_frames = 0
        logger.info("Tracker reset.")

    def _decode_raw(self, outputs: dict, orig_h: int, orig_w: int) -> list:
        """Decode raw multi-scale outputs into all detections above threshold.

        Returns list of (class_id, score, x1, y1, x2, y2).
        """
        dets = []
        for stride, box_key, cls_key in self._scale_keys:
            boxes = outputs[box_key][0]            # (H, W, 4) ltrb in stride units
            probs = _sigmoid(outputs[cls_key][0])  # (H, W, 80)

            scores = probs.max(axis=2)             # (H, W)
            cls_ids = probs.argmax(axis=2)         # (H, W)

            ys, xs = np.where(scores > self._conf_thr)
            for y, x in zip(ys, xs):
                cx = (x + 0.5) * stride
                cy = (y + 0.5) * stride
                l, t, r, b = boxes[y, x] * stride
                x1 = int(np.clip((cx - l) / 640 * orig_w, 0, orig_w - 1))
                y1 = int(np.clip((cy - t) / 640 * orig_h, 0, orig_h - 1))
                x2 = int(np.clip((cx + r) / 640 * orig_w, 0, orig_w - 1))
                y2 = int(np.clip((cy + b) / 640 * orig_h, 0, orig_h - 1))
                if x2 > x1 and y2 > y1:
                    dets.append((int(cls_ids[y, x]), float(scores[y, x]), x1, y1, x2, y2))
        return dets

    def _pick_detection(self, dets: list):
        """Pick best detection from list using IoU lock when tracking.

        When a track is active: prefer the detection with the highest IoU
        against the current bbox — ignoring class and score.
        When no active track: pick the highest-confidence detection.
        Returns (class_id, score, x1, y1, x2, y2) or None.
        """
        if not dets:
            return None

        if self._track_bbox_f is None:
            return max(dets, key=lambda d: d[1])

        best_iou = -1.0
        best = None
        for det in dets:
            _, _, x1, y1, x2, y2 = det
            iou = _iou(self._track_bbox_f, np.array([x1, y1, x2, y2], dtype=float))
            if iou > best_iou:
                best_iou = iou
                best = det

        # Only accept if overlaps enough; otherwise treat as lost
        return best if best_iou >= _IOU_LOCK_THR else None

    def _update_tracker(self, detection) -> None:
        """Update EMA tracker with a new detection (or None if lost)."""
        if detection is None:
            self._lost_frames += 1
            return

        class_id, score, x1, y1, x2, y2 = detection
        new_bbox = np.array([x1, y1, x2, y2], dtype=float)

        if self._track_bbox_f is None:
            self._track_bbox_f = new_bbox
        else:
            self._track_bbox_f = _ALPHA * new_bbox + (1.0 - _ALPHA) * self._track_bbox_f

        self._track_class_id = class_id
        self._track_class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else str(class_id)
        self._track_score = score
        self._lost_frames = 0

    def process(self, result: FrameResult) -> FrameResult:
        """Run YOLO detection on result.frame, with EMA tracking.

        Args:
            result: FrameResult with frame populated (BGR, any resolution).

        Returns:
            FrameResult with bbox, class_id, class_name, det_confidence,
            bbox_height_px populated. If no detection: class_id = -1.
        """
        try:
            frame_bgr = result.frame
            orig_h, orig_w = frame_bgr.shape[:2]

            resized = cv2.resize(frame_bgr, (self._input_size, self._input_size))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            batch = to_nhwc_batch(rgb)

            if self._raw_mode:
                outputs = self._model.infer_all(self._pipeline, batch)
                all_dets = self._decode_raw(outputs, orig_h, orig_w)
            else:
                # NMS postprocess format: raw[0] = list[80] of (N,5) arrays
                raw = self._model.infer(self._pipeline, batch)
                all_dets = []
                for class_id, class_dets in enumerate(raw[0]):
                    if class_dets is None or len(class_dets) == 0:
                        continue
                    for det in np.asarray(class_dets):
                        score = float(det[4])
                        if score >= self._conf_thr:
                            # Hailo NMS output order: [y1n, x1n, y2n, x2n, score]
                            y1 = int(np.clip(det[0] * orig_h, 0, orig_h - 1))
                            x1 = int(np.clip(det[1] * orig_w, 0, orig_w - 1))
                            y2 = int(np.clip(det[2] * orig_h, 0, orig_h - 1))
                            x2 = int(np.clip(det[3] * orig_w, 0, orig_w - 1))
                            all_dets.append((class_id, score, x1, y1, x2, y2))

            self._update_tracker(self._pick_detection(all_dets))

            if self._track_bbox_f is None or self._lost_frames >= _MAX_LOST:
                result.class_id = -1
                result.class_name = ""
                result.det_confidence = 0.0
                result.bbox = (0, 0, 0, 0)
                result.bbox_height_px = 0.0
                return result

            x1, y1, x2, y2 = (int(v) for v in self._track_bbox_f)
            result.class_id = self._track_class_id
            result.class_name = self._track_class_name
            result.det_confidence = self._track_score
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
