"""
Object detection + distance measurement with metric calibration.

Two distance estimators, fused:
  1. Pinhole height model  (for objects of known real height, fully in-frame)
       distance = fy * H_real / h_px
     This is metrically exact given the intrinsics — no depth-net scale drift.
  2. Depth-net 15th-percentile  (fallback for unknown / cropped objects)
       15th pct of inner-50% bbox depth — foreground-biased.

Calibration anchor:
  Known-height detections (e.g. refrigerator = 1.60 m, person = 1.70 m) yield a
  ground-truth distance. The ratio  pinhole / depth-net  gives a global scale
  correction that is applied to every depth-net-only measurement, locking the
  whole scene to real metric scale.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import cv2


# Real-world heights (metres) for COCO classes with reasonably standard size.
# The refrigerator anchor (1.60 m) was provided as ground truth for this dataset.
KNOWN_HEIGHTS: dict[str, float] = {
    "refrigerator":  1.60,
    "person":        1.70,
    "chair":         0.90,
    "couch":         0.80,
    "bench":         0.85,
    "dining table":  0.75,
    "tv":            0.60,
    "laptop":        0.22,
    "microwave":     0.30,
    "oven":          0.90,
    "toaster":       0.20,
    "sink":          0.85,
    "toilet":        0.70,
    "bed":           0.60,
    "potted plant":  0.45,
    "bottle":        0.25,
    "vase":          0.30,
    "backpack":      0.45,
    "suitcase":      0.55,
    "bicycle":       1.05,
    "motorcycle":    1.10,
    "car":           1.50,
    "fire hydrant":  0.75,
    "parking meter": 1.20,
    "stop sign":     2.00,
    "traffic light": 3.00,
}

# Subset of KNOWN_HEIGHTS reliable enough to anchor the GLOBAL depth scale:
# large, rigid objects whose real height is well-standardized. Small / highly
# variable objects (bottle, backpack, laptop, vase) still get a pinhole distance
# for display but are excluded from scale calibration to avoid noise.
ANCHOR_CLASSES: set[str] = {
    "refrigerator", "person", "chair", "couch", "dining table",
    "oven", "sink", "toilet", "bed", "car", "bicycle", "motorcycle",
}


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]    # x1, y1, x2, y2 in original frame pixels
    distance_m: Optional[float] = None       # final fused distance
    distance_depth: Optional[float] = None   # raw depth-net estimate
    distance_pinhole: Optional[float] = None # known-height geometric estimate
    method: str = "none"                     # "pinhole" | "depth" | "depth*scale"
    world_pos:  Optional[np.ndarray] = None  # (3,) XYZ in world frame


class ObjectDetector:
    """YOLOv8n on GPU → bbox + class + foreground-biased depth distance."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf_thresh: float = 0.35,
        device: str = "cuda",
    ):
        from ultralytics import YOLO
        self.model  = YOLO(model_name)
        self.conf   = conf_thresh
        self.device = device
        # Running depth-net scale correction, anchored by known-height detections.
        self.depth_scale = 1.0
        self._scale_samples: list[float] = []
        print(f"[detector] {model_name} ready on {device}  ({len(self.model.names)} classes)")

    def detect(self, bgr: np.ndarray) -> list[Detection]:
        results = self.model(bgr, conf=self.conf, device=self.device,
                             verbose=False, imgsz=640)
        dets: list[Detection] = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0])
                dets.append(Detection(
                    class_name=self.model.names[cls_id],
                    confidence=float(box.conf[0]),
                    bbox_xyxy=(x1, y1, x2, y2),
                ))
        return dets

    def measure_distances(
        self,
        dets: list[Detection],
        depth_map: np.ndarray,
        camera=None,
        d_min: float = 0.3,
        d_max: float = 15.0,
    ) -> None:
        """
        Compute a calibrated metric distance for every detection.

        Pass 1: depth-net distance for all + pinhole distance for known, in-frame
                objects; collect pinhole/depth ratios as scale samples.
        Pass 2: fuse — prefer pinhole; otherwise depth-net * global scale.
        """
        H, W = depth_map.shape
        border = 4   # px tolerance for "touching frame edge"

        # ── Pass 1: raw estimates ──────────────────────────────────────────
        for det in dets:
            x1, y1, x2, y2 = det.bbox_xyxy
            bw, bh = x2 - x1, y2 - y1

            # Depth-net: 15th pct of inner-50% patch
            ix1, ix2 = max(0, x1 + bw // 4), min(W, x2 - bw // 4)
            iy1, iy2 = max(0, y1 + bh // 4), min(H, y2 - bh // 4)
            if ix2 > ix1 and iy2 > iy1:
                patch = depth_map[iy1:iy2, ix1:ix2]
                valid = patch[(patch > d_min) & (patch < d_max)]
                if len(valid) >= 5:
                    det.distance_depth = float(np.percentile(valid, 15))

            # Pinhole: known height + fully visible (not cropped top/bottom)
            h_real = KNOWN_HEIGHTS.get(det.class_name)
            fully_in_frame = (y1 > border) and (y2 < H - border)
            if h_real is not None and fully_in_frame and bh > 10 and camera is not None:
                d_pin = camera.fy * h_real / bh
                if d_min < d_pin < d_max * 2:
                    det.distance_pinhole = float(d_pin)
                    # Only reliable anchor classes calibrate the global scale
                    if (det.class_name in ANCHOR_CLASSES
                            and det.distance_depth and det.distance_depth > 0.1):
                        self._scale_samples.append(d_pin / det.distance_depth)

        # ── Update global scale (running median, clamped to sane range) ─────
        if self._scale_samples:
            self._scale_samples = self._scale_samples[-200:]
            self.depth_scale = float(
                np.clip(np.median(self._scale_samples), 0.3, 3.0)
            )

        # ── Pass 2: fuse ───────────────────────────────────────────────────
        for det in dets:
            if det.distance_pinhole is not None:
                det.distance_m = det.distance_pinhole
                det.method = "pinhole"
            elif det.distance_depth is not None:
                det.distance_m = det.distance_depth * self.depth_scale
                det.method = "depth*scale" if abs(self.depth_scale - 1.0) > 0.02 else "depth"

    def project_to_world(
        self,
        dets: list[Detection],
        pose_c2w: np.ndarray,
        camera,
    ) -> None:
        """Lift each bbox centre to 3D world coords using its measured depth."""
        for det in dets:
            if det.distance_m is None:
                continue
            x1, y1, x2, y2 = det.bbox_xyxy
            u = (x1 + x2) * 0.5
            v = (y1 + y2) * 0.5
            z = det.distance_m
            p_world = pose_c2w @ np.array([
                (u - camera.cx) * z / camera.fx,
                (v - camera.cy) * z / camera.fy,
                z, 1.0,
            ])
            det.world_pos = p_world[:3].copy()

    @staticmethod
    def class_color(class_name: str) -> tuple[int, int, int]:
        """Deterministic, saturated BGR colour per class name."""
        h = (hash(class_name) * 2654435761) % 180   # OpenCV uint8 HSV: H in [0, 179]
        hsv = np.uint8([[[int(h), 210, 215]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        return int(bgr[0]), int(bgr[1]), int(bgr[2])


class ObjectMemory:
    """
    Persistent semantic map: detected objects accumulate across frames.
    Re-detections within merge_dist metres update the stored position
    (exponential moving average) rather than adding a duplicate.
    """

    def __init__(self, merge_dist: float = 0.8):
        self._objects: list[dict] = []
        self.merge_dist = merge_dist

    def update(self, dets: list[Detection], frame_id: int) -> None:
        for det in dets:
            if det.world_pos is None:
                continue
            pos = det.world_pos[:3]
            for obj in self._objects:
                if (obj["name"] == det.class_name
                        and np.linalg.norm(obj["pos"] - pos) < self.merge_dist):
                    obj["pos"]   = 0.7 * obj["pos"] + 0.3 * pos   # smooth update
                    obj["frame"] = frame_id
                    obj["dist"]  = det.distance_m
                    break
            else:
                self._objects.append({
                    "name":  det.class_name,
                    "pos":   pos.copy(),
                    "frame": frame_id,
                    "dist":  det.distance_m,
                })

    @property
    def objects(self) -> list[dict]:
        return self._objects
