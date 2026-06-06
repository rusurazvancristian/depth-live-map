import os
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

os.environ.setdefault("HF_HOME", "D:/hf_cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "D:/hf_cache")


class DepthEngine:
    """GPU-accelerated metric depth estimation via DepthAnything V2."""

    def __init__(self, model_id: str, fallback_id: str, device: str = "cuda",
                 metric_scale: float = 1.0):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.is_metric = False
        # Global metric correction applied to every depth map at the source, so
        # point cloud / BEV / object distances are all consistently scaled.
        # Anchored on the refrigerator (1.6 m known height) — the indoor model
        # overestimates real depth by ~1.8x, so the correction is ~0.55.
        self.metric_scale = metric_scale

        try:
            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
            self.is_metric = "Metric" in model_id
            print(f"[depth] Loaded {model_id} on {self.device}")
        except Exception as e:
            print(f"[depth] {model_id} failed ({e}), falling back to {fallback_id}")
            self.processor = AutoImageProcessor.from_pretrained(fallback_id)
            self.model = AutoModelForDepthEstimation.from_pretrained(fallback_id)

        self.model = self.model.to(self.device).eval()

    @torch.inference_mode()
    def infer(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Return depth map (H×W float32, meters) matching input spatial size."""
        h, w = bgr_frame.shape[:2]
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        # predicted_depth: [1, H', W'] or [H', W']
        depth = outputs.predicted_depth.squeeze().cpu().float().numpy()

        depth_resized = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        if not self.is_metric:
            # Relative disparity → pseudo-metric via calibration to target median
            depth_resized = self._relative_to_metric(depth_resized)

        depth_resized = depth_resized * self.metric_scale
        return depth_resized.astype(np.float32)

    @staticmethod
    def _relative_to_metric(
        disp: np.ndarray,
        target_median_m: float = 4.0,
        d_min: float = 0.3,
        d_max: float = 15.0,
    ) -> np.ndarray:
        """Map disparity-like output (larger=closer) to pseudo-metric depth."""
        eps = 1e-6
        disp = np.clip(disp, np.percentile(disp, 2), np.percentile(disp, 98))
        disp_norm = (disp - disp.min()) / (disp.max() - disp.min() + eps)
        # Invert: closer pixels have higher disp → lower depth
        depth = 1.0 / (disp_norm + eps)
        scale = target_median_m / (np.median(depth) + eps)
        return np.clip(depth * scale, d_min, d_max)
