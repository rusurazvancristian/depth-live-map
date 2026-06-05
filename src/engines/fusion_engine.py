import numpy as np
import logging
import os
import json
import math
from typing import Dict, Any

from src.engines.base_engine import BaseEngine
from data_contract import FrameResult

logger = logging.getLogger(__name__)

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("onnxruntime not found. FusionEngine will run in dry-run/fallback mode.")


class FusionEngine(BaseEngine):
    """ONNX-based distance fusion Multi-Layer Perceptron (MLP) running on Pi CPU.
    
    Fuses geometric distance, relative depth score, and class ID to output 
    calibrated metric distance and confidence intervals.
    
    Reads: d_geometric_m, rel_depth_score, class_id
    Writes: final_distance_m, log_variance, confidence_68, confidence_95
    """

    def __init__(self, onnx_path: str, norm_path: str = None) -> None:
        """Initializes the FusionEngine.
        
        Args:
            onnx_path: Path to the exported fusion_mlp.onnx model.
            norm_path: Optional path to the normalization parameters JSON/Pt file.
                              If None, will look for a JSON file next to the ONNX model.
        """
        self.onnx_path = onnx_path
        self._session = None
        
        # Load normalization parameters (default values if file is missing/invalid)
        self.mean = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.std = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        
        # Find and load normalization parameters
        if norm_path is None:
            norm_path = os.path.splitext(onnx_path)[0] + "_norm.json"
            
        self._load_norm_params(norm_path)

        if ONNX_AVAILABLE:
            try:
                self._session = ort.InferenceSession(
                    self.onnx_path,
                    providers=["CPUExecutionProvider"]
                )
                logger.info(f"Loaded Fusion MLP ONNX model from {self.onnx_path}")
            except Exception as e:
                logger.error(f"Failed to load ONNX session: {e}")
                self._session = None

    def _load_norm_params(self, path: str) -> None:
        """Loads normalization mean and standard deviation from a JSON or PyTorch Pt file."""
        if path.endswith(".pt"):
            try:
                import torch
                if os.path.exists(path):
                    data = torch.load(path, map_location="cpu")
                    if isinstance(data, dict) and "mean" in data and "std" in data:
                        # Extract mean
                        m = data["mean"]
                        self.mean = m.cpu().numpy() if hasattr(m, "cpu") else np.array(m, dtype=np.float32)
                        # Extract std
                        s = data["std"]
                        self.std = s.cpu().numpy() if hasattr(s, "cpu") else np.array(s, dtype=np.float32)
                        logger.info(f"Loaded normalization parameters from PyTorch file: {path}")
                        return
            except Exception as e:
                logger.warning(f"Failed to load PyTorch norm file {path}: {e}")
            
            # Fall back to JSON paths if PyTorch failed or torch is not installed
            json_path = os.path.splitext(path)[0] + ".json"
            if os.path.exists(json_path):
                path = json_path
            else:
                json_path_alt = os.path.splitext(path)[0] + "_norm.json"
                if os.path.exists(json_path_alt):
                    path = json_path_alt

        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                self.mean = np.array(data["mean"], dtype=np.float32)
                self.std = np.array(data["std"], dtype=np.float32)
                logger.info(f"Loaded normalization parameters from JSON file: {path}")
            except Exception as e:
                logger.warning(f"Error reading normalization config {path}, using defaults: {e}")
        else:
            logger.warning(f"Normalization config {path} not found. Using defaults.")

    def process(self, result: FrameResult) -> FrameResult:
        """Fuses input features to compute final metric distance and uncertainty.
        
        Args:
            result: The current FrameResult object.
            
        Returns:
            The modified FrameResult object.
        """
        # Pre-checks: If geometric estimate and depth estimate are both NaN, degrade.
        # However, the MLP can handle individual NaNs if trained with sentinels (-1.0).
        d_geom = result.d_geometric_m
        rel_depth = result.rel_depth_score
        
        # Replace NaNs with sentinel values (-1.0)
        feat_geom = -1.0 if math.isnan(d_geom) else float(d_geom)
        feat_depth = -1.0 if math.isnan(rel_depth) else float(rel_depth)
        feat_class = -1.0 if result.class_id < 0 else float(result.class_id)

        # If we have no valid inputs at all, degrade to NaN outputs
        if feat_geom == -1.0 and feat_depth == -1.0:
            result.final_distance_m = float("nan")
            result.log_variance = float("nan")
            result.confidence_68 = (0.0, 0.0)
            result.confidence_95 = (0.0, 0.0)
            return result

        try:
            # 1. Normalize features
            raw_features = np.array([feat_geom, feat_depth, feat_class], dtype=np.float32)
            norm_features = (raw_features - self.mean) / (self.std + 1e-8)
            input_batch = np.expand_dims(norm_features, axis=0)  # Shape: (1, 3)

            # 2. Run ONNX Inference
            if ONNX_AVAILABLE and self._session is not None:
                # Get model inputs/outputs
                input_name = self._session.get_inputs()[0].name
                pred = self._session.run(None, {input_name: input_batch})[0]  # Shape: (1, 2)
                dist = float(pred[0, 0])
                log_var = float(pred[0, 1])
            else:
                # Dry-run fallback: use geometric distance or simple relative scaling if ONNX isn't ready
                logger.debug("ONNX MLP not active. Running fallback heuristic.")
                if not math.isnan(d_geom):
                    dist = d_geom
                    log_var = float(np.log(max(0.01, 0.15 * d_geom) ** 2))  # 15% variance heuristic
                else:
                    # Synthetic scaling based on depth score
                    dist = float(1.0 + (1.0 - feat_depth) * 10.0)
                    log_var = float(np.log(1.5 ** 2))

            # 3. Compute uncertainty bounds
            sigma = float(np.exp(0.5 * log_var))
            
            # Constrain metric outputs to physical plausibility
            dist = float(np.clip(dist, 0.1, 100.0))
            
            result.final_distance_m = dist
            result.log_variance = log_var
            result.confidence_68 = (float(np.clip(dist - sigma, 0.1, 100.0)), float(np.clip(dist + sigma, 0.1, 100.0)))
            result.confidence_95 = (float(np.clip(dist - 2 * sigma, 0.1, 100.0)), float(np.clip(dist + 2 * sigma, 0.1, 100.0)))

        except Exception as e:
            logger.error(f"Error in FusionEngine: {e}")
            result.final_distance_m = float("nan")
            result.log_variance = float("nan")
            result.confidence_68 = (0.0, 0.0)
            result.confidence_95 = (0.0, 0.0)

        return result
