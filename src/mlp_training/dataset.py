import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import json
import os
from typing import Tuple, Dict, Any

class DistanceFusionDataset(Dataset):
    """PyTorch Dataset for loading and preprocessing metric distance training pairs.
    
    Loads feature columns (d_geometric_m, rel_depth_score, class_id) and 
    target ground-truth distances (true_distance_m) from a CSV file.
    
    Cleans NaNs by replacing them with the sentinel value -1.0, and supports
    computing/applying normalisation configurations.
    """

    def __init__(
        self,
        csv_path: str,
        is_training: bool = True,
        norm_params: Dict[str, Any] = None
    ) -> None:
        """Initializes the DistanceFusionDataset.
        
        Args:
            csv_path: Path to the CSV file containing the training dataset.
            is_training: True if dataset is used for training (will compute normalization).
                         False if used for validation/test (requires passing norm_params).
            norm_params: Dictionary containing 'mean' and 'std' for feature scaling.
        """
        self.csv_path = csv_path
        self.is_training = is_training
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset CSV file not found at: {csv_path}")

        # Load CSV data
        df = pd.read_csv(csv_path)
        
        # Required columns mapping
        required_cols = ["d_geometric_m", "rel_depth_score", "class_id", "true_distance_m"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}' in dataset CSV.")

        # Clean NaN features with sentinel value (-1.0)
        df["d_geometric_m"] = df["d_geometric_m"].fillna(-1.0)
        df["rel_depth_score"] = df["rel_depth_score"].fillna(-1.0)
        df["class_id"] = df["class_id"].fillna(-1.0)
        
        # Drop rows where target is NaN (we need valid ground truth)
        df = df.dropna(subset=["true_distance_m"])

        # Extract features and targets as numpy arrays
        self.raw_features = df[["d_geometric_m", "rel_depth_score", "class_id"]].values.astype(np.float32)
        self.targets = df["true_distance_m"].values.astype(np.float32)

        # Handle Normalization
        if is_training:
            # Calculate mean and std on training split
            self.mean = np.mean(self.raw_features, axis=0)
            self.std = np.std(self.raw_features, axis=0) + 1e-8
        else:
            if norm_params is None:
                raise ValueError("norm_params must be provided for non-training dataset.")
            self.mean = np.array(norm_params["mean"], dtype=np.float32)
            self.std = np.array(norm_params["std"], dtype=np.float32)

        # Apply normalization: x_norm = (x - mean) / std
        self.features = (self.raw_features - self.mean) / self.std

    def get_norm_params(self) -> Dict[str, list[float]]:
        """Returns the computed normalization parameters as a JSON-serializable dictionary."""
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist()
        }

    def save_norm_params(self, save_path: str) -> None:
        """Saves computed normalization parameters to a JSON file.
        
        This JSON file will be loaded by the FusionEngine on the Raspberry Pi.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(self.get_norm_params(), f, indent=4)

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fetches the normalized feature tensor and target distance tensor at the given index."""
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, y
