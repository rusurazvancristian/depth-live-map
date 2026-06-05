import torch
import torch.nn as nn
from typing import Tuple

class DistanceFusionMLP(nn.Module):
    """PyTorch Multi-Layer Perceptron for fusing geometric and relative depth estimates.
    
    Accepts normalized features and outputs a metric distance prediction along with 
    a log variance for calibrated uncertainty estimation.
    
    Inputs (shape: [Batch, 3]):
        [0] d_geometric_m     - Pinhole camera geometric estimate (metres)
        [1] rel_depth_score   - Median relative depth from Depth Anything V2 [0, 1]
        [2] class_id          - Continuous float value of the object class ID
        
    Outputs (shape: [Batch, 2]):
        [0] final_distance_m  - Fused metric distance prediction (metres)
        [1] log_variance      - Natural logarithm of prediction variance, ln(σ²)
    """
    
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, dropout_rate: float = 0.1) -> None:
        """Initializes the DistanceFusionMLP network structure.
        
        Args:
            input_dim: Dimension of input features (default is 3).
            hidden_dim: Number of neurons in hidden layers.
            dropout_rate: Dropout probability for regularization.
        """
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 2)  # Outputs [distance, log_variance]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass.
        
        Args:
            x: Input tensor of shape (B, 3).
            
        Returns:
            Output tensor of shape (B, 2) with predicted distance and log variance.
        """
        return self.net(x)


def gaussian_nll_loss(
    pred: torch.Tensor, 
    target: torch.Tensor
) -> torch.Tensor:
    """Computes the Gaussian Negative Log-Likelihood (NLL) Loss.
    
    This loss function penalizes prediction error while calibrating predicted log variance.
    If the prediction error is high, the model is incentivized to increase log_variance.
    If the prediction error is low, the model is incentivized to decrease log_variance.
    
    Loss = 0.5 * [log_var + (target - pred_dist)² / exp(log_var)]
    
    Args:
        pred: Model predictions of shape (B, 2) -> [pred_dist, log_var].
        target: Ground-truth target distances of shape (B,) or (B, 1).
        
    Returns:
        Scalar loss tensor.
    """
    # Extract prediction elements
    pred_dist = pred[:, 0]
    log_var = pred[:, 1]
    
    # Ensure target is matching 1D shape (B,)
    target = target.view(-1)
    
    # Calculate precision (1 / variance)
    precision = torch.exp(-log_var)
    
    # Compute the NLL loss term
    loss = 0.5 * (log_var + ((target - pred_dist) ** 2) * precision)
    
    return torch.mean(loss)
