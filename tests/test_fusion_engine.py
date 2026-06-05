import pytest
import torch
import numpy as np
import math
from unittest.mock import MagicMock

from data_contract import FrameResult
from src.engines.base_engine import BaseEngine
from src.engines.depth_engine import DepthEngine
from src.mlp_training.model import DistanceFusionMLP, gaussian_nll_loss

def test_mlp_output_shape():
    """Verifies that DistanceFusionMLP processes features and outputs correct dimensions.
    
    Data Contract Verification:
        - Input shape: [Batch, 3] representing (d_geometric_m, rel_depth_score, class_id)
        - Output shape: [Batch, 2] representing (final_distance_m, log_variance)
    """
    batch_size = 8
    model = DistanceFusionMLP(input_dim=3, hidden_dim=32)
    
    # Create dummy tensor representing [geometric_dist, relative_depth, class_id]
    dummy_input = torch.randn(batch_size, 3)
    
    # Run forward pass
    output = model(dummy_input)
    
    # Verify dimensions
    assert output.shape == (batch_size, 2), (
        f"Expected output shape {(batch_size, 2)}, got {output.shape}. "
        "The model must return exactly two outputs per sample: Mean and Log-Variance."
    )


def test_gaussian_nll_loss():
    """Verifies that the gaussian_nll_loss function computes correctly and returns a scalar.
    
    Data Contract Verification:
        - The loss function must accept predictions [Batch, 2] and targets [Batch].
        - The loss must be computed as a single scalar value (0D tensor) for backpropagation.
    """
    batch_size = 16
    pred = torch.randn(batch_size, 2, requires_grad=True)
    target = torch.empty(batch_size).uniform_(1.0, 10.0)
    
    # Calculate loss
    loss = gaussian_nll_loss(pred, target)
    
    # Verify output is a scalar tensor
    assert isinstance(loss, torch.Tensor), "Loss output must be a PyTorch Tensor."
    assert loss.ndim == 0, f"Expected scalar loss (0 dimensions), got {loss.ndim} dimensions."
    assert not torch.isnan(loss), "Loss computation resulted in NaN."
    
    # Verify backpropagation is possible
    loss.backward()
    assert pred.grad is not None, "Gradients were not computed properly during backward pass."


def test_depth_engine_interface():
    """Verifies that DepthEngine conforms to the BaseEngine contract.
    
    Data Contract Verification:
        - DepthEngine must inherit from BaseEngine.
        - DepthEngine.process must accept a FrameResult and return a FrameResult.
        - Output FrameResult must contain valid float relative depth and variance values.
    """
    # Verify class inheritance
    assert issubclass(DepthEngine, BaseEngine), "DepthEngine must inherit from BaseEngine."
    
    # Instantiate engine with dummy parameters (non-NPU fallback mode)
    engine = DepthEngine(hef_path="dummy_path.hef")
    
    # Create test FrameResult input contract
    input_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = FrameResult(
        frame=input_frame,
        timestamp=100.0,
        bbox=(50, 50, 150, 150),
        bbox_height_px=100.0,
        class_id=0,
        class_name="person"
    )
    
    # Process through the depth engine
    output = engine.process(result)
    
    # Verify outputs conform to contract
    assert isinstance(output, FrameResult), "Engine process method must return a FrameResult instance."
    assert not math.isnan(output.rel_depth_score), "rel_depth_score must be computed and populated."
    assert not math.isnan(output.depth_variance), "depth_variance must be computed and populated."
    assert 0.0 <= output.rel_depth_score <= 1.0, f"rel_depth_score must be normalized in [0, 1], got {output.rel_depth_score}."
