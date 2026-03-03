"""
Reference PyTorch baseline model.

Do NOT modify this file. It serves as the reference implementation
that the agent must optimize via custom CUDA kernels.

The agent reads this file to understand:
  - The model architecture (Model class)
  - Input shapes (get_inputs)
  - Initialization parameters (get_init_inputs)
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple example model: computes alpha * a + b.

    This is the baseline implementation using standard PyTorch ops.
    The agent must accelerate the forward pass via custom CUDA kernels.
    """

    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.alpha * a + b


def get_inputs():
    """Return list of sample input tensors for verification and profiling."""
    return [
        torch.randn(1, 128, device="cuda"),
        torch.randn(1, 128, device="cuda"),
    ]


def get_init_inputs():
    """Return arguments for Model.__init__()."""
    return [2.0]
