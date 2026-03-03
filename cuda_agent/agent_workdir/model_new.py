"""
Optimized model using custom CUDA extensions.

The agent modifies this file to replace PyTorch operations with
custom CUDA kernels compiled into the cuda_extension module.

Rules:
  - Must have the same API signature as model.py (Model class,
    get_inputs, get_init_inputs).
  - May NOT use torch.nn.functional operations.
  - Must call cuda_extension.* for all computation.
"""

import torch
import torch.nn as nn

import cuda_extension  # noqa: F401 — compiled via utils/compile.sh


class ModelNew(nn.Module):
    """
    Optimized model: computes alpha * a + b via custom CUDA axpby kernel.
    """

    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Call the custom CUDA kernel: out[i] = alpha * a[i] + b[i]
        return cuda_extension.axpby_forward(a, b, self.alpha, 0)


def get_inputs():
    return [
        torch.randn(1, 128, device="cuda"),
        torch.randn(1, 128, device="cuda"),
    ]


def get_init_inputs():
    return [2.0]
