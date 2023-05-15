"""Custom activation functions."""
import torch
import torch.nn as nn

from cacheflow import activation_ops


class SiluAndMul(nn.Module):
    """An activation function for SwiGLU.

    The function computes x -> silu(x[:d]) * x[d:] where d = x.shape[1] // 2.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,        # (num_tokens, 2 * d)
    ) -> torch.Tensor:          # (num_tokens, d)
        num_tokens = x.shape[0]
        d = x.shape[1] // 2
        out = torch.empty(num_tokens, d, dtype=x.dtype, device=x.device)
        activation_ops.silu_and_mul(out, x)
        return out
