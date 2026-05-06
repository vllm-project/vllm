# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn.functional as F
from torch import Tensor

from ..op import register_op


@register_op
def fatrelu_and_mul(x: Tensor, threshold: float = 0.0) -> Tensor:
    """FATReLU gated activation: threshold(x[:d]) * x[d:]

    Computes FATReLU(x[:d]) * x[d:] where FATReLU zeros values below threshold.
    Used in openbmb/MiniCPM-S-1B-sft.

    Shapes:
        x: (..., 2 * d)
        return: (..., d)
    """
    d = x.shape[-1] // 2
    return F.threshold(x[..., :d], threshold, 0.0) * x[..., d:]


@fatrelu_and_mul.register_input_generator
def _fatrelu_and_mul_input_generator(
    num_tokens: int, hidden_size: int, dtype: torch.dtype, threshold: float = 0.0
) -> tuple:
    x = torch.randn(num_tokens, hidden_size * 2, dtype=dtype)
    return (x, threshold)


@register_op
def relu2(x: Tensor) -> Tensor:
    """ReLU-squared activation: relu(x)^2

    Applies the relu^2 activation introduced in https://arxiv.org/abs/2109.08668v2

    Shapes:
        x: (..., d)
        return: (..., d)
    """
    return torch.square(F.relu(x))


@relu2.register_input_generator
def _relu2_input_generator(
    num_tokens: int, hidden_size: int, dtype: torch.dtype
) -> tuple:
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    return (x,)


@register_op
def gelu_and_mul_sparse(
    x: Tensor, std_multiplier: float, approximate: str = "none"
) -> Tensor:
    """Sparse GeGLU activation: gelu(gaussian_topk(x[:d])) * x[d:]

    Applies a Gaussian-based top-k sparsification to the gate before GELU.
    Used in Gemma3n models.

    Args:
        x: Input tensor with shape (..., 2 * d)
        std_multiplier: Threshold multiplier derived from target sparsity via
            normal distribution icdf (normal_dist.icdf(activation_sparsity)).
        approximate: GELU approximation mode, 'none' or 'tanh'.

    Shapes:
        x: (..., 2 * d)
        return: (..., d)
    """
    d = x.shape[-1] // 2
    gate = x[..., :d]
    mean = torch.mean(gate, dim=-1, keepdim=True)
    std = torch.std(gate, dim=-1, keepdim=True, unbiased=False)
    sparse_gate = F.relu(gate - (mean + std * std_multiplier))
    return F.gelu(sparse_gate, approximate=approximate) * x[..., d:]


@gelu_and_mul_sparse.register_input_generator
def _gelu_and_mul_sparse_input_generator(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    std_multiplier: float = 1.28,
    approximate: str = "none",
) -> tuple:
    # std_multiplier=1.28 corresponds to ~90% sparsity (icdf(0.9) ≈ 1.28)
    x = torch.randn(num_tokens, hidden_size * 2, dtype=dtype)
    return (x, std_multiplier, approximate)


@register_op
def swigluoai_and_mul(x: Tensor, alpha: float = 1.702, limit: float = 7.0) -> Tensor:
    """SwiGLU-OAI gated activation with clamping, interleaved input layout.

    Computes (up + 1) * (gate * sigmoid(alpha * gate)) where gate/up are
    interleaved: gate = x[..., ::2], up = x[..., 1::2], with clamping.
    Reference: https://github.com/huggingface/transformers/blob/v4.55.0/src/
    transformers/models/gpt_oss/modeling_gpt_oss.py#L106-L110

    Shapes:
        x: (..., 2 * d)  — interleaved gate/up layout
        return: (..., d)
    """
    gate, up = x[..., ::2], x[..., 1::2]
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return (up + 1) * (gate * torch.sigmoid(alpha * gate))


@swigluoai_and_mul.register_input_generator
def _swigluoai_and_mul_input_generator(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    alpha: float = 1.702,
    limit: float = 7.0,
) -> tuple:
    x = torch.randn(num_tokens, hidden_size * 2, dtype=dtype)
    return (x, alpha, limit)
