# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch import Tensor

from ..op import register_op


@register_op
def rms_norm(
    x: Tensor, weight: Tensor | None, epsilon: float, variance_size: int | None = None
) -> Tensor:
    """Weighted root-mean-square layer normalization"""
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x_var = x if variance_size is None else x[..., :variance_size]
    variance = x_var.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    if weight is not None:
        x = x.to(weight.dtype) * weight
    return x.to(orig_dtype)


@rms_norm.register_input_generator
def _rms_norm_input_generator(
    num_tokens: int, hidden_size: int, dtype: torch.dtype, epsilon: float = 1e-5
) -> tuple:
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    weight = torch.randn(hidden_size, dtype=dtype)
    return x, weight, epsilon


# Reductions in rms_norm accumulate rounding error at large shapes
# (e.g. 32768x16384), causing a few elements out of millions to exceed
# the default float16 tolerance.
rms_norm.override_tolerance(torch.float16, atol=1e-2, rtol=2e-3)


@register_op(allow_inplace=True)
def fused_add_rms_norm(
    x: Tensor,
    x_residual: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Fused add and weighted root-mean-square layer normalization"""
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x = x + x_residual.to(torch.float32)
    x_residual = x.to(orig_dtype)

    x_var = x if variance_size is None else x[..., :variance_size]
    variance = x_var.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    if weight is not None:
        x = x.to(weight.dtype) * weight
    return x.to(orig_dtype), x_residual


# fused_add_rms_norm has similar rounding error accumulation as rms_norm
fused_add_rms_norm.override_tolerance(torch.float16, atol=1e-2, rtol=2e-3)


@fused_add_rms_norm.register_input_generator
def _fused_add_rms_norm_input_generator(
    num_tokens: int, hidden_size: int, dtype: torch.dtype, epsilon: float = 1e-5
) -> tuple:
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    x_residual = torch.randn(num_tokens, hidden_size, dtype=dtype)
    weight = torch.randn(hidden_size, dtype=dtype)
    return x, x_residual, weight, epsilon
