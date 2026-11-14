# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch import Tensor

from ..op import register_op


@register_op
def rms_norm(
    x: Tensor, weight: Tensor | None, epsilon: float, variance_size: int | None = None
) -> Tensor:
    """Weighted root-mean-square layer normalization."""
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    x_var = x if variance_size is None else x[..., :variance_size]
    variance = x_var.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    if weight is not None:
        x = x.to(weight.dtype) * weight
    return x.to(orig_dtype)


def _fake_norm_output(x: Tensor, weight: Tensor | None, dtype: torch.dtype) -> Tensor:
    scale = torch.empty((*x.shape[:-1], 1), device=x.device, dtype=x.dtype)
    x = x * scale
    if weight is not None:
        x = x.to(weight.dtype) * weight
    return x.to(dtype)


@rms_norm.register_fake
def _rms_norm_fake(
    x: Tensor, weight: Tensor | None, epsilon: float, variance_size: int | None = None
) -> Tensor:
    if x.layout != torch.strided or x.ndim == 0 or variance_size is not None:
        return rms_norm.impls["native"].impl_fn(x, weight, epsilon, variance_size)
    return _fake_norm_output(x.to(torch.float32), weight, x.dtype)


@rms_norm.register_input_generator
def _rms_norm_input_generator(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    epsilon: float = 1e-5,
    device: torch.device | str | None = None,
) -> tuple:
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)
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
    """Fused add and weighted root-mean-square layer normalization."""
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


@fused_add_rms_norm.register_fake
def _fused_add_rms_norm_fake(
    x: Tensor,
    x_residual: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    if (
        x.layout != torch.strided
        or x_residual.layout != torch.strided
        or x.ndim == 0
        or variance_size is not None
    ):
        return fused_add_rms_norm.impls["native"].impl_fn(
            x, x_residual, weight, epsilon, variance_size
        )
    dtype = x.dtype
    x = x.to(torch.float32) + x_residual.to(torch.float32)
    return _fake_norm_output(x, weight, dtype), x.to(dtype)


# fused_add_rms_norm has similar rounding error accumulation as rms_norm
fused_add_rms_norm.override_tolerance(torch.float16, atol=1e-2, rtol=2e-3)


@fused_add_rms_norm.register_input_generator
def _fused_add_rms_norm_input_generator(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    epsilon: float = 1e-5,
    device: torch.device | str | None = None,
) -> tuple:
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    x_residual = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)
    return x, x_residual, weight, epsilon
