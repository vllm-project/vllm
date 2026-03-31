# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import Tensor

from vllm import ir
from vllm.platforms import current_platform

OINK_AVAILABLE = current_platform.has_device_capability(100) and hasattr(
    torch.ops, "oink"
)


def has_oink_op(name: str) -> bool:
    """Check if a specific oink op is registered."""
    return OINK_AVAILABLE and hasattr(torch.ops.oink, name)


def _can_view_as_2d(x: Tensor) -> bool:
    """Return True if x.view(-1, x.shape[-1]) is viewable (no copy)."""
    if x.dim() < 2:
        return False
    if x.dim() == 2:
        return True
    # For a view(-1, N) to be valid, all leading dims must be contiguous with
    # respect to each other (size-1 dims are ignored).
    for dim in range(x.dim() - 1):
        # Strides for size-1 dims are irrelevant and can be arbitrary.
        if x.size(dim + 1) != 1 and x.stride(dim) != x.stride(dim + 1) * x.size(
            dim + 1
        ):
            return False
    return True


def _is_oink_stride_compatible_2d(x_2d: Tensor) -> bool:
    """Return True if x_2d meets Oink's pointer-path stride constraints."""
    if x_2d.dim() != 2:
        return False
    if x_2d.stride(1) != 1:
        return False
    # Match Oink's vectorization constraint: stride(0) divisible by 256b.
    if x_2d.dtype in (torch.float16, torch.bfloat16):
        divby = 16
    elif x_2d.dtype == torch.float32:
        divby = 8
    else:
        return False
    return (x_2d.stride(0) % divby) == 0


oink_rms_supported = (
    lambda x, weight, epsilon, variance_size=None: variance_size is None
    and weight is not None
    and x.dim() >= 2
    and x.dtype == weight.dtype
    and weight.is_contiguous()
    and _can_view_as_2d(x)
    and _is_oink_stride_compatible_2d(x.view(-1, x.shape[-1]))
)
"""
Oink rms only supports 2d-like inputs with contiguous weight 
and no variance_size override.
"""


@ir.ops.rms_norm.register_impl(
    "oink", supports_args=oink_rms_supported, supported=has_oink_op("rmsnorm")
)
def rms_norm(
    x: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> Tensor:
    assert variance_size is None
    x_2d = x.view(-1, x.shape[-1])
    return torch.ops.oink.rmsnorm(x_2d, weight, epsilon).view_as(x)


oink_add_rms_supported = (
    lambda x, x_residual, weight, epsilon, variance_size=None: variance_size is None
    and weight is not None
    and x.dim() >= 2
    and x.dtype == weight.dtype
    and weight.is_contiguous()
    and _can_view_as_2d(x)
    and _is_oink_stride_compatible_2d(x.view(-1, x.shape[-1]))
    # residual must have 2d-compatible strides and match x shape/dtype
    and x.dtype == x_residual.dtype
    and x.shape == x_residual.shape  # implies _can_view_as_2d(x_residual)
    and _is_oink_stride_compatible_2d(x_residual.view(-1, x_residual.shape[-1]))
)
"""
Oink fused_add_rms_norm has the same constraints as rms_norm,
and residual must be 2d-like with compatible strides.
"""


@ir.ops.fused_add_rms_norm.register_impl(
    "oink",
    supports_args=oink_add_rms_supported,
    supported=has_oink_op("fused_add_rms_norm"),
    inplace=True,
)
def fused_add_rms_norm(
    x: Tensor,
    x_residual: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    assert variance_size is None
    x_2d = x.view(-1, x.shape[-1])
    residual_2d = x_residual.view(-1, x_residual.shape[-1])
    torch.ops.oink.fused_add_rms_norm_(x_2d, residual_2d, weight, epsilon)
    return x, x_residual
