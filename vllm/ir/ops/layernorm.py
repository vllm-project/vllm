# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn.functional as F
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
    x = x.to(orig_dtype)
    if weight is not None:
        x = x * weight
    return x


@register_op
def rms_norm_gated(
    x: torch.Tensor,
    weight: Tensor,
    bias: Tensor,
    z: torch.Tensor | None,
    epsilon: float,
    group_size: int | None = None,
    norm_before_gate: bool = False,
    activation: str = "",
) -> Tensor:
    orig_dtype = x.dtype
    x = x.float()
    weight = weight.float()
    z = z.float() if z is not None else None

    # Apply gating before normalization if needed
    if z is not None and not norm_before_gate:
        x = x * F.silu(z)

    # RMS Normalization
    if group_size is None:
        # Standard RMS norm across the last dimension
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + epsilon)
        out = x_normed * weight

    else:
        # Group RMS norm
        from einops import rearrange

        x_group = rearrange(x, "... (g d) -> ... g d", d=group_size)
        variance = x_group.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x_group * torch.rsqrt(variance + epsilon)
        out = rearrange(x_normed, "... g d -> ... (g d)") * weight

    # Apply gating after normalization if needed
    if z is not None and norm_before_gate:
        out = out * F.silu(z)

    return out.to(orig_dtype)
