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
    if weight is not None:
        x = x.to(weight.dtype) * weight
    return x.to(orig_dtype)


@register_op
def mixer2_rms_norm_gated(
    x: Tensor,
    gate: Tensor,
    weight: Tensor | None,
    epsilon: float,
    group_size: int | None = None,
) -> Tensor:
    input_dtype = x.dtype
    x = x * F.silu(gate.to(torch.float32))
    if group_size is None:
        # Standard RMSNorm: compute variance over the full hidden dimension
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + epsilon)
    else:
        # Grouped RMSNorm: compute variance independently within each group
        *prefix_dims, hidden_dims = x.shape
        x_grouped = x.view(*prefix_dims, hidden_dims // group_size, group_size)
        variance = x_grouped.pow(2).mean(dim=-1, keepdim=True)
        x_grouped = x_grouped * torch.rsqrt(variance + epsilon)
        x = x_grouped.view(*prefix_dims, hidden_dims)

    if weight is not None:
        x = x.to(weight.dtype) * weight
    return x.to(input_dtype)
