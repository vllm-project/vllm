# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

_FP8_DTYPE = current_platform.fp8_dtype()
_FP8_FINFO = torch.finfo(_FP8_DTYPE)
_FP8_MAX = 224.0 if current_platform.is_fp8_fnuz() else _FP8_FINFO.max
_FP8_MIN = -224.0 if current_platform.is_fp8_fnuz() else _FP8_FINFO.min
_FP8_MIN_SCALING_FACTOR = 1.0 / (_FP8_MAX * 512.0)


def quantize_fp8_per_tensor(
        x: torch.Tensor,
        scale: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-tensor FP8 quantization.
    
    Args:
        x: Input tensor to quantize
        scale: Optional pre-computed scale (for static quantization)
    
    Returns:
        Quantized tensor and scale
    """
    if scale is None:
        x_max = x.abs().max().unsqueeze(-1).to(torch.float32)
        scale = (x_max / _FP8_MAX).clamp(min=_FP8_MIN_SCALING_FACTOR)

    # Even for dynamic per-token scales,
    # reciprocal performs slightly better than division
    out = x.to(torch.float32) * scale.reciprocal()
    out = out.clamp(_FP8_MIN, _FP8_MAX).to(_FP8_DTYPE)
    return out, scale


def quantize_fp8_per_token(
    x: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    scale_ub: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-token FP8 quantization.

    Args:
        x: Input tensor to quantize
        scale: Optional pre-computed scale (for static quantization)
        scale_ub: Optional upper bound for scale
    
    Returns:
        Quantized tensor and scale
    """
    if scale is None:
        x_max, _ = x.abs().max(dim=-1)
        x_max = x_max.unsqueeze(-1).to(torch.float32)
        if scale_ub is not None:
            x_max = x_max.clamp(max=scale_ub)
        scale = (x_max / _FP8_MAX).clamp(min=_FP8_MIN_SCALING_FACTOR)

    out = x.to(torch.float32) * scale.reciprocal()
    out = out.clamp(_FP8_MIN, _FP8_MAX).to(_FP8_DTYPE)
    return out, scale


def quantize_fp8_per_group(x: torch.Tensor,
                           group_size: int,
                           column_major_scales: bool = False
                           ) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-group FP8 quantization.

    Args:
        x: Input tensor to quantize
        group_size: Size of quantization groups
        column_major_scales: If True, output scales in column-major format
    
    Returns:
        Quantized tensor and per-group scales
    """
    orig_shape = x.shape
    hidden_dim = x.shape[-1]
    num_groups = (hidden_dim + group_size - 1) // group_size
    padded_dim = num_groups * group_size

    if padded_dim != hidden_dim:
        padding = padded_dim - hidden_dim
        x = F.pad(x, (0, padding), mode='constant', value=0.0)

    x_grouped = x.view(-1, num_groups, group_size)
    absmax = x_grouped.abs().max(dim=-1, keepdim=True)[0].float()
    scales = (absmax / _FP8_MAX).clamp(min=_FP8_MIN_SCALING_FACTOR)

    x_scaled = x_grouped / scales
    x_quant = x_scaled.clamp(_FP8_MIN, _FP8_MAX).to(_FP8_DTYPE)

    x_quant = x_quant.view(-1, padded_dim)
    if padded_dim != hidden_dim:
        x_quant = x_quant[..., :hidden_dim]
    x_quant = x_quant.view(orig_shape)

    scales = scales.squeeze(-1)
    scales = scales.reshape(orig_shape[:-1] + (num_groups, ))

    if column_major_scales:
        scales = scales.transpose(-2, -1).contiguous()

    return x_quant, scales
