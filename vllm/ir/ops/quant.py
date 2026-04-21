# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn.functional as F
from torch import Tensor

from ..op import register_op


def get_fp8_min_max(fp8_dtype: torch.dtype) -> tuple[float, float]:
    """Get min/max representable values for FP8 quantization."""
    if fp8_dtype == torch.float8_e4m3fnuz:
        return -224.0, 224.0
    return torch.finfo(fp8_dtype).min, torch.finfo(fp8_dtype).max


def _pad_token_dim(out: Tensor, num_token_padding: int | None) -> Tensor:
    # This currently generates an extra Triton kernel in compilation.
    # Fortunately, we don't use padding if compiling.
    # TODO(luka): benchmark torch._scaled_mm to hopefully remove padding
    #  in general.
    if num_token_padding is not None:
        padding = max(num_token_padding - out.size(0), 0)
        if padding > 0:
            out = F.pad(out, (0, 0, 0, padding), "constant", 0.0)
    return out


@register_op
def static_quant_fp8(
    x: Tensor,
    scale: Tensor,
    fp8_dtype: torch.dtype,
    num_token_padding: int | None = None,
) -> Tensor:
    """Static per-tensor or per-token FP8 quantization with pre-computed scale"""
    fp8_min, fp8_max = get_fp8_min_max(fp8_dtype)
    out = x.to(torch.float32) * scale.to(torch.float32).reciprocal()
    out_clamped = out.clamp(fp8_min, fp8_max).to(fp8_dtype)
    return _pad_token_dim(out_clamped, num_token_padding)


@register_op
def static_group_quant_fp8(
    x: Tensor,
    scale: Tensor,
    fp8_dtype: torch.dtype,
    num_token_padding: int | None = None,
) -> Tensor:
    """Static group FP8 quantization with pre-computed per-group scales"""
    fp8_min, fp8_max = get_fp8_min_max(fp8_dtype)

    # Normalize scale to 2D: [num_groups] -> [1, num_groups]
    # Example: [1, 2] shape (2,) -> [[1, 2]] shape (1, 2)
    if scale.ndim == 1:
        scale = scale.unsqueeze(-2)

    # Group broadcast: repeat each group scale across group_size elements
    # Example: x shape (M, 4), scale [[1, 2]] shape (1, 2) ->
    # [[1, 1, 2, 2]] shape (1, 4) for group_size=2

    target_cols = x.shape[-1]
    scale_cols = scale.shape[-1]

    assert target_cols % scale_cols == 0
    repeat_factor = target_cols // scale_cols
    scale = scale.repeat_interleave(repeat_factor, dim=-1)

    out = (
        (x.to(torch.float32) * scale.to(torch.float32).reciprocal())
        .clamp(fp8_min, fp8_max)
        .to(fp8_dtype)
    )
    return _pad_token_dim(out, num_token_padding)


@register_op
def dynamic_quant_fp8(
    x: Tensor,
    per_token: bool,
    fp8_dtype: torch.dtype,
    scale_ub: Tensor | None = None,
    num_token_padding: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Dynamic per-tensor or per-token FP8 quantization with computed scale"""
    fp8_min, fp8_max = get_fp8_min_max(fp8_dtype)
    fp8_min_scaling_factor = 1.0 / (fp8_max * 512.0)
    if per_token:
        x_max, _ = x.abs().max(dim=-1)
        x_max = x_max.unsqueeze(-1).to(torch.float32)
    else:
        x_max = x.abs().max().unsqueeze(-1).to(torch.float32)

    if scale_ub is not None:
        x_max = x_max.clamp(max=scale_ub)

    scale = (x_max / fp8_max).clamp(min=fp8_min_scaling_factor)
    out = (
        (x.to(torch.float32) * scale.to(torch.float32).reciprocal())
        .clamp(fp8_min, fp8_max)
        .to(fp8_dtype)
    )
    return _pad_token_dim(out, num_token_padding), scale


@register_op
def dynamic_group_quant_fp8(
    x: Tensor,
    group_shape: list[int],
    column_major: bool,
    use_ue8m0: bool,
    fp8_dtype: torch.dtype,
    scale_alignment: int = 1,
) -> tuple[Tensor, Tensor]:
    """Dynamic group FP8 quantization with computed per-group scales"""
    fp8_min, fp8_max = get_fp8_min_max(fp8_dtype)
    fp8_min_scaling_factor = 1.0 / (fp8_max * 512.0)
    orig_shape = x.shape
    hidden_dim = x.shape[-1]
    group_size = group_shape[-1]
    num_groups = (hidden_dim + group_size - 1) // group_size
    padded_dim = num_groups * group_size

    if padded_dim != hidden_dim:
        padding = padded_dim - hidden_dim
        x = F.pad(x, (0, padding), mode="constant", value=0.0)

    x_grouped = x.view(-1, num_groups, group_size)
    absmax = x_grouped.abs().max(dim=-1, keepdim=True)[0].to(torch.float32)
    scales_raw = absmax / fp8_max
    if use_ue8m0:
        scales_raw = torch.exp2(torch.ceil(torch.log2(scales_raw)))
    scales = scales_raw.clamp(min=fp8_min_scaling_factor)

    x_scaled = x_grouped / scales
    x_quant = x_scaled.clamp(fp8_min, fp8_max).to(fp8_dtype)

    x_quant = x_quant.view(-1, padded_dim)
    if padded_dim != hidden_dim:
        x_quant = x_quant[..., :hidden_dim]
    x_quant = x_quant.view(orig_shape)

    scales = scales.squeeze(-1)
    scales = scales.reshape(orig_shape[:-1] + (num_groups,))
    if column_major:
        scales = scales.transpose(-2, -1).contiguous().transpose(-1, -2)

    return x_quant, scales
