# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn.functional as F
from torch import Tensor

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
    group_broadcast,
)
from vllm.platforms import current_platform

from ..op import register_op

_FP8_DTYPE = current_platform.fp8_dtype()
_FP8_MIN, _FP8_MAX = get_fp8_min_max()
_FP8_MIN_SCALING_FACTOR = 1.0 / (_FP8_MAX * 512.0)


def quant_fp8(x: Tensor, scale: Tensor) -> Tensor:
    out = (
        x.to(torch.float32)
        * group_broadcast(scale.to(torch.float32), x.shape[-2:]).reciprocal()
    )
    return out.clamp(_FP8_MIN, _FP8_MAX).to(_FP8_DTYPE)


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
    x: Tensor, scale: Tensor, num_token_padding: int | None = None
) -> Tensor:
    return _pad_token_dim(quant_fp8(x, scale), num_token_padding)


@register_op
def static_group_quant_fp8(
    x: Tensor, scale: Tensor, num_token_padding: int | None = None
) -> Tensor:
    return _pad_token_dim(quant_fp8(x, scale), num_token_padding)


@register_op
def dynamic_quant_fp8(
    x: Tensor,
    per_token: bool,
    scale_ub: Tensor | None = None,
    num_token_padding: int | None = None,
) -> tuple[Tensor, Tensor]:
    if per_token:
        x_max, _ = x.abs().max(dim=-1)
        x_max = x_max.unsqueeze(-1).to(torch.float32)
        if scale_ub is not None:
            x_max = x_max.clamp(max=scale_ub)
    else:
        x_max = x.abs().max().unsqueeze(-1).to(torch.float32)
    scale = (x_max / _FP8_MAX).clamp(min=_FP8_MIN_SCALING_FACTOR)
    return _pad_token_dim(quant_fp8(x, scale), num_token_padding), scale


@register_op
def dynamic_group_quant_fp8(
    x: Tensor,
    group_shape: list[int],
    column_major: bool,
    use_ue8m0: bool,
    scale_alignment: int = 1,
) -> tuple[Tensor, Tensor]:
    orig_shape = x.shape
    hidden_dim = x.shape[-1]
    group_size = group_shape[-1]
    num_groups = (hidden_dim + group_size - 1) // group_size
    padded_dim = num_groups * group_size

    if padded_dim != hidden_dim:
        padding = padded_dim - hidden_dim
        x = F.pad(x, (0, padding), mode="constant", value=0.0)

    x_grouped = x.view(-1, num_groups, group_size)
    absmax = x_grouped.abs().max(dim=-1, keepdim=True)[0].float()
    scales_raw = absmax / _FP8_MAX
    if use_ue8m0:
        scales_raw = torch.exp2(torch.ceil(torch.log2(scales_raw)))
    scales = (scales_raw).clamp(min=_FP8_MIN_SCALING_FACTOR)

    x_scaled = x_grouped / scales
    x_quant = x_scaled.clamp(_FP8_MIN, _FP8_MAX).to(_FP8_DTYPE)

    x_quant = x_quant.view(-1, padded_dim)
    if padded_dim != hidden_dim:
        x_quant = x_quant[..., :hidden_dim]
    x_quant = x_quant.view(orig_shape)

    scales = scales.squeeze(-1)
    scales = scales.reshape(orig_shape[:-1] + (num_groups,))
    if column_major:
        scales = scales.transpose(-2, -1).contiguous().transpose(-1, -2)

    return x_quant, scales
