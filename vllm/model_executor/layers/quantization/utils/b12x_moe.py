# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight preparation helpers for native B12X MoE kernels."""

import torch

from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    swizzle_blockscale,
)
from vllm.utils.math_utils import round_up


def _reorder_w13_to_w31(
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not weight.is_contiguous() or not scale.is_contiguous():
        raise ValueError("NVFP4 MoE weights and scales must be contiguous")
    if weight.size(1) % 2 != 0 or scale.size(1) != weight.size(1):
        raise ValueError("gated NVFP4 MoE weights and scales must have even rows")

    half = weight.size(1) // 2
    return (
        torch.cat((weight[:, half:], weight[:, :half]), dim=1).contiguous(),
        torch.cat((scale[:, half:], scale[:, :half]), dim=1).contiguous(),
    )


def _pad_dim(tensor: torch.Tensor, dim: int, pad_size: int) -> torch.Tensor:
    if pad_size <= 0:
        return tensor

    dim %= tensor.ndim
    shape = list(tensor.shape)
    original_size = shape[dim]
    shape[dim] += pad_size
    padded = tensor.new_zeros(shape)
    slices = [slice(None)] * tensor.ndim
    slices[dim] = slice(0, original_size)
    padded[tuple(slices)] = tensor
    return padded.contiguous()


def _pad_gated_rows(tensor: torch.Tensor, half_pad_size: int) -> torch.Tensor:
    if tensor.size(1) % 2 != 0:
        raise ValueError("gated NVFP4 MoE tensors must have even row counts")
    half_size = tensor.size(1) // 2
    first, second = tensor.split(half_size, dim=1)
    return torch.cat(
        (_pad_dim(first, 1, half_pad_size), _pad_dim(second, 1, half_pad_size)),
        dim=1,
    ).contiguous()


def _pad_gated_weights(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pad_size = round_up(w13_scale.size(1), 128) - w13.size(1)
    if pad_size <= 0:
        return w13, w13_scale, w2, w2_scale
    if w13_scale.size(1) != w13.size(1):
        raise ValueError("w13 weight and scale row counts must match")
    if w13.size(1) % 2 != 0 or pad_size % 2 != 0:
        raise ValueError("gated NVFP4 MoE padding must split evenly")

    half_pad_size = pad_size // 2
    if half_pad_size % 16 != 0:
        raise ValueError("NVFP4 MoE padding must preserve 16-value scale blocks")

    half_size = w13.size(1) // 2
    if w2.size(2) * 2 != half_size:
        raise ValueError("w2 shape does not match gated w13")
    if w2_scale.size(2) * 16 != half_size:
        raise ValueError("w2 scale shape does not match gated w13")

    return (
        _pad_gated_rows(w13, half_pad_size),
        _pad_gated_rows(w13_scale, half_pad_size),
        _pad_dim(w2, 2, half_pad_size // 2),
        _pad_dim(w2_scale, 2, half_pad_size // 16),
    )


def _per_expert_scale(
    scale: torch.Tensor,
    num_experts: int,
    name: str,
) -> torch.Tensor:
    scale = scale.to(torch.float32)
    if scale.dim() == 0:
        return scale.expand(num_experts).contiguous()
    if scale.dim() == 1:
        if scale.numel() != num_experts:
            raise ValueError(
                f"{name} must have {num_experts} elements, got {scale.numel()}"
            )
        return scale.contiguous()
    if scale.dim() == 2:
        if scale.size(0) != num_experts:
            raise ValueError(
                f"{name} first dimension must be {num_experts}, got {scale.size(0)}"
            )
        return scale.max(dim=1).values.contiguous()
    raise ValueError(f"{name} must be scalar, 1D, or 2D, got {tuple(scale.shape)}")


def prepare_nvfp4_moe_layer_for_b12x(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w13_scale_2: torch.Tensor,
    a13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    w2_scale_2: torch.Tensor,
    a2_scale: torch.Tensor,
    is_act_and_mul: bool,
    reorder_w13: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Prepare native B12X NVFP4 MoE weights and scales."""
    num_experts = w13.shape[0]
    a13_scale = _per_expert_scale(a13_scale, num_experts, "a13_scale")
    a2_scale = _per_expert_scale(a2_scale, num_experts, "a2_scale")

    if reorder_w13 and is_act_and_mul:
        w13, w13_scale = _reorder_w13_to_w31(w13, w13_scale)
    if is_act_and_mul:
        w13, w13_scale, w2, w2_scale = _pad_gated_weights(w13, w13_scale, w2, w2_scale)

    w13_scale = swizzle_blockscale(w13_scale)
    pad_size = w13_scale.size(1) - w13.size(1)
    if pad_size > 0:
        if is_act_and_mul:
            raise RuntimeError("gated NVFP4 MoE padding must precede scale swizzling")
        w13 = torch.nn.functional.pad(w13, (0, 0, 0, pad_size))
        w2 = torch.nn.functional.pad(w2, (0, pad_size // 2, 0, 0))
        w2_scale = torch.nn.functional.pad(w2_scale, (0, pad_size // 16))

    w2_scale = swizzle_blockscale(w2_scale)
    return w13, w13_scale, w13_scale_2, a13_scale, w2, w2_scale, w2_scale_2, a2_scale
