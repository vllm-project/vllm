# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Weight preparation helpers for b12x MoE kernels."""

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    swap_w13_to_w31,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    swizzle_blockscale,
)
from vllm.utils.math_utils import round_up


def _pad_gated_weights(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if w13_scale.size(1) != w13.size(1):
        raise ValueError("w13 weight and scale row counts must match")
    half_size = w13.size(1) // 2
    if half_size * 2 != w13.size(1):
        raise ValueError("gated NVFP4 MoE weights must have even row counts")
    if w2.size(2) * 2 != half_size:
        raise ValueError("w2 shape does not match gated w13")
    if w2_scale.size(2) * 16 != half_size:
        raise ValueError("w2 scale shape does not match gated w13")

    half_pad_size = round_up(half_size, 64) - half_size
    if half_pad_size == 0:
        return w13, w13_scale, w2, w2_scale

    def pad_rows(tensor: torch.Tensor) -> torch.Tensor:
        shape = tensor.shape
        return F.pad(
            tensor.reshape(shape[0], 2, half_size, *shape[2:]),
            (0, 0, 0, half_pad_size),
        ).flatten(1, 2)

    return (
        pad_rows(w13),
        pad_rows(w13_scale),
        F.pad(w2, (0, half_pad_size // 2)),
        F.pad(w2_scale, (0, half_pad_size // 16)),
    )


def _per_expert_scale(
    scale: torch.Tensor,
    num_experts: int,
    name: str,
) -> torch.Tensor:
    if scale.dim() == 0:
        scale = scale.expand(num_experts)
    elif scale.dim() == 2:
        scale = scale.amax(dim=1)
    if scale.dim() != 1 or scale.numel() != num_experts:
        raise ValueError(f"{name} must contain one value per expert")
    return scale.to(torch.float32).contiguous()


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
    """Prepare b12x NVFP4 MoE weights and scales."""
    num_experts = w13.shape[0]
    a13_scale = _per_expert_scale(a13_scale, num_experts, "a13_scale")
    a2_scale = _per_expert_scale(a2_scale, num_experts, "a2_scale")

    if reorder_w13 and is_act_and_mul:
        w13 = swap_w13_to_w31(w13)
        w13_scale = swap_w13_to_w31(w13_scale)
    if is_act_and_mul:
        w13, w13_scale, w2, w2_scale = _pad_gated_weights(w13, w13_scale, w2, w2_scale)

    w13_scale = swizzle_blockscale(w13_scale)
    pad_size = w13_scale.size(1) - w13.size(1)
    if pad_size > 0:
        if is_act_and_mul:
            raise RuntimeError("gated NVFP4 MoE padding must precede scale swizzling")
        w13 = F.pad(w13, (0, 0, 0, pad_size))
        w2 = F.pad(w2, (0, pad_size // 2, 0, 0))
        w2_scale = F.pad(w2_scale, (0, pad_size // 16))

    w2_scale = swizzle_blockscale(w2_scale)
    return w13, w13_scale, w13_scale_2, a13_scale, w2, w2_scale, w2_scale_2, a2_scale
