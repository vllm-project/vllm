# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import fp8_einsum


def get_fp8_weight_scale(layer: nn.Module) -> torch.Tensor | None:
    if hasattr(layer, "weight_scale_inv"):
        return layer.weight_scale_inv
    if hasattr(layer, "weight_scale"):
        return layer.weight_scale
    return None


def maybe_unpack_linear_output(
    output: torch.Tensor | tuple[torch.Tensor, torch.Tensor | None],
) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output


def inv_rope_bf16_o_proj(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
) -> torch.Tensor:
    num_tokens, num_heads, head_dim = o.shape
    expected_heads = n_groups * heads_per_group
    expected_head_dim = nope_dim + rope_dim
    if num_heads != expected_heads:
        raise ValueError(f"Expected {expected_heads} heads, got {num_heads}.")
    if head_dim != expected_head_dim:
        raise ValueError(
            f"Expected head dimension {expected_head_dim}, got {head_dim}."
        )
    if rope_dim % 2 != 0:
        raise ValueError(f"rope_dim must be even, got {rope_dim}.")

    grouped = o.reshape(num_tokens, n_groups, heads_per_group, head_dim)
    projected = grouped.clone()

    rope = projected[..., nope_dim:]
    rope_pairs = rope.reshape(*rope.shape[:-1], rope_dim // 2, 2)
    cos_sin = cos_sin_cache.index_select(0, positions)
    cos, sin = cos_sin.chunk(2, dim=-1)
    cos = cos[:, None, None, :, None].to(dtype=rope.dtype)
    sin = sin[:, None, None, :, None].to(dtype=rope.dtype)

    x0 = rope_pairs[..., 0:1]
    x1 = rope_pairs[..., 1:2]
    rope_pairs.copy_(torch.cat((x0 * cos + x1 * sin, x1 * cos - x0 * sin), dim=-1))

    wo_a_weight = getattr(wo_a, "weight", None)
    wo_a_input_size = (
        wo_a_weight.shape[-1]
        if wo_a_weight is not None and wo_a_weight.ndim >= 2
        else getattr(wo_a, "input_size", heads_per_group * head_dim)
    )
    flattened_size = num_heads * head_dim
    if flattened_size % wo_a_input_size != 0:
        raise ValueError(
            "Cannot reshape O-proj input of size "
            f"{flattened_size} into groups of size {wo_a_input_size}."
        )

    wo_a_groups = flattened_size // wo_a_input_size
    wo_a_input = projected.reshape(num_tokens, wo_a_groups, wo_a_input_size)

    if (
        wo_a_weight is not None
        and wo_a_weight.ndim == 2
        and wo_a_weight.shape[0] % o_lora_rank == 0
        and wo_a_weight.shape[0] // o_lora_rank == wo_a_groups
    ):
        grouped_weight = wo_a_weight.reshape(wo_a_groups, o_lora_rank, wo_a_input_size)
        return torch.einsum("bgi,gri->bgr", wo_a_input, grouped_weight)

    return maybe_unpack_linear_output(wo_a(wo_a_input))


def compute_fp8_einsum_recipe() -> tuple[tuple[int, int, int], bool]:
    """fp8_einsum recipe + scale layout for the current GPU arch.

    SM90: FP32 block scales stay [g, r/128, d/128] → sfb_gran_mn=128.
    SM100: INT32 packed scales become [g, r, ...] → sfb_gran_mn=1.

    Returns ``(einsum_recipe, tma_aligned_scales)`` for ``deep_gemm_fp8_o_proj``.
    """
    cap = current_platform.get_device_capability()
    assert cap is not None, "DeepseekV4 attention requires a CUDA device"
    einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, 128)
    tma_aligned_scales = cap.major >= 10
    return einsum_recipe, tma_aligned_scales


def deep_gemm_fp8_o_proj(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    wo_b: nn.Module,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
    einsum_recipe: tuple[int, int, int],
    tma_aligned_scales: bool,
) -> torch.Tensor:
    """O projection: inverse RoPE + FP8 quant + einsum + wo_b.

    Shared by the FlashMLA and FlashInfer CUDA backends. ``einsum_recipe`` /
    ``tma_aligned_scales`` come from ``compute_fp8_einsum_recipe``.
    """
    weight_scale = get_fp8_weight_scale(wo_a)
    if weight_scale is None:
        z = inv_rope_bf16_o_proj(
            o,
            positions,
            cos_sin_cache,
            wo_a,
            n_groups=n_groups,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            o_lora_rank=o_lora_rank,
        )
        return wo_b(z.flatten(1))

    o_fp8, o_scale = fused_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
        tma_aligned_scales=tma_aligned_scales,
    )
    z = torch.empty(
        (o.shape[0], n_groups, o_lora_rank),
        device=o.device,
        dtype=torch.bfloat16,
    )
    fp8_einsum(
        "bhr,hdr->bhd",
        (o_fp8, o_scale),
        (wo_a.weight, weight_scale),
        z,
        recipe=einsum_recipe,
    )
    return wo_b(z.flatten(1))
