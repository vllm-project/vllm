# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import fp8_einsum


def compute_fp8_einsum_recipe() -> tuple[tuple[int, int, int], bool]:
    """fp8_einsum recipe + scale layout for the current GPU arch.

    SM90: FP32 block scales stay [g, r/128, d/128] → sfb_gran_mn=128.
    SM100: INT32 packed scales become [g, r, ...] → sfb_gran_mn=1.
    SM12x (GB10): Hopper K-granularity (1, 128, 128) with FP32 activation
    scales (see below).

    Returns ``(einsum_recipe, tma_aligned_scales)`` for ``deep_gemm_fp8_o_proj``.
    """
    cap = current_platform.get_device_capability()
    assert cap is not None, "DeepseekV4 attention requires a CUDA device"
    if cap.major == 12:
        # Hopper K-granularity (1, 128, 128) with FP32 activation scales.
        # The INT32-packed UE8M0 variant (tma_aligned_scales=True) is
        # numerically WRONG on SM12x: DeepGEMM's einsum misreads the packed
        # lanes (~2^32 error, measured on GB10 vs an fp32 reference). The FP32
        # producer output is TMA-aligned (stride(-1) == tma_aligned_size(T,4))
        # on this codebase, so the FP32→UE8M0 pack path in DeepGEMM consumes
        # it correctly for both aligned and non-aligned token counts.
        return (1, 128, 128), False
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
    weight_scale = (
        wo_a.weight_scale if hasattr(wo_a, "weight_scale") else wo_a.weight_scale_inv
    )
    weight = wo_a.weight
    # fp8_einsum's "bhr,hdr->bhd" runs get_shape<3> on B and its scale and does
    # not reshape: both must be 3D (h, d, r) = (n_groups, o_lora_rank, D) and
    # (n_groups, o_lora_rank // 128, D // 128). Layers loaded outside the
    # DeepGEMM scaled-mm path (b12x/FlashMLA/FlashInfer backends) keep the flat
    # checkpoint layout (n_groups*o_lora_rank, D), which trips the shape assert.
    if weight.ndim == 2:
        weight = weight.view(n_groups, o_lora_rank, -1)
        weight_scale = weight_scale.view(n_groups, o_lora_rank // 128, -1)
    fp8_einsum(
        "bhr,hdr->bhd",
        (o_fp8, o_scale),
        (weight, weight_scale),
        z,
        recipe=einsum_recipe,
    )
    return wo_b(z.flatten(1))
