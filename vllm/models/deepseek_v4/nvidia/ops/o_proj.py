# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm import envs
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    w8a8_triton_block_scaled_mm,
)
from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import fp8_einsum


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
    weight_scale = (
        wo_a.weight_scale if hasattr(wo_a, "weight_scale") else wo_a.weight_scale_inv
    )
    if envs.VLLM_BATCH_INVARIANT and not tma_aligned_scales:
        # The BI DeepGEMM fork does not currently accept the grouped scale
        # layout used by fp8_einsum on Hopper. Preserve its verified W8A8
        # semantics with the deterministic block-scaled Triton kernel.
        #
        # SM100 scales are packed UE8M0/TMA tensors. They cannot be reshaped as
        # logical FP32 block scales and must be consumed directly by DeepGEMM.
        group_weight = wo_a.weight.reshape(n_groups, o_lora_rank, -1)
        group_weight_scale = weight_scale.reshape(
            n_groups,
            o_lora_rank // 128,
            group_weight.shape[-1] // 128,
        )
        z = torch.stack(
            [
                w8a8_triton_block_scaled_mm(
                    o_fp8[:, group].contiguous(),
                    group_weight[group],
                    o_scale[:, group].contiguous(),
                    group_weight_scale[group],
                    block_size=[128, 128],
                    output_dtype=torch.bfloat16,
                )
                for group in range(n_groups)
            ],
            dim=1,
        )
    else:
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
