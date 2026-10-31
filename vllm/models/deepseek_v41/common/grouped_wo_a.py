# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grouped ``wo_a`` output projection for DeepSeek-V4.1."""

import torch

from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.models.deepseek_v41.common.ops import fused_inv_rope_fp8_quant


class DeepseekV4GroupedWoA(ColumnParallelLinear):
    """``wo_a`` as one projection per attention head group.

    Group ``g`` of the inverse-rotated attention output only meets output rows
    ``g * o_lora_rank : (g + 1) * o_lora_rank``.
    """

    def grouped_input(
        self,
        o: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        *,
        n_groups: int,
        heads_per_group: int,
        nope_dim: int,
        rope_dim: int,
    ) -> torch.Tensor:
        """The BF16 inverse-rotated groups, ``[num_tokens, n_groups, group_dim]``."""
        x, _ = fused_inv_rope_fp8_quant(
            o,
            positions,
            cos_sin_cache,
            n_groups=n_groups,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            quantize=False,
        )
        return x

    def grouped_forward(
        self,
        o: torch.Tensor | QuantizedActivation,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        *,
        n_groups: int,
        heads_per_group: int,
        nope_dim: int,
        rope_dim: int,
        o_lora_rank: int,
        einsum_recipe: tuple[int, int, int],
        tma_aligned_scales: bool,
    ) -> torch.Tensor:
        """Inverse RoPE plus grouped GEMM: ``[num_tokens, n_groups, o_lora_rank]``."""
        # NVIDIA-only; the shared attention layer must not import nvidia/ops.
        from vllm.models.deepseek_v4.nvidia.ops.o_proj import deep_gemm_fp8_o_proj

        z = deep_gemm_fp8_o_proj(
            o,
            positions,
            cos_sin_cache,
            self,
            lambda flat: flat,
            n_groups=n_groups,
            heads_per_group=heads_per_group,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            o_lora_rank=o_lora_rank,
            einsum_recipe=einsum_recipe,
            tma_aligned_scales=tma_aligned_scales,
        )
        return z.view(z.shape[0], n_groups, o_lora_rank)
