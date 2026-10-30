# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grouped ``wo_a`` output projection shared by DeepSeek-V4 and V4.1."""

import torch

from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    fused_inv_rope_fp8_quant,
)
from vllm.utils.deep_gemm import fp8_einsum


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
        """Inverse RoPE plus the grouped GEMM, ``[num_tokens, n_groups, o_lora_rank]``.

        A QuantizedActivation ``o`` was already rotated and cast by the
        attention kernel, one group per slot with the live groups first, so
        only the padding groups are sliced off.
        """
        use_fp8 = self.weight.dtype == torch.float8_e4m3fn
        if isinstance(o, QuantizedActivation):
            assert use_fp8
            x, x_scale = o.data[:, :n_groups], o.scale[:, :n_groups]
        else:
            x, x_scale = fused_inv_rope_fp8_quant(
                o,
                positions,
                cos_sin_cache,
                n_groups=n_groups,
                heads_per_group=heads_per_group,
                nope_dim=nope_dim,
                rope_dim=rope_dim,
                quant_group_size=einsum_recipe[2],
                tma_aligned_scales=tma_aligned_scales,
                quantize=use_fp8,
            )
        z = torch.empty(
            (x.shape[0], n_groups, o_lora_rank), device=x.device, dtype=torch.bfloat16
        )
        if use_fp8:
            weight_scale = (
                self.weight_scale
                if hasattr(self, "weight_scale")
                else self.weight_scale_inv
            )
            fp8_einsum(
                "bhr,hdr->bhd",
                (x, x_scale),
                (self.weight, weight_scale),
                z,
                recipe=einsum_recipe,
            )
        else:
            grouped_weight = self.weight.view(n_groups, o_lora_rank, -1)
            torch.bmm(
                x.transpose(0, 1), grouped_weight.transpose(1, 2), out=z.transpose(0, 1)
            )
        return z
