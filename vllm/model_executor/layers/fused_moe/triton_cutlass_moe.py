# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.cutlass_moe import CutlassExpertsFp8
from vllm.model_executor.layers.fused_moe.fallback import FallbackExperts
from vllm.model_executor.layers.fused_moe.fused_moe import TritonExperts
from vllm.platforms import current_platform


class TritonOrCutlassExperts(FallbackExperts):
    """Cutlass with fallback to Triton for low latency shapes on SM100."""

    def __init__(
        self,
        e: int,
        n: int,
        k: int,
        out_dtype: torch.dtype | None,
        quant_config: FusedMoEQuantConfig,
        device: torch.dtype,
    ):
        self.is_sm100 = current_platform.has_device_capability(100)
        super().__init__(
            experts=CutlassExpertsFp8(e, n, k, out_dtype, quant_config, device),
            fallback_experts=TritonExperts(quant_config),
        )

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: str,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # Small batch fallback for sm100.
        if self.is_sm100 and M <= 8:
            return self.fallback_experts.workspace_shapes(
                M,
                N,
                K,
                topk,
                global_num_experts,
                local_num_experts,
                expert_tokens_meta,
                activation,
            )
        else:
            return self.experts.workspace_shapes(
                M,
                N,
                K,
                topk,
                global_num_experts,
                local_num_experts,
                expert_tokens_meta,
                activation,
            )

    def _select_experts_impl(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
    ) -> mk.FusedMoEPermuteExpertsUnpermute:
        # Small batch fallback for sm100.
        if self.is_sm100 and hidden_states.shape[0] <= 8:
            return self.fallback_experts
        else:
            return self.experts
