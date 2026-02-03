# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.cutlass_moe import CutlassExpertsFp8
from vllm.model_executor.layers.fused_moe.fallback import FallbackExperts
from vllm.model_executor.layers.fused_moe.fused_moe import TritonExperts
from vllm.platforms import current_platform


class TritonOrCutlassExperts(FallbackExperts):
    """Cutlass with fallback to Triton for low latency shapes on SM100."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        self.is_sm100 = current_platform.has_device_capability(100)
        super().__init__(
            experts=CutlassExpertsFp8(moe_config, quant_config),
            fallback_experts=TritonExperts(moe_config, quant_config),
        )

    @staticmethod
    def get_clses() -> tuple[
        type[mk.FusedMoEPermuteExpertsUnpermute],
        type[mk.FusedMoEPermuteExpertsUnpermute],
    ]:
        return (CutlassExpertsFp8, TritonExperts)

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
