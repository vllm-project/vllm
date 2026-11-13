# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import CutlassExpertsFp8
from vllm.model_executor.layers.fused_moe.experts.fallback import FallbackExperts
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.platforms import current_platform


class TritonOrCutlassExperts(FallbackExperts):
    """Cutlass with fallback to Triton for low latency shapes on SM100/SM120."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        # Batches of up to this many tokens run on Triton. On SM120 the MoE
        # GEMMs are bandwidth-bound well past decode sizes and Triton is
        # faster there; CUTLASS wins from ~1.5k tokens.
        if current_platform.is_device_capability_family(120):
            self.triton_max_tokens = 1024
        elif current_platform.has_device_capability(100):
            self.triton_max_tokens = 8
        else:
            self.triton_max_tokens = 0
        super().__init__(
            experts=CutlassExpertsFp8(moe_config, quant_config),
            fallback_experts=TritonExperts(moe_config, quant_config),
        )

    @staticmethod
    def get_clses() -> tuple[
        type[mk.FusedMoEExpertsModular],
        type[mk.FusedMoEExpertsModular],
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
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        if self.triton_max_tokens >= M:
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
    ) -> mk.FusedMoEExpertsModular:
        if hidden_states.shape[0] <= self.triton_max_tokens:
            return self.fallback_experts
        else:
            return self.experts
