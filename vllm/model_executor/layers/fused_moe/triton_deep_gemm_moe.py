# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.deep_gemm_moe import (
    DeepGemmExperts,
    _valid_deep_gemm,
    _valid_deep_gemm_shape,
)
from vllm.model_executor.layers.fused_moe.fused_moe import TritonExperts
from vllm.utils.deep_gemm import (
    is_deep_gemm_e8m0_used,
)


class FallbackExperts(ABC, mk.FusedMoEPermuteExpertsUnpermute):
    def __init__(
        self,
        experts: mk.FusedMoEPermuteExpertsUnpermute,
        fallback_experts: mk.FusedMoEPermuteExpertsUnpermute,
    ):
        super().__init__(experts.quant_config)

        self.fallback_experts = fallback_experts
        self.experts = experts

    @property
    def activation_formats(
        self,
    ) -> tuple[mk.FusedMoEActivationFormat, mk.FusedMoEActivationFormat]:
        assert (
            self.fallback_experts.activation_formats == self.expert.activation_formats
        )
        return self.fallback_experts.activation_formats

    def supports_chunking(self) -> bool:
        return (
            self.expert.supports_chunking()
            and self.fallback_experts.supports_chunking()
        )

    def supports_expert_map(self) -> bool:
        return (
            self.expert.supports_expert_map()
            and self.fallback_experts.supports_expert_map()
        )

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        e_war = self.expert.finalize_weight_and_reduce_impl()
        fbe_war = self.fallback_experts.finalize_weight_and_reduce_impl()
        is_dge_war = e_war is not None
        is_fbe_war = fbe_war is not None

        if is_dge_war and is_fbe_war:
            assert e_war == fbe_war, (
                "Both implementations should agree on WeightAndReduce impls. "
                f"Got e_war: {e_war}, and fbe_war: {fbe_war}"
            )

        if e_war is not None:
            return e_war
        assert fbe_war is not None
        return fbe_war

    @abstractmethod
    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """Logic for allocating workspace depending on experts implementation."""
        raise NotImplementedError

    @abstractmethod
    def _select_experts_impl(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
    ) -> mk.FusedMoEPermuteExpertsUnpermute:
        """Logic for dispatching between experts implementation."""
        raise NotImplementedError

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        experts = self._select_experts_impl(hidden_states, w1, w2)
        experts.apply(
            output,
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation,
            global_num_experts,
            expert_map,
            a1q_scale,
            a2_scale,
            workspace13,
            workspace2,
            expert_tokens_meta,
            apply_router_weight_on_input,
        )


class TritonOrDeepGemmExperts(FallbackExperts):
    def __init__(self, quant_config: FusedMoEQuantConfig):
        super().__init__(
            experts=DeepGemmExperts(quant_config),
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
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # Note: the deep gemm workspaces are strictly larger than the triton
        # workspaces so we can be pessimistic here and allocate for DeepGemm
        # even if we fall back to triton later, e.g. if expert maps are set.
        if is_deep_gemm_e8m0_used() or _valid_deep_gemm_shape(M, N, K):
            return self.experts.workspace_shapes(
                M,
                N,
                K,
                topk,
                global_num_experts,
                local_num_experts,
                expert_tokens_meta,
            )
        else:
            return self.fallback_experts.workspace_shapes(
                M,
                N,
                K,
                topk,
                global_num_experts,
                local_num_experts,
                expert_tokens_meta,
            )

    def select_gemm_impl(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
    ):
        if is_deep_gemm_e8m0_used() or _valid_deep_gemm(hidden_states, w1, w2):
            return self.experts
        else:
            return self.fallback_experts
