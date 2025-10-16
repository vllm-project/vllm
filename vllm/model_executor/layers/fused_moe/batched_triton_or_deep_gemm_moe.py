# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe import (
    BatchedDeepGemmExperts,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.deep_gemm_utils import deep_gemm_block_shape
from vllm.model_executor.layers.fused_moe.fused_batched_moe import BatchedTritonExperts


class BatchedTritonOrDeepGemmExperts(mk.FusedMoEPermuteExpertsUnpermute):
    def __init__(
        self,
        max_num_tokens: int,
        num_dispatchers: int,
        quant_config: FusedMoEQuantConfig,
        allow_deep_gemm: bool = False,
    ):
        super().__init__(quant_config)

        # Store the original request for deep gemm
        deep_gemm_requested = allow_deep_gemm

        self.allow_deep_gemm = (
            allow_deep_gemm
            and self.quant_config.use_fp8_w8a8
            and self.block_shape == deep_gemm_block_shape()
        )

        self.batched_deep_gemm_experts = (
            BatchedDeepGemmExperts(
                max_num_tokens=max_num_tokens,
                num_dispatchers=num_dispatchers,
                quant_config=self.quant_config,
            )
            if self.allow_deep_gemm
            else None
        )

        # If deep gemm was requested but is not available (either due to
        # unsupported configuration or missing dependencies), check if
        # we should allow fallback to batched triton kernel
        if deep_gemm_requested and self.batched_deep_gemm_experts is None:
            if not envs.VLLM_ALLOW_BATCHED_TRITON_FALLBACK:
                raise RuntimeError(
                    "DeepGemm was requested but is not available. "
                    "The batched triton kernel fallback is disabled by default. "
                    "Set VLLM_ALLOW_BATCHED_TRITON_FALLBACK=1 to enable the fallback "
                    "for debugging purposes."
                )

        self.batched_triton_experts = (
            BatchedTritonExperts(
                max_num_tokens=max_num_tokens,
                num_dispatchers=num_dispatchers,
                quant_config=self.quant_config,
            )
            if self.batched_deep_gemm_experts is None
            else None
        )

        assert (
            self.batched_deep_gemm_experts is not None
            or self.batched_triton_experts is not None
        )

    @property
    def activation_formats(
        self,
    ) -> tuple[mk.FusedMoEActivationFormat, mk.FusedMoEActivationFormat]:
        if self.batched_triton_experts is not None:
            assert (
                self.batched_deep_gemm_experts is None
                or self.batched_deep_gemm_experts.activation_formats
                == self.batched_triton_experts.activation_formats
            )
            return self.batched_triton_experts.activation_formats
        else:
            assert self.batched_deep_gemm_experts is not None
            return self.batched_deep_gemm_experts.activation_formats

    def supports_chunking(self) -> bool:
        bdge = self.batched_deep_gemm_experts
        bte = self.batched_triton_experts
        return (bdge is None or bdge.supports_chunking()) and (
            bte is None or bte.supports_chunking()
        )

    def supports_expert_map(self) -> bool:
        bdge = self.batched_deep_gemm_experts
        bte = self.batched_triton_experts
        return (bdge is None or bdge.supports_expert_map()) and (
            bte is None or bte.supports_expert_map()
        )

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        bdge = self.batched_deep_gemm_experts
        bte = self.batched_triton_experts
        bdge_war = bdge.finalize_weight_and_reduce_impl() if bdge else None
        bte_war = bte.finalize_weight_and_reduce_impl() if bte else None
        is_bdge_war = bdge_war is not None
        is_bte_war = bte_war is not None

        if is_bdge_war and is_bte_war:
            assert bdge_war == bte_war, (
                "Both implementations should agree on WeightAndReduce impls. "
                f"Got bdge_war: {bdge_war}, and bte_war: {bte_war}"
            )

        if bdge_war is not None:
            return bdge_war

        assert bte_war is not None
        return bte_war

    def workspace_dtype(self, act_dtype: torch.dtype) -> torch.dtype:
        return act_dtype

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_metadata: mk.ExpertTokensMetadata | None,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # Note: the deep gemm workspaces are strictly larger than the triton
        # workspaces so we can be pessimistic here and allocate for DeepGemm
        # even if we fall back to triton later, e.g. if expert maps are set.
        if self.allow_deep_gemm:
            assert self.batched_deep_gemm_experts is not None
            return self.batched_deep_gemm_experts.workspace_shapes(
                M,
                N,
                K,
                topk,
                global_num_experts,
                local_num_experts,
                expert_tokens_metadata,
            )
        else:
            assert self.batched_triton_experts is not None
            return self.batched_triton_experts.workspace_shapes(
                M,
                N,
                K,
                topk,
                global_num_experts,
                local_num_experts,
                expert_tokens_metadata,
            )

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
        experts = (
            self.batched_deep_gemm_experts
            if self.allow_deep_gemm
            else self.batched_triton_experts
        )
        assert experts is not None
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
