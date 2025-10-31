# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
    get_mk_alignment_for_contiguous_layout,
    is_deep_gemm_e8m0_used,
)


class TritonOrDeepGemmExperts(mk.FusedMoEPermuteExpertsUnpermute):
    def __init__(
        self,
        quant_config: FusedMoEQuantConfig,
        allow_deep_gemm: bool = False,
    ):
        super().__init__(quant_config)

        self.triton_expert = TritonExperts(quant_config)

        self.allow_deep_gemm = (
            allow_deep_gemm
            and self.quant_config.use_fp8_w8a8
            and self.block_shape == get_mk_alignment_for_contiguous_layout()
        )

        self.deep_gemm_expert = (
            DeepGemmExperts(self.quant_config) if self.allow_deep_gemm else None
        )

    @property
    def activation_formats(
        self,
    ) -> tuple[mk.FusedMoEActivationFormat, mk.FusedMoEActivationFormat]:
        assert (
            self.deep_gemm_expert is None
            or self.triton_expert.activation_formats
            == self.deep_gemm_expert.activation_formats
        )
        return self.triton_expert.activation_formats

    def supports_chunking(self) -> bool:
        dge = self.deep_gemm_expert
        te = self.triton_expert
        return (dge is None or dge.supports_chunking()) and (
            te is None or te.supports_chunking()
        )

    def supports_expert_map(self) -> bool:
        dge = self.deep_gemm_expert
        te = self.triton_expert
        return (dge is None or dge.supports_expert_map()) and (
            te is None or te.supports_expert_map()
        )

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        dge = self.deep_gemm_expert
        te = self.triton_expert
        dge_war = dge.finalize_weight_and_reduce_impl() if dge else None
        te_war = te.finalize_weight_and_reduce_impl() if te else None
        is_dge_war = dge_war is not None
        is_te_war = te_war is not None

        if is_dge_war and is_te_war:
            assert dge_war == te_war, (
                "Both implementations should agree on WeightAndReduce impls. "
                f"Got dge_war: {dge_war}, and te_war: {te_war}"
            )

        if dge_war is not None:
            return dge_war

        assert te_war is not None
        return te_war

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
        if self.allow_deep_gemm and (
            is_deep_gemm_e8m0_used() or _valid_deep_gemm_shape(M, N, K)
        ):
            assert self.deep_gemm_expert is not None
            return self.deep_gemm_expert.workspace_shapes(
                M,
                N,
                K,
                topk,
                global_num_experts,
                local_num_experts,
                expert_tokens_meta,
            )
        else:
            return self.triton_expert.workspace_shapes(
                M,
                N,
                K,
                topk,
                global_num_experts,
                local_num_experts,
                expert_tokens_meta,
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
        use_deep_gemm = self.allow_deep_gemm and (
            is_deep_gemm_e8m0_used() or _valid_deep_gemm(hidden_states, w1, w2)
        )

        experts = self.deep_gemm_expert if use_deep_gemm else self.triton_expert
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
