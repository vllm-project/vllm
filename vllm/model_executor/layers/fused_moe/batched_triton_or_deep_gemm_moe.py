# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, Optional

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe import (
    BatchedDeepGemmExperts)
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.fused_batched_moe import (
    BatchedTritonExperts)


class BatchedTritonOrDeepGemmExperts(mk.FusedMoEPermuteExpertsUnpermute):

    def __init__(self,
                 max_num_tokens: int,
                 num_dispatchers: int,
                 use_fp8_w8a8: bool = False,
                 use_int8_w8a8: bool = False,
                 use_int8_w8a16: bool = False,
                 use_int4_w4a16: bool = False,
                 block_shape: Optional[list[int]] = None,
                 per_act_token_quant: bool = False,
                 allow_deep_gemm: bool = False):
        assert not use_int8_w8a8, "NYI"
        assert not use_int8_w8a16, "NYI"
        assert not use_int4_w4a16, "NYI"

        super().__init__(
            FusedMoEQuantConfig.make(
                use_fp8_w8a8=use_fp8_w8a8,
                use_int8_w8a8=use_int8_w8a8,
                use_int8_w8a16=use_int8_w8a16,
                use_int4_w4a16=use_int4_w4a16,
                block_shape=block_shape,
                per_act_token_quant=per_act_token_quant,
            ))

        self.batched_triton_experts = BatchedTritonExperts(
            max_num_tokens=max_num_tokens,
            num_dispatchers=num_dispatchers,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_act_token_quant=self.per_act_token_quant,
            block_shape=self.block_shape,
        )

        self.allow_deep_gemm = (allow_deep_gemm and use_fp8_w8a8
                                and self.block_shape
                                == BatchedDeepGemmExperts.DEEPGEMM_BLOCK_SHAPE)

        self.batched_deep_gemm_experts = BatchedDeepGemmExperts(
            max_num_tokens=max_num_tokens,
            num_dispatchers=num_dispatchers,
            block_shape=self.block_shape,  # type: ignore[arg-type]
        ) if self.allow_deep_gemm else None

        assert (self.batched_deep_gemm_experts is not None
                or self.batched_triton_experts is not None)

    @property
    def activation_formats(
        self
    ) -> tuple[mk.FusedMoEActivationFormat, mk.FusedMoEActivationFormat]:
        if self.batched_triton_experts is not None:
            assert (self.batched_deep_gemm_experts is None
                    or self.batched_deep_gemm_experts.activation_formats
                    == self.batched_triton_experts.activation_formats)
            return self.batched_triton_experts.activation_formats
        else:
            assert self.batched_deep_gemm_experts is not None
            return self.batched_deep_gemm_experts.activation_formats

    def supports_chunking(self) -> bool:
        bdge = self.batched_deep_gemm_experts
        bte = self.batched_triton_experts
        return ((bdge is None or bdge.supports_chunking())
                and (bte is None or bte.supports_chunking()))

    def supports_expert_map(self) -> bool:
        bdge = self.batched_deep_gemm_experts
        bte = self.batched_triton_experts
        return ((bdge is None or bdge.supports_expert_map())
                and (bte is None or bte.supports_expert_map()))

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
                f"Got bdge_war: {bdge_war}, and bte_war: {bte_war}")

        if bdge_war is not None:
            return bdge_war

        assert bte_war is not None
        return bte_war

    def workspace_shapes(
        self,
        a: torch.Tensor,
        aq: torch.Tensor,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_metadata: Optional[mk.ExpertTokensMetadata],
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], torch.dtype]:
        # Note: the deep gemm workspaces are strictly larger than the triton
        # workspaces so we can be pessimistic here and allocate for DeepGemm
        # even if we fall back to triton later, e.g. if expert maps are set.
        if self.allow_deep_gemm:
            assert self.batched_deep_gemm_experts is not None
            return self.batched_deep_gemm_experts.workspace_shapes(
                a, aq, M, N, K, topk, global_num_experts, local_num_experts,
                expert_tokens_metadata)
        else:
            assert self.batched_triton_experts is not None
            return self.batched_triton_experts.workspace_shapes(
                a, aq, M, N, K, topk, global_num_experts, local_num_experts,
                expert_tokens_metadata)

    def apply(self, output: torch.Tensor, hidden_states: torch.Tensor,
              w1: torch.Tensor, w2: torch.Tensor, topk_weights: torch.Tensor,
              topk_ids: torch.Tensor, activation: str, global_num_experts: int,
              expert_map: Optional[torch.Tensor],
              w1_scale: Optional[torch.Tensor],
              w2_scale: Optional[torch.Tensor], w1_zp: Optional[torch.Tensor],
              w2_zp: Optional[torch.Tensor], a1q_scale: Optional[torch.Tensor],
              a2_scale: Optional[torch.Tensor], workspace13: torch.Tensor,
              workspace2: torch.Tensor,
              expert_tokens_meta: Optional[mk.ExpertTokensMetadata],
              apply_router_weight_on_input: bool,
              extra_expert_args: Optional[dict[str, Any]]):
        experts = (self.batched_deep_gemm_experts
                   if self.allow_deep_gemm else self.batched_triton_experts)
        assert experts is not None
        experts.apply(output, hidden_states, w1, w2, topk_weights, topk_ids,
                      activation, global_num_experts, expert_map, w1_scale,
                      w2_scale, w1_zp, w2_zp, a1q_scale, a2_scale, workspace13,
                      workspace2, expert_tokens_meta,
                      apply_router_weight_on_input, extra_expert_args)
