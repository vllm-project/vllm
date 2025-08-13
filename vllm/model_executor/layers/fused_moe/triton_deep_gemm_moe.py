# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, Optional

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.deep_gemm_moe import (
    DeepGemmExperts, _valid_deep_gemm, _valid_deep_gemm_shape,
    deep_gemm_block_shape)
from vllm.model_executor.layers.fused_moe.fused_moe import TritonExperts
from vllm.utils.deep_gemm import is_blackwell_deep_gemm_e8m0_used


class TritonOrDeepGemmExperts(mk.FusedMoEPermuteExpertsUnpermute):

    def __init__(
        self,
        use_fp8_w8a8: bool = False,
        use_int8_w8a8: bool = False,
        use_int8_w8a16: bool = False,
        use_int4_w4a16: bool = False,
        use_mxfp4_w4a4: bool = False,
        per_act_token_quant: bool = False,
        block_shape: Optional[list[int]] = None,
        allow_deep_gemm: bool = False,
    ):
        super().__init__(
            FusedMoEQuantConfig.make(
                use_fp8_w8a8=use_fp8_w8a8,
                use_int8_w8a8=use_int8_w8a8,
                use_int8_w8a16=use_int8_w8a16,
                use_int4_w4a16=use_int4_w4a16,
                use_mxfp4_w4a4=use_mxfp4_w4a4,
                per_act_token_quant=per_act_token_quant,
                block_shape=block_shape,
            ))
        self.triton_expert = TritonExperts(
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int4_w4a16=use_int4_w4a16,
            use_int8_w8a16=use_int8_w8a16,
            use_mxfp4_w4a4=use_mxfp4_w4a4,
            per_act_token_quant=per_act_token_quant,
            block_shape=block_shape,
        )

        self.allow_deep_gemm = (allow_deep_gemm and use_fp8_w8a8 and
                                self.block_shape == deep_gemm_block_shape())

        self.deep_gemm_expert = DeepGemmExperts(
        ) if self.allow_deep_gemm else None

    @property
    def activation_formats(
        self
    ) -> tuple[mk.FusedMoEActivationFormat, mk.FusedMoEActivationFormat]:
        assert (self.deep_gemm_expert is None
                or self.triton_expert.activation_formats
                == self.deep_gemm_expert.activation_formats)
        return self.triton_expert.activation_formats

    def supports_chunking(self) -> bool:
        dge = self.deep_gemm_expert
        te = self.triton_expert
        return ((dge is None or dge.supports_chunking())
                and (te is None or te.supports_chunking()))

    def supports_expert_map(self) -> bool:
        dge = self.deep_gemm_expert
        te = self.triton_expert
        return ((dge is None or dge.supports_expert_map())
                and (te is None or te.supports_expert_map()))

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
                f"Got dge_war: {dge_war}, and te_war: {te_war}")

        if dge_war is not None:
            return dge_war

        assert te_war is not None
        return te_war

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
        expert_tokens_meta: Optional[mk.ExpertTokensMetadata],
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], torch.dtype]:
        # Note: the deep gemm workspaces are strictly larger than the triton
        # workspaces so we can be pessimistic here and allocate for DeepGemm
        # even if we fall back to triton later, e.g. if expert maps are set.
        if self.allow_deep_gemm and (is_blackwell_deep_gemm_e8m0_used()
                                     or _valid_deep_gemm_shape(M, N, K)):
            assert self.deep_gemm_expert is not None
            return self.deep_gemm_expert.workspace_shapes(
                a, aq, M, N, K, topk, global_num_experts, local_num_experts,
                expert_tokens_meta)
        else:
            return self.triton_expert.workspace_shapes(a, aq, M, N, K, topk,
                                                       global_num_experts,
                                                       local_num_experts,
                                                       expert_tokens_meta)

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
        use_deep_gemm = (self.allow_deep_gemm
                         and (_valid_deep_gemm(hidden_states, w1, w2)
                              or is_blackwell_deep_gemm_e8m0_used()))

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
            w1_scale,
            w2_scale,
            w1_zp,
            w2_zp,
            a1q_scale,
            a2_scale,
            workspace13,
            workspace2,
            expert_tokens_meta,
            apply_router_weight_on_input,
            extra_expert_args,
        )
