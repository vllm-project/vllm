# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.deep_gemm_moe import (
    DeepGemmExperts, _valid_deep_gemm, _valid_deep_gemm_shape)
from vllm.model_executor.layers.fused_moe.fused_moe import TritonExperts


class TritonOrDeepGemmExperts(mk.FusedMoEPermuteExpertsUnpermute):

    def __init__(self,
                 use_fp8_w8a8: bool = False,
                 use_int8_w8a8: bool = False,
                 use_int8_w8a16: bool = False,
                 use_int4_w4a16: bool = False,
                 per_channel_quant: bool = False,
                 block_shape: Optional[list[int]] = None,
                 block_m: Optional[int] = None,
                 allow_deep_gemm: bool = False):
        super().__init__()
        self.triton_expert = TritonExperts(use_fp8_w8a8=use_fp8_w8a8,
                                           use_int8_w8a8=use_int8_w8a8,
                                           use_int4_w4a16=use_int4_w4a16,
                                           use_int8_w8a16=use_int8_w8a16,
                                           per_channel_quant=per_channel_quant,
                                           block_shape=block_shape,
                                           block_m=block_m)
        self.allow_deep_gemm = allow_deep_gemm
        self.use_fp8_w8a8 = use_fp8_w8a8
        self.deep_gemm_expert = DeepGemmExperts(
        ) if self.allow_deep_gemm else None

    def workspace_shapes(
        self,
        a: torch.Tensor,
        aq: torch.Tensor,
        M: int,
        N: int,
        K: int,
        topk: int,
        num_experts: int,
    ) -> tuple[int, int, torch.dtype]:
        # Note: the deep gemm workspaces are strictly larger than the triton
        # workspaces so we can be pessimistic here and allocate for DeepGemm
        # even if we fall back to triton later, e.g. if expert maps are set.
        if self.allow_deep_gemm and _valid_deep_gemm_shape(M, N, K):
            assert self.deep_gemm_expert is not None
            return self.deep_gemm_expert.workspace_shapes(
                a, aq, M, N, K, topk, num_experts)
        else:
            return self.triton_expert.workspace_shapes(a, aq, M, N, K, topk,
                                                       num_experts)

    def apply(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str,
        global_num_experts: int,
        expert_map: Optional[torch.Tensor],
        w1_scale: Optional[torch.Tensor],
        w2_scale: Optional[torch.Tensor],
        w1_zp: Optional[torch.Tensor],
        w2_zp: Optional[torch.Tensor],
        a1q_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_num_tokens: Optional[torch.Tensor],
    ) -> torch.Tensor:
        N = w1.size(1)
        if (self.allow_deep_gemm and self.use_fp8_w8a8 and N > 512
                and _valid_deep_gemm(hidden_states, w1, w2)):
            assert self.deep_gemm_expert is not None
            return self.deep_gemm_expert.apply(
                hidden_states,
                w1,
                w2,
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
                expert_num_tokens,
            )
        else:
            return self.triton_expert.apply(
                hidden_states,
                w1,
                w2,
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
                expert_num_tokens,
            )
