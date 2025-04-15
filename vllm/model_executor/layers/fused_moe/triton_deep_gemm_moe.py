# SPDX-License-Identifier: Apache-2.0
import importlib.util
from typing import List, Optional, Tuple

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.deep_gemm_moe import (
    DeepGemmExperts,
    _valid_deep_gemm_shape,
    _valid_deep_gemm,
)
from vllm.model_executor.layers.fused_moe.fused_moe import TritonExpert

class TritonOrDeepGemmExperts(mk.FusedMoEPermuteExpertsUnpermute):

    def __init__(
        self,
        use_fp8_w8a8: bool,
        use_int8_w8a16: bool,
        use_int4_w4a16: bool,
        block_shape: Optional[List[int]] = None,
        block_m: Optional[int] = None,
        allow_deep_gemm: bool = False
    ):
        super().__init__()
        self.triton_expert = TritonExpert(
            use_fp8_w8a8,
            use_int4_w4a16,
            use_int8_w8a16,
            block_shape,
            block_m
        )
        self.deep_gemm_expert = DeepGemmExperts()
        self.allow_deep_gemm = allow_deep_gemm
        self.use_fp8_w8a8 = use_fp8_w8a8

    def workspace_shapes(
        self,
        a_dtype: torch.dtype,
        M: int,
        N: int,
        K: int,
        topk: int,
        num_experts: int,
        a: torch.Tensor,
    ) -> Tuple[int, int, torch.dtype]:
        # Note: the deep gemm workspaces are strictly larger than the triton
        # workspaces so we can be pessimistic here and allocate for DeepGemm
        # even if we fall back to triton later, e.g. if expert maps are set.
        if self.allow_deep_gemm and _valid_deep_gemm_shape(M, N, K):
            return self.deep_gemm_expert.workspace_shapes(a_dtype, M, N, K, topk, num_experts, a)
        else:
            return self.triton_expert.workspace_shapes(a_dtype, M, N, K, topk, num_experts, a)

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
    ) -> torch.Tensor:
        N = w1.shape[1]
        if (self.allow_deep_gemm and self.use_fp8_w8a8 and N > 512
            and _valid_deep_gemm(hidden_states, w1, w2, expert_map)):
            return self.deep_gemm_expert(
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
            )
        else:
            return self.triton_expert(
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
            )
