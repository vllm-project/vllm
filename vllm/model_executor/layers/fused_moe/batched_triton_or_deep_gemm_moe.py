# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe import (
    BatchedDeepGemmExperts)
from vllm.model_executor.layers.fused_moe.fused_batched_moe import (
    BatchedTritonExperts)


class BatchedTritonOrDeepGemmExperts(mk.FusedMoEPermuteExpertsUnpermute):

    def __init__(self,
                 max_num_tokens: int,
                 world_size: int,
                 dp_size: int,
                 use_fp8_w8a8: bool = False,
                 use_int8_w8a8: bool = False,
                 use_int8_w8a16: bool = False,
                 use_int4_w4a16: bool = False,
                 per_channel_quant: bool = False,
                 block_shape: Optional[list[int]] = None,
                 allow_deep_gemm: bool = False):
        super().__init__()
        assert not use_int8_w8a8, "NYI"
        assert not use_int8_w8a16, "NYI"
        assert not use_int4_w4a16, "NYI"

        self.max_num_tokens = max_num_tokens
        self.world_size = world_size
        self.dp_size = dp_size
        self.use_fp8_w8a8 = use_fp8_w8a8
        self.use_int8_w8a8 = use_int8_w8a8
        self.use_int8_w8a16 = use_int8_w8a16
        self.use_int4_w4a16 = use_int4_w4a16
        self.per_channel_quant = per_channel_quant
        self.block_shape = block_shape
        self.allow_deep_gemm = allow_deep_gemm

        # BatchedTritonKernel doesn't support block quantization
        # at the moment.
        self.batched_triton_experts = BatchedTritonExperts(
            max_num_tokens=self.max_num_tokens,
            use_fp8_w8a8=self.use_fp8_w8a8,
            use_int8_w8a8=self.use_int8_w8a8,
            use_int8_w8a16=self.use_int8_w8a16,
            use_int4_w4a16=self.use_int4_w4a16,
            per_channel_quant=self.per_channel_quant,
            block_shape=self.block_shape,
            world_size=self.world_size,
            dp_size=self.dp_size) if self.block_shape is None else None

        is_fp8_128_block_quantized = (self.use_fp8_w8a8
                                      and self.block_shape is not None
                                      and len(self.block_shape) == 2 and all(
                                          [b == 128
                                           for b in self.block_shape]))
        self.batched_deep_gemm_experts = BatchedDeepGemmExperts(
            max_num_tokens=self.max_num_tokens,
            world_size=self.world_size,
            dp_size=self.dp_size,
            block_shape=self.block_shape,  # type: ignore[arg-type]
        ) if (self.allow_deep_gemm and is_fp8_128_block_quantized) else None

        assert (self.batched_deep_gemm_experts is not None
                or self.batched_triton_experts is not None)

    def supports_chunking(self) -> bool:
        bdge = self.batched_deep_gemm_experts
        bte = self.batched_triton_experts
        return ((bdge is None or bdge.supports_chunking())
                and (bte is None or bte.supports_chunking()))

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
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], torch.dtype]:
        # Note: the deep gemm workspaces are strictly larger than the triton
        # workspaces so we can be pessimistic here and allocate for DeepGemm
        # even if we fall back to triton later, e.g. if expert maps are set.
        if self.allow_deep_gemm and self.batched_deep_gemm_experts is not None:
            return self.batched_deep_gemm_experts.workspace_shapes(
                a, aq, M, N, K, topk, global_num_experts, local_num_experts)
        else:
            assert self.batched_triton_experts is not None
            return self.batched_triton_experts.workspace_shapes(
                a, aq, M, N, K, topk, global_num_experts, local_num_experts)

    def apply(
        self,
        output: torch.Tensor,
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
    ):
        use_batched_deep_gemm_experts = (self.allow_deep_gemm
                                         and self.batched_deep_gemm_experts
                                         is not None)
        experts = (self.batched_deep_gemm_experts
                   if use_batched_deep_gemm_experts else
                   self.batched_triton_experts)
        assert experts is not None
        experts.apply(output, hidden_states, w1, w2, topk_ids, activation,
                      global_num_experts, expert_map, w1_scale, w2_scale,
                      w1_zp, w2_zp, a1q_scale, a2_scale, workspace13,
                      workspace2, expert_num_tokens)
