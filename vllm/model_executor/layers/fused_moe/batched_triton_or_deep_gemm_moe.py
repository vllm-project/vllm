# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe import (
    BatchedDeepGemmExperts,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.fused_batched_moe import BatchedTritonExperts
from vllm.utils.deep_gemm import get_mk_alignment_for_contiguous_layout


class BatchedTritonOrDeepGemmExperts(mk.FusedMoEPermuteExpertsUnpermute):
    def __init__(
        self,
        max_num_tokens: int,
        num_dispatchers: int,
        quant_config: FusedMoEQuantConfig,
        use_deep_gemm: bool = False,
    ):
        super().__init__(quant_config)

        if use_deep_gemm:
            # Validate DeepGEMM requirements upfront
            if not self.quant_config.use_fp8_w8a8:
                raise ValueError(
                    "DeepGEMM requires FP8 W8A8 quantization, but "
                    f"quant_config.use_fp8_w8a8={self.quant_config.use_fp8_w8a8}"
                )
            expected_block_shape = get_mk_alignment_for_contiguous_layout()
            if self.block_shape != expected_block_shape:
                raise ValueError(
                    "DeepGEMM requires block_shape to match contiguous layout "
                    f"alignment. Got block_shape={self.block_shape}, "
                    f"expected {expected_block_shape}"
                )
            self.experts: BatchedDeepGemmExperts | BatchedTritonExperts = (
                BatchedDeepGemmExperts(
                    max_num_tokens=max_num_tokens,
                    num_dispatchers=num_dispatchers,
                    quant_config=self.quant_config,
                )
            )
        else:
            self.experts = BatchedTritonExperts(
                max_num_tokens=max_num_tokens,
                num_dispatchers=num_dispatchers,
                quant_config=self.quant_config,
            )

    @property
    def activation_formats(
        self,
    ) -> tuple[mk.FusedMoEActivationFormat, mk.FusedMoEActivationFormat]:
        return self.experts.activation_formats

    def supports_chunking(self) -> bool:
        return self.experts.supports_chunking()

    def supports_expert_map(self) -> bool:
        return self.experts.supports_expert_map()

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return self.experts.finalize_weight_and_reduce_impl()

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
        return self.experts.workspace_shapes(
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
        self.experts.apply(
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
