# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Aiter-based expert processing for Mori integration.
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
    rocm_aiter_fused_experts,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)


class AiterMoriExperts(mk.FusedMoEPermuteExpertsUnpermute):
    """
    Aiter-based expert processing that works with Mori dispatch/combine.

    This class bridges Mori's all2all communication with Aiter's optimized
    expert computation kernels for AMD GPUs.
    """

    def __init__(
        self,
        max_num_tokens: int,
        quant_config: FusedMoEQuantConfig,
    ):
        from vllm.platforms.rocm import on_mi3xx

        if not on_mi3xx():
            raise RuntimeError("AiterMoriExperts should be used on AMD mi3xx GPUs")

        super().__init__(
            quant_config=quant_config,
        )
        self.max_num_tokens = max_num_tokens

    @property
    def activation_formats(
        self,
    ) -> tuple[mk.FusedMoEActivationFormat, mk.FusedMoEActivationFormat]:
        """Aiter expects Standard format for both input and output."""
        return (
            mk.FusedMoEActivationFormat.Standard,
            mk.FusedMoEActivationFormat.Standard,
        )

    def supports_chunking(self) -> bool:
        """Aiter kernels support chunking."""
        return True

    def supports_expert_map(self) -> bool:
        """Aiter kernels support expert mapping."""
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        """Aiter handles weight and reduce internally."""
        return TopKWeightAndReduceNoOP()

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
        """
        Aiter kernels manage memory internally, so minimal workspace is needed.
        """
        workspace1 = (M, K)
        workspace2 = (0,)  # No intermediate workspace needed
        output_shape = (M, K)
        return (workspace1, workspace2, output_shape)

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
        """
        Process expert computation using Aiter kernels.
        Works with pre-dispatched tokens from Mori all2all.
        """
        if expert_tokens_meta is not None:
            expert_num_tokens = expert_tokens_meta.expert_num_tokens
        else:
            expert_num_tokens = None

        # Call Aiter fused MoE expert processing
        result = rocm_aiter_fused_experts(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            expert_map=expert_map,
            expert_num_tokens=expert_num_tokens,
            output_dtype=output.dtype,
            quant_config=self.quant_config,
            a1q_scale=a1q_scale,
        )

        # Copy result to output tensor
        output.copy_(result)
