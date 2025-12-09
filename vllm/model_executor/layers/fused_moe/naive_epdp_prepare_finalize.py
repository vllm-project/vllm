# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Prepare/Finalize implementation for EP+DP without a specialized all2all backend.

This uses the generic dispatch/combine from the EP group for token redistribution
across DP ranks, with local routing on each rank.
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import get_ep_group
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous,
    TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input


class NaiveEPDPPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    """
    Prepare/Finalize for EP+DP using naive dispatch/combine.

    This implementation uses the generic EP group dispatch/combine operations
    instead of specialized all2all kernels. The dispatch happens at the
    layer level (before routing), so prepare() only handles quantization.
    The finalize() method performs the combine operation and weight/reduce.

    Note: This is less efficient than specialized all2all kernels like PPLX
    or DeepEP, but provides a fallback implementation when those are not available.
    """

    def __init__(self, is_sequence_parallel: bool, ep_size: int):
        super().__init__()
        self.is_sequence_parallel = is_sequence_parallel
        self.ep_size_ = ep_size

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        return None

    def num_dispatchers(self) -> int:
        return self.ep_size_

    def output_is_reduced(self) -> bool:
        # combine() reduces across EP ranks
        return True

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
    ) -> mk.PrepareResultType:
        """
        Quantize inputs for MoE computation.

        Note: Dispatch across DP ranks has already happened at the layer level
        (before routing), so this method only handles quantization.
        """
        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            # TODO: this only works for topK=1, will need to update for topK>1
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1.mul_(topk_weights.to(a1.dtype))

        a1q, a1q_scale = moe_kernel_quantize_input(
            a1,
            quant_config.a1_scale,
            quant_config.quant_dtype,
            quant_config.per_act_token_quant,
            quant_config.block_shape,
        )

        return a1q, a1q_scale, None, None, None

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        """
        Combine expert outputs across DP ranks and apply weights/reduce.
        """
        combined_output = get_ep_group().combine(
            fused_expert_output, self.is_sequence_parallel
        )

        if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
            weight_and_reduce_impl = TopKWeightAndReduceContiguous()

        weight_and_reduce_impl.apply(
            output=output,
            fused_expert_output=combined_output,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
