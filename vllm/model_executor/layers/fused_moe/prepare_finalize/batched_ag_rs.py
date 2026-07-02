# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import get_ep_group
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.prepare_finalize.batched import (
    bucket_tokens_to_batched,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
    TopKWeightAndReduceNaiveBatched,
)


class BatchedAgRsPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """All-gather / reduce-scatter EP dispatch in the batched activation format.

    Bridges the AllGather+ReduceScatter all-to-all with the
    `[E_local, max_num_tokens, K]` batched format consumed by
    `BatchedTritonExperts`. The full batch is replicated on every rank.
    """

    def __init__(
        self,
        max_num_tokens: int,
        num_local_experts: int,
        num_dispatchers: int,
        rank: int,
        is_sequence_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.max_num_tokens = max_num_tokens
        self.num_local_experts = num_local_experts
        self._num_dispatchers = num_dispatchers
        self.rank = rank
        self.is_sequence_parallel = is_sequence_parallel

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    def max_num_tokens_per_rank(self) -> int | None:
        return self.max_num_tokens

    def topk_indices_dtype(self) -> torch.dtype | None:
        return None

    def num_dispatchers(self) -> int:
        return self._num_dispatchers

    def output_is_reduced(self) -> bool:
        return False

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        if quant_config.quant_dtype is not None or defer_input_quant:
            raise NotImplementedError(
                f"{self.__class__.__name__} supports unquantized MoE only."
            )

        if apply_router_weight_on_input:
            assert topk_ids.size(1) == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        # All-gather the global token batch + routing across the group.
        a1g, topk_weights_g, topk_ids_g = get_ep_group().dispatch(
            a1,
            topk_weights,
            topk_ids,
            is_sequence_parallel=self.is_sequence_parallel,
        )

        b_a1, tokens_per_expert = bucket_tokens_to_batched(
            a1g,
            topk_ids_g,
            num_experts,
            self.num_local_experts,
            self.rank,
            self.max_num_tokens,
            a1g.dtype,
        )
        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=tokens_per_expert, expert_num_tokens_cpu=None
        )

        # Return the gathered (global) routing so finalize scatters over the
        # full token set before the reduce-scatter combine.
        return b_a1, None, expert_tokens_meta, topk_ids_g, topk_weights_g

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
            weight_and_reduce_impl = TopKWeightAndReduceNaiveBatched(self.rank)

        # Reduce the batched expert output to per-(global)token rows.
        out = weight_and_reduce_impl.apply(
            output=None,
            fused_expert_output=fused_expert_output,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )

        # Reduce-scatter back to this rank's local token slice.
        output.copy_(
            get_ep_group().combine(out, is_sequence_parallel=self.is_sequence_parallel)
        )
