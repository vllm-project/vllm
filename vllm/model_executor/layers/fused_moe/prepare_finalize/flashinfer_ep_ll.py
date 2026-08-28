# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer moe_ep low-latency prepare/finalize (EXPERT_MAJOR / BatchedExperts).

Maps directly onto vLLM's batched-experts contract: FlashInfer's LL EXPERT_MAJOR
dispatch returns `[num_local_experts, max_tokens_per_rank * world, hidden]`, where each
padded row is pre-assigned to one local expert (row // cap) — exactly the
`BatchedExperts` activation format `BatchedTritonExperts` consumes. combine reweights
per token on receive (using the topk_weights bound at handle-create), so finalize is a
plain combine.

Scope: bf16 activations (MVP). Quantized dispatch (fp8/nvfp4) is a follow-up.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
)

from .flashinfer_ep_common import FlashInferEPPrepareAndFinalizeBase

if TYPE_CHECKING:
    from flashinfer.moe_ep import Fleet


class FlashInferEPLLPrepareAndFinalize(FlashInferEPPrepareAndFinalizeBase):
    """Prepare/Finalize using FlashInfer moe_ep low-latency (EXPERT_MAJOR)."""

    def __init__(
        self,
        fleet: Fleet,
        max_tokens_per_rank: int,
        num_dispatchers: int,
        num_local_experts: int,
    ):
        super().__init__(fleet, num_dispatchers, num_local_experts)
        self.max_tokens_per_rank = max_tokens_per_rank

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.BatchedExperts

    def max_num_tokens_per_rank(self) -> int | None:
        return self.max_tokens_per_rank

    # NCCL-EP instantiates its low-latency kernels for a fixed set of hidden
    # sizes (contrib/nccl_ep/device/macros.cuh SWITCH_HIDDEN); anything else
    # trips EP_HOST_ASSERT(false and "Unsupported hidden") in low_latency.cu
    # and aborts the worker. Note this is NOT the same list DeepEP-LL uses --
    # NCCL-EP has no 3072 case.
    # NOTE: Keep this list sorted, maybe_roundup_layer_hidden_size depends on it.
    SUPPORTED_HIDDEN_SIZES = [2048, 2560, 4096, 5120, 6144, 7168, 8192]
    assert sorted(set(SUPPORTED_HIDDEN_SIZES)) == SUPPORTED_HIDDEN_SIZES

    @staticmethod
    def maybe_roundup_layer_hidden_size(hidden_size: int) -> int:
        # Round up to the closest supported hidden size.
        for x in FlashInferEPLLPrepareAndFinalize.SUPPORTED_HIDDEN_SIZES:
            if x >= hidden_size:
                return x
        raise ValueError(
            f"Hidden Size {hidden_size} is greater than the maximum supported "
            f"hidden size "
            f"{FlashInferEPLLPrepareAndFinalize.SUPPORTED_HIDDEN_SIZES[-1]}"
        )

    def _assert_bf16(self, quant_config: FusedMoEQuantConfig) -> None:
        if quant_config is not None and quant_config.quant_dtype is not None:
            raise NotImplementedError(
                "FlashInferEPLLPrepareAndFinalize currently supports bf16 dispatch "
                f"only; got quant_dtype={quant_config.quant_dtype!r}."
            )

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
        self._assert_bf16(quant_config)

        if apply_router_weight_on_input:
            assert topk_ids.size(1) == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        from flashinfer.moe_ep import DispatchInputParams

        combine_weights = self._combine_weights(
            topk_weights, apply_router_weight_on_input
        )
        handle = self._make_handle(topk_ids, combine_weights)
        d = handle.dispatch(DispatchInputParams(x=[a1]))

        # d.expert_tensors: [num_local_experts, max_tokens_per_rank*world, hidden].
        expert_x = d.expert_tensors
        # d.expert_counts: per-local-expert received counts (int32, device) written by
        # the library — surface as ExpertTokensMetadata so the batched experts kernel
        # can skip padded rows.
        expert_tokens_meta = None
        if d.expert_counts is not None:
            expert_tokens_meta = mk.ExpertTokensMetadata(
                expert_num_tokens=d.expert_counts, expert_num_tokens_cpu=None
            )

        # EXPERT_MAJOR rows are pre-routed by position, so no per-token routing is
        # returned (combine reweights via the bound weights). bf16 → no scales.
        return expert_x, None, expert_tokens_meta, None, None

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        assert isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate), (
            "FlashInfer moe_ep LL applies weights + reduces inside combine."
        )
        from flashinfer.moe_ep import CombineInputParams

        handle = self._pop_handle()
        try:
            # fused_expert_output: [num_local_experts, cap, hidden] (batched). combine
            # gathers back to origin ranks, reweighting per token on receive, into
            # `output` in place.
            handle.combine(CombineInputParams(x=[fused_expert_output], out=output))
        finally:
            # LL non-staged self-drains, but call complete() to honor the handle
            # lifecycle uniformly (no-op in non-staged mode).
            handle.complete()
