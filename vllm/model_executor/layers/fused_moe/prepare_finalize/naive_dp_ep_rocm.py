# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dual-batch-overlap (DBO) aware prepare/finalize for the naive DP/EP path.

The naive path runs its DP all-gather (dispatch) and reduce-scatter (combine)
synchronously on the calling stream. Under DBO that leaves the compute stream
idle for the whole collective while the peer ubatch could be computing. These
subclasses hand each collective to the shared comm stream, using the same
compute/comm handoff the DeepEP-HT prepare/finalize uses.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch

from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import (
    MoEPrepareAndFinalizeNaiveDPEPModular,
)
from vllm.v1.worker.ubatching import (
    dbo_enabled,
    dbo_switch_to_compute_sync,
    dbo_yield_and_switch_from_compute_to_comm,
)


def _comm_overlap_active() -> bool:
    # Graph capture must keep every collective on the capture stream: moving
    # one to the comm stream changes the topology of the captured decode
    # graph. Overlap is therefore limited to eager steps (prefill).
    return dbo_enabled() and not torch.cuda.is_current_stream_capturing()


class MoEPrepareAndFinalizeNaiveDPEPModularROCmDBO(
    MoEPrepareAndFinalizeNaiveDPEPModular
):
    """Modular naive DP/EP prepare/finalize with ROCm DBO comm overlap."""

    @contextmanager
    def _comm_region(self) -> Iterator[None]:
        if not _comm_overlap_active():
            yield
            return
        # Release the CPU to the peer ubatch so it can enqueue compute while
        # this ubatch's collective runs; the compute stream waits on exit.
        dbo_yield_and_switch_from_compute_to_comm()
        try:
            yield
        finally:
            dbo_switch_to_compute_sync()

    def supports_async(self) -> bool:
        # The modular kernel treats "not async" as "no DBO" and asserts out of
        # DBO (FusedMoEKernelModularImpl._prepare/_finalize). Report async
        # only while a DBO context is active so the non-DBO path keeps its
        # exact current behavior; the DBO handoff itself happens inside
        # prepare()/finalize().
        return dbo_enabled()

    def prepare_async(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: mk.FusedMoEQuantConfig,
        defer_input_quant: bool,
    ) -> mk.ReceiverType:
        result = self.prepare(
            a1,
            topk_weights,
            topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_config,
            defer_input_quant,
        )
        return lambda: result

    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> Callable[[], None]:
        self.finalize(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
        )
        return lambda: None
