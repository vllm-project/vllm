# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared plumbing for the FlashInfer `moe_ep` (NCCL-EP) prepare/finalize adapters.

Both the low-latency (BatchedExperts) and high-throughput (Standard) adapters drive
the same `flashinfer.moe_ep` sequence per forward:

    handle = fleet.create_handle(HandleParams(topk_ids), knobs)
    d = handle.dispatch(DispatchInputParams(x=[a1]))          # prepare()
    ... vLLM runs the expert GEMM on d.expert_tensors ...
    handle.combine(CombineInputParams(x=[expert_out], out))   # finalize()
    handle.complete()

The `Fleet` (durable transport sizing + comm) is owned by the all2all *manager*
(`FlashInferEP*All2AllManager`) and passed in; the per-forward `Handle` (bound to this
step's `topk_ids`) is created in `prepare()` and consumed in `finalize()`. Handles are
stashed per micro-batch id so dual-batch-overlap (DBO) works like the DeepEP adapters.
"""

from __future__ import annotations

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.v1.worker.ubatching import dbo_current_ubatch_id


class FlashInferEPPrepareAndFinalizeBase(mk.FusedMoEPrepareAndFinalizeModular):
    """Common base for the FlashInfer moe_ep LL / HT prepare-finalize adapters."""

    def __init__(self, fleet, num_dispatchers: int, num_local_experts: int):
        super().__init__()
        self._fleet = fleet
        self._num_dispatchers = num_dispatchers
        self._num_local_experts = num_local_experts
        # Per-ubatch handle slots (DBO uses up to 2 concurrent micro-batches).
        self._handles: list = [None, None]

    # ---- static capability declarations shared by both adapters -------------

    def num_dispatchers(self) -> int:
        return self._num_dispatchers

    def output_is_reduced(self) -> bool:
        # combine() gathers each token's expert output back to its origin rank and
        # sums across the ranks that held it — the finalize output is reduced.
        return True

    def topk_indices_dtype(self) -> torch.dtype | None:
        # nccl.ep v0.1 binds topk_idx as int64 at create_handle.
        return torch.int64

    # ---- shared handle lifecycle -------------------------------------------

    def _make_handle(self, topk_ids, combine_weights, extra_knobs=()):
        """Create a per-forward FlashInfer Handle bound to this step's routing.

        `combine_weights` is bound as the reweight applied on receive during
        combine (LL EXPERT_MAJOR) / forward dispatch (HT). Runs on the current
        CUDA stream so it composes with the surrounding vLLM graph.
        """
        from flashinfer.moe_ep import (
            HandleAlgoKnobTopKWeights,
            HandleAlgoKnobUserStream,
            HandleParams,
        )

        knobs = [
            HandleAlgoKnobUserStream(stream=torch.cuda.current_stream().cuda_stream),
            HandleAlgoKnobTopKWeights(weights=combine_weights),
            *extra_knobs,
        ]
        if topk_ids.dtype != torch.int64:
            topk_ids = topk_ids.to(torch.int64)
        handle = self._fleet.create_handle(
            HandleParams(topk_ids=topk_ids), algo_knobs=knobs
        )
        self._handles[dbo_current_ubatch_id()] = handle
        return handle

    def _pop_handle(self):
        idx = dbo_current_ubatch_id()
        handle = self._handles[idx]
        assert handle is not None, "finalize() called before prepare() on this ubatch"
        self._handles[idx] = None
        return handle

    @staticmethod
    def _combine_weights(topk_weights, apply_router_weight_on_input):
        # When the router weight was already applied to the input activations,
        # combine must not re-apply it — bind ones instead.
        if apply_router_weight_on_input:
            return torch.ones_like(topk_weights)
        return topk_weights
