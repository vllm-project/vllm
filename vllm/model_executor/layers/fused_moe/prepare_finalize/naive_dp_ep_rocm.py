# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dual-batch-overlap (DBO) aware prepare/finalize for the naive DP/EP path.

The naive path runs its DP all-gather (dispatch) and reduce-scatter (combine)
synchronously on the calling stream. Under DBO that leaves the compute stream
idle for the whole collective while the peer ubatch could be computing. This
subclass hands each collective to the shared comm stream, using the same
compute/comm handoff the DeepEP-HT prepare/finalize uses.
"""

from collections.abc import Iterator
from contextlib import contextmanager

import torch

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

    def supports_dbo(self) -> bool:
        # prepare()/finalize() hand the collective to the comm stream
        # themselves, so the synchronous path is DBO-safe. supports_async()
        # stays False: it also selects MK-internal shared-expert overlap.
        return True
