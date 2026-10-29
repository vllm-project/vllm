# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
from unittest.mock import patch

import numpy as np

from vllm.v1.worker import ubatching
from vllm.v1.worker.ubatch_utils import maybe_create_ubatch_slices
from vllm.v1.worker.ubatching import (
    dbo_current_ubatch_token_offset,
    make_ubatch_contexts,
)


def test_token_offset_without_dbo_context():
    assert dbo_current_ubatch_token_offset() == 0


def test_token_offset_defaults_to_zero():
    ready_barrier = threading.Barrier(3)
    ctxs = make_ubatch_contexts(
        num_micro_batches=2,
        compute_stream=object(),
        comm_stream=object(),
        forward_contexts=[object(), object()],  # type: ignore[list-item]  # stand-ins
        ready_barrier=ready_barrier,
    )
    assert [ctx.token_offset for ctx in ctxs] == [0, 0]


def test_token_offset_per_ubatch_thread():
    compute_stream = object()
    ready_barrier = threading.Barrier(3)
    ctxs = make_ubatch_contexts(
        num_micro_batches=2,
        compute_stream=compute_stream,
        comm_stream=object(),
        forward_contexts=[object(), object()],  # type: ignore[list-item]  # stand-ins
        ready_barrier=ready_barrier,
        token_offsets=[0, 7],
    )
    assert [ctx.token_offset for ctx in ctxs] == [0, 7]

    offsets_seen: dict[int, int] = {}

    def run(ctx: ubatching.UBatchContext) -> None:
        with ctx:
            offsets_seen[ctx.id] = dbo_current_ubatch_token_offset()

    # Keep torch.cuda.set_stream out of the way. Patch once here rather than
    # per thread: a per-thread patch exit would restore the real function
    # while the other thread is still inside its context.
    with patch.object(ubatching, "current_stream", return_value=compute_stream):
        threads = [threading.Thread(target=run, args=(ctx,)) for ctx in ctxs]
        for thread in threads:
            thread.start()
        ready_barrier.wait()
        ctxs[0].cpu_wait_event.set()
        for thread in threads:
            thread.join()

    assert offsets_seen == {0: 0, 1: 7}


def test_ubatch_slice_bounds_are_python_ints():
    # The last bound comes from np.cumsum; numpy scalars leaking into the
    # per-ubatch metadata break Triton argument specialization.
    ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
        should_ubatch=True,
        num_scheduled_tokens=np.array([5, 9, 6], dtype=np.int32),
        num_tokens_padded=20,
        num_reqs_padded=3,
        num_ubatches=2,
    )
    for slices in (ubatch_slices, ubatch_slices_padded):
        assert slices is not None
        for s in slices:
            for bound in (s.token_slice.start, s.token_slice.stop):
                assert type(bound) is int
