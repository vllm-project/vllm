# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
from unittest.mock import patch

from vllm.v1.worker import ubatching
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
        forward_contexts=[object(), object()],
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
        forward_contexts=[object(), object()],
        ready_barrier=ready_barrier,
        token_offsets=[0, 7],
    )
    assert [ctx.token_offset for ctx in ctxs] == [0, 7]

    offsets_seen: dict[int, int] = {}

    def run(ctx: ubatching.UBatchContext, expected: int) -> None:
        # Keep torch.cuda.set_stream (no CUDA device here) out of the way.
        with (
            patch.object(ubatching, "current_stream", return_value=compute_stream),
            ctx,
        ):
            offsets_seen[ctx.id] = dbo_current_ubatch_token_offset()
        assert offsets_seen[ctx.id] == expected

    threads = [
        threading.Thread(target=run, args=(ctxs[0], 0)),
        threading.Thread(target=run, args=(ctxs[1], 7)),
    ]
    for thread in threads:
        thread.start()
    ready_barrier.wait()
    ctxs[0].cpu_wait_event.set()
    for thread in threads:
        thread.join()

    assert offsets_seen == {0: 0, 1: 7}
