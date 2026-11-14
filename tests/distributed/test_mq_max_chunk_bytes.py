# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test that VLLM_MQ_MAX_CHUNK_BYTES_MB bounds every shm ring buffer an executor
creates, not just the broadcast queue.

The bug: `multiproc_executor` read the env var for `rpc_broadcast_mq` but built
`worker_response_mq` with `MessageQueue(1, 1)`, falling back to the 24 MiB
signature default. That queue is created once PER WORKER, so at TP=2 it asks
for 2 x 24 MiB x 10 chunks = 480 MiB regardless of the env var, and the env var
cannot bring total usage under a small /dev/shm. Since #48879 added
`check_shm_free_space`, that is a hard boot failure rather than a latent SIGBUS.

The fix: pass the env-derived chunk size at every construction site.
"""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

import vllm.envs as envs
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.v1.executor import multiproc_executor


@pytest.mark.parametrize("chunk_mb", [1, 4])
def test_worker_response_mq_honours_env(
    monkeypatch: pytest.MonkeyPatch, chunk_mb: int
) -> None:
    """The per-worker response queue must use VLLM_MQ_MAX_CHUNK_BYTES_MB."""
    monkeypatch.setattr(envs, "VLLM_MQ_MAX_CHUNK_BYTES_MB", chunk_mb)

    captured: dict[str, Any] = {}

    class RecordingMessageQueue:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            captured.update(kwargs)

        @staticmethod
        def create_from_handle(handle: Any, rank: Any) -> Any:
            return MagicMock()

    monkeypatch.setattr(multiproc_executor, "MessageQueue", RecordingMessageQueue)

    worker_proc = SimpleNamespace(worker=SimpleNamespace(rank=0))
    vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(nnodes_within_dp=1))

    # Unbound call with a stand-in `self`: the queues are all we want to build.
    init_message_queues = cast(Any, multiproc_executor.WorkerProc._init_message_queues)
    init_message_queues(
        worker_proc, input_shm_handle=MagicMock(), vllm_config=vllm_config
    )

    assert captured["max_chunk_bytes"] == chunk_mb * 1024 * 1024


def test_ring_buffer_scales_with_chunk_bytes() -> None:
    """A smaller chunk size must actually shrink the shm allocation.

    Guards the assumption the fix relies on: that bounding max_chunk_bytes
    bounds what check_shm_free_space sees.
    """
    small = MessageQueue(1, 1, max_chunk_bytes=1024 * 1024)
    try:
        # 1 MiB x 10 chunks plus per-chunk metadata, far below the 240 MiB the
        # 24 MiB default would request.
        assert small.buffer.total_bytes_of_buffer < 16 * 1024 * 1024
        assert small.buffer.max_chunk_bytes == 1024 * 1024
    finally:
        del small
