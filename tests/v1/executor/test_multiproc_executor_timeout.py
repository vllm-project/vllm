# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque

from vllm.v1.executor.multiproc_executor import MultiprocExecutor, WorkerProc


def test_collective_rpc_clamps_expired_timeout(monkeypatch):
    """Ensure stale RPC deadlines cannot become indefinite MQ waits."""

    class MockBroadcastMQ:
        def enqueue(self, obj):
            pass

    class MockResponseMQ:
        def __init__(self):
            self.timeouts = []

        def dequeue(self, timeout=None):
            self.timeouts.append(timeout)
            return WorkerProc.ResponseStatus.SUCCESS, "ok"

    executor = MultiprocExecutor.__new__(MultiprocExecutor)
    executor.rpc_broadcast_mq = MockBroadcastMQ()
    executor.response_mqs = [MockResponseMQ()]
    executor.futures_queue = deque()
    executor.is_failed = False

    monotonic_values = iter([100.0, 102.0])
    monkeypatch.setattr(
        "vllm.v1.executor.multiproc_executor.time.monotonic",
        lambda: next(monotonic_values),
    )

    result = executor.collective_rpc("test_method", timeout=1, unique_reply_rank=0)

    assert result == "ok"
    assert executor.response_mqs[0].timeouts == [0.0]
