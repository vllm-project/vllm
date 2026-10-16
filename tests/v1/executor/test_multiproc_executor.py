# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref
from collections import deque
from types import SimpleNamespace
from typing import Any

import pytest

from vllm.v1.executor import multiproc_executor
from vllm.v1.executor.multiproc_executor import MultiprocExecutor, WorkerProc
from vllm.v1.outputs import DraftTokenIds


class _ExitWorkerLoop(RuntimeError):
    pass


class _RpcPayload:
    pass


class _PayloadLifetimeCheckingQueue:
    def __init__(self) -> None:
        self.payload_ref: weakref.ReferenceType[_RpcPayload] | None = None
        self.dequeue_count = 0

    def dequeue(self, *, indefinite: bool):
        assert indefinite
        self.dequeue_count += 1
        if self.dequeue_count == 1:
            payload = _RpcPayload()
            self.payload_ref = weakref.ref(payload)
            return "consume", (payload,), {}, None

        assert self.payload_ref is not None
        assert self.payload_ref() is None
        raise _ExitWorkerLoop


def test_worker_rpc_payload_released_before_next_dequeue():
    queue = _PayloadLifetimeCheckingQueue()
    worker_proc: Any = WorkerProc.__new__(WorkerProc)
    worker_proc.rpc_broadcast_mq = queue
    worker_proc.rank = 0
    worker_proc.worker = SimpleNamespace(consume=lambda payload: payload)
    worker_proc.handle_output = lambda output: None

    with pytest.raises(_ExitWorkerLoop):
        worker_proc.worker_busy_loop()

    assert queue.dequeue_count == 2


def test_execute_worker_rpc_returns_worker_exception():
    def fail():
        raise RuntimeError("test error")

    worker_proc: Any = WorkerProc.__new__(WorkerProc)
    worker_proc.rank = 0
    worker_proc.worker = SimpleNamespace(fail=fail)
    outputs: list[Any] = []
    worker_proc.handle_output = outputs.append

    worker_proc._execute_worker_rpc(("fail", (), {}, None))

    assert len(outputs) == 1
    assert isinstance(outputs[0], RuntimeError)
    assert str(outputs[0]) == "test error"


@pytest.mark.parametrize("stalled", [False, True])
@pytest.mark.parametrize(
    "method,result",
    [
        ("take_draft_token_ids", DraftTokenIds(["req"], [[1, 2, 3]])),
        ("execute_dummy_batch", None),
    ],
)
def test_model_rpc_uses_execute_model_timeout(monkeypatch, method, result, stalled):
    """A wedged model RPC must not block EngineCore forever. For
    execute_dummy_batch this is an idle DP rank stuck in a collective with a
    hung peer."""
    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "7")

    def dequeue(*, timeout=None):
        assert timeout is not None and 0 < timeout <= 7
        if stalled:
            raise TimeoutError
        return WorkerProc.ResponseStatus.SUCCESS, result

    executor: Any = MultiprocExecutor.__new__(MultiprocExecutor)
    executor.is_failed = False
    executor.output_rank = 1
    executor.futures_queue = deque()
    executor.rpc_broadcast_mq = SimpleNamespace(enqueue=lambda payload: None)
    executor.response_mqs = [None, SimpleNamespace(dequeue=dequeue)]

    if stalled:
        with pytest.raises(TimeoutError, match=f"{method} timed out"):
            getattr(executor, method)()
    else:
        assert getattr(executor, method)() is result


class _ReplyQueue:
    def __init__(self, replies: list) -> None:
        self.replies = deque(replies)

    def dequeue(self, timeout=None):
        return self.replies.popleft()


def test_collective_rpc_drains_every_rank_on_failure():
    """A failure on one rank used to raise before the other ranks' replies
    were read, so the next call returned the previous call's replies."""
    ok, failure = WorkerProc.ResponseStatus.SUCCESS, WorkerProc.ResponseStatus.FAILURE
    executor = object.__new__(MultiprocExecutor)
    executor.rpc_broadcast_mq = SimpleNamespace(enqueue=lambda msg: None)
    executor.is_failed = False
    executor.futures_queue = deque()
    executor.response_mqs = [
        _ReplyQueue([(failure, "rank 0 failed"), (ok, "second-0")]),
        _ReplyQueue([(failure, "rank 1 failed"), (ok, "second-1")]),
    ]
    with pytest.raises(RuntimeError, match="rank 0 failed"):
        executor.collective_rpc("first")
    assert executor.collective_rpc("second") == ["second-0", "second-1"]


class _BlockedQueue:
    """A rank stuck in a collective: it never replies."""

    def __init__(self) -> None:
        self.timeouts: list = []

    def dequeue(self, timeout=None):
        self.timeouts.append(timeout)
        assert timeout is not None, "would block forever"
        raise TimeoutError


def test_collective_rpc_failure_drain_does_not_wait_forever():
    """Draining after a failure must not hang on a rank that never replies,
    e.g. one blocked in a collective the failed rank never joined."""
    failure = WorkerProc.ResponseStatus.FAILURE
    blocked = _BlockedQueue()
    executor = object.__new__(MultiprocExecutor)
    executor.rpc_broadcast_mq = SimpleNamespace(enqueue=lambda msg: None)
    executor.is_failed = False
    executor.futures_queue = deque()
    executor.response_mqs = [_ReplyQueue([(failure, "rank 0 failed")]), blocked]
    with pytest.raises(RuntimeError, match="rank 0 failed"):
        executor.collective_rpc("first")
    assert blocked.timeouts[0] <= multiproc_executor._FAILURE_DRAIN_TIMEOUT_S
    # The reply queues are out of step, so later calls fail fast.
    assert executor.is_failed
    with pytest.raises(RuntimeError, match="Executor failed"):
        executor.collective_rpc("second")
