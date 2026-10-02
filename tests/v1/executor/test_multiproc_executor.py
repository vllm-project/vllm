# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref
from collections import deque
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

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


def test_collective_rpc_drains_every_rank_before_raising():
    """A failing rank must not leave the other ranks' responses queued, or they
    would be read as the answers of the next RPC (such as the wake-up that
    recovers a failed sleep)."""
    executor: Any = MultiprocExecutor.__new__(MultiprocExecutor)
    executor.rpc_broadcast_mq = SimpleNamespace(enqueue=lambda msg: None)
    executor.is_failed = False
    executor.futures_queue = deque()
    failure = (WorkerProc.ResponseStatus.FAILURE, "boom")
    executor.response_mqs = [
        Mock(
            dequeue=Mock(side_effect=[failure, (WorkerProc.ResponseStatus.SUCCESS, r)])
        )
        for r in ("r0", "r1")
    ]

    with pytest.raises(RuntimeError, match="boom"):
        executor.collective_rpc("sleep")

    assert executor.collective_rpc("wake_up") == ["r0", "r1"]


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
def test_take_draft_token_ids_uses_execute_model_timeout(monkeypatch, stalled):
    """A wedged draft-token readback must not block EngineCore forever."""
    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "7")
    draft = DraftTokenIds(["req"], [[1, 2, 3]])

    def dequeue(*, timeout=None):
        assert timeout is not None and 0 < timeout <= 7
        if stalled:
            raise TimeoutError
        return WorkerProc.ResponseStatus.SUCCESS, draft

    executor: Any = MultiprocExecutor.__new__(MultiprocExecutor)
    executor.is_failed = False
    executor.output_rank = 1
    executor.futures_queue = deque()
    executor.rpc_broadcast_mq = SimpleNamespace(enqueue=lambda payload: None)
    executor.response_mqs = [None, SimpleNamespace(dequeue=dequeue)]

    if stalled:
        with pytest.raises(TimeoutError, match="take_draft_token_ids timed out"):
            executor.take_draft_token_ids()
    else:
        assert executor.take_draft_token_ids() is draft
