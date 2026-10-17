# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref
from collections import deque
from types import SimpleNamespace
from typing import Any

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


@pytest.mark.parametrize(
    "method", ["execute_model", "sample_tokens", "execute_dummy_batch"]
)
def test_non_output_model_step_rpc_exception_stops_worker(method):
    def fail():
        raise RuntimeError("non-output rank failure")

    worker_proc: Any = WorkerProc.__new__(WorkerProc)
    worker_proc.rank = 0
    worker_proc.worker = SimpleNamespace(**{method: fail})
    worker_proc.handle_output = lambda output: pytest.fail(
        "non-output workers must not publish a successful response"
    )

    with pytest.raises(RuntimeError, match="non-output rank failure"):
        worker_proc._execute_worker_rpc((method, (), {}, 1))


def test_non_output_non_model_rpc_exception_does_not_stop_worker():
    def fail():
        raise RuntimeError("non-model RPC failure")

    worker_proc: Any = WorkerProc.__new__(WorkerProc)
    worker_proc.rank = 0
    worker_proc.worker = SimpleNamespace(take_draft_token_ids=fail)
    worker_proc.handle_output = lambda output: pytest.fail(
        "non-output workers must not publish a response"
    )

    worker_proc._execute_worker_rpc(("take_draft_token_ids", (), {}, 1))


def test_model_step_rpc_exception_prevents_next_dequeue():
    marker_executed = False
    dequeue_count = 0

    def fail():
        raise RuntimeError("non-output rank failure")

    def marker():
        nonlocal marker_executed
        marker_executed = True

    def dequeue(*, indefinite):
        nonlocal dequeue_count
        assert indefinite
        dequeue_count += 1
        if dequeue_count == 1:
            return "execute_model", (), {}, 1
        if dequeue_count == 2:
            return "marker", (), {}, None
        raise _ExitWorkerLoop

    worker_proc: Any = WorkerProc.__new__(WorkerProc)
    worker_proc.rpc_broadcast_mq = SimpleNamespace(dequeue=dequeue)
    worker_proc.rank = 0
    worker_proc.worker = SimpleNamespace(execute_model=fail, marker=marker)
    worker_proc.handle_output = lambda output: None

    with pytest.raises(RuntimeError, match="non-output rank failure"):
        worker_proc.worker_busy_loop()

    assert not marker_executed
    assert dequeue_count == 1


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
