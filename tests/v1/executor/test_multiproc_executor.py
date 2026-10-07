# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref
from collections import deque
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from vllm.config import ParallelConfig
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


@pytest.mark.parametrize("non_block", [False, True])
@pytest.mark.parametrize("fault_tolerance", [False, True])
def test_failed_rpc_does_not_leave_stale_responses(fault_tolerance, non_block):
    executor = MultiprocExecutor.__new__(MultiprocExecutor)
    executor.is_failed = False
    executor.parallel_config = ParallelConfig(enable_fault_tolerance=fault_tolerance)
    executor.futures_queue = deque()
    executor.rpc_broadcast_mq = Mock()
    success = WorkerProc.ResponseStatus.SUCCESS
    failure = WorkerProc.ResponseStatus.FAILURE
    executor.response_mqs = [
        Mock(
            dequeue=Mock(side_effect=[(failure, "device error"), (success, "retry 0")])
        ),
        Mock(dequeue=Mock(side_effect=[(failure, "peer error"), (success, "retry 1")])),
    ]

    with pytest.raises(RuntimeError, match="device error") as exc:
        result = executor.collective_rpc("execute_model", non_block=non_block)
        if non_block:
            result.result()

    if fault_tolerance:
        assert executor.collective_rpc("handle_ft_command") == ["retry 0", "retry 1"]
        assert "rank 0" in str(exc.value) and "rank 1" in str(exc.value)
        assert "peer error" in str(exc.value)
    else:
        executor.response_mqs[1].dequeue.assert_not_called()


def test_failed_rpc_drain_respects_deadline():
    executor = MultiprocExecutor.__new__(MultiprocExecutor)
    executor.is_failed = False
    executor.parallel_config = ParallelConfig(enable_fault_tolerance=True)
    executor.futures_queue = deque()
    executor.rpc_broadcast_mq = Mock()
    executor.response_mqs = [
        Mock(
            dequeue=Mock(
                return_value=(WorkerProc.ResponseStatus.FAILURE, "device error")
            )
        ),
        Mock(dequeue=Mock(side_effect=TimeoutError)),
    ]

    with pytest.raises(TimeoutError, match="RPC call to execute_model timed out"):
        executor.collective_rpc("execute_model", timeout=1)

    timeout = executor.response_mqs[1].dequeue.call_args.kwargs["timeout"]
    assert 0 <= timeout <= 1


def test_failed_executor_does_not_drain_dead_worker():
    executor = MultiprocExecutor.__new__(MultiprocExecutor)
    executor.is_failed = False
    executor.parallel_config = ParallelConfig(enable_fault_tolerance=True)
    executor.futures_queue = deque()
    executor.rpc_broadcast_mq = Mock()

    def fail(**kwargs):
        executor.is_failed = True
        return WorkerProc.ResponseStatus.FAILURE, "worker exited"

    executor.response_mqs = [Mock(dequeue=Mock(side_effect=fail)), Mock()]
    with pytest.raises(RuntimeError, match="worker exited"):
        executor.collective_rpc("execute_model")
    executor.response_mqs[1].dequeue.assert_not_called()
