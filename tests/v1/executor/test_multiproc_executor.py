# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref
from collections import deque
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.multiproc_executor import MultiprocExecutor, WorkerProc
from vllm.v1.outputs import DraftTokenIds


def _make_executor(*responses: list[Any]) -> Any:
    executor: Any = MultiprocExecutor.__new__(MultiprocExecutor)
    executor.is_failed = False
    executor.failure_callback = None
    executor.sleeping_tags = set()
    executor.rpc_broadcast_mq = Mock()
    executor.futures_queue = deque()
    executor.response_mqs = [
        Mock(dequeue=Mock(side_effect=replies)) for replies in responses
    ]
    return executor


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
    executor.failure_callback = None
    executor.output_rank = 1
    executor.futures_queue = deque()
    executor.rpc_broadcast_mq = SimpleNamespace(enqueue=lambda payload: None)
    executor.response_mqs = [None, SimpleNamespace(dequeue=dequeue)]

    if stalled:
        with pytest.raises(TimeoutError, match="take_draft_token_ids timed out"):
            executor.take_draft_token_ids()
    else:
        assert executor.take_draft_token_ids() is draft


@pytest.mark.cpu_test
@pytest.mark.parametrize("non_block", [False, True])
@pytest.mark.parametrize("failed_ranks", [(0,), (1,), (0, 1)])
def test_failed_collective_rpc_preserves_next_call_responses(
    non_block: bool, failed_ranks: tuple[int, ...]
) -> None:
    """Drain replies and preserve the first error in response-queue order."""
    success = WorkerProc.ResponseStatus.SUCCESS
    failure = WorkerProc.ResponseStatus.FAILURE
    executor = _make_executor(
        *[
            [
                (failure, f"rank {rank} failed")
                if rank in failed_ranks
                else (success, None),
                (success, f"next-rank-{rank}"),
            ]
            for rank in range(2)
        ]
    )

    message = f"rank {failed_ranks[0]} failed"
    expected = ["next-rank-0", "next-rank-1"]
    if non_block:
        failed = executor.collective_rpc("first", non_block=True)
        following = executor.collective_rpc("next", non_block=True)
        assert following.result() == expected
        with pytest.raises(RuntimeError, match=message):
            failed.result()
    else:
        with pytest.raises(RuntimeError, match=message):
            executor.collective_rpc("first")
        assert executor.collective_rpc("next") == expected


@pytest.mark.cpu_test
def test_failed_sleep_drains_before_wake_and_scheduler_resume() -> None:
    """Connect Core, Executor and RPC collection; resume only after wake succeeds."""
    success = WorkerProc.ResponseStatus.SUCCESS
    failure = WorkerProc.ResponseStatus.FAILURE
    executor = _make_executor(
        [(failure, "sleep failed"), (success, None), (success, None)],
        [(success, None), (failure, "wake failed"), (success, None)],
    )
    core = EngineCore.__new__(EngineCore)
    core.model_executor = executor
    core.pause_scheduler = Mock(return_value=None)
    core.resume_scheduler = Mock()

    with pytest.raises(RuntimeError, match="sleep failed"):
        core.sleep(level=1)
    core.pause_scheduler.assert_called_once_with(mode="abort", clear_cache=True)
    assert executor.is_sleeping
    core.resume_scheduler.assert_not_called()

    with pytest.raises(RuntimeError, match="wake failed"):
        core.wake_up()
    assert executor.is_sleeping
    core.resume_scheduler.assert_not_called()

    assert core.wake_up() is True
    assert not executor.is_sleeping
    assert [
        call.args[0][0] for call in executor.rpc_broadcast_mq.enqueue.call_args_list
    ] == ["sleep", "wake_up", "wake_up"]
    core.resume_scheduler.assert_called_once_with()


@pytest.mark.cpu_test
@pytest.mark.parametrize("timeout", [None, 8.0, 30.0])
def test_worker_error_drain_shares_existing_rpc_budget(
    monkeypatch: pytest.MonkeyPatch, timeout: float | None
) -> None:
    """Use the configured worker budget without extending a shorter caller deadline."""
    monkeypatch.setenv("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "6")
    clock = SimpleNamespace(now=10.0)
    monkeypatch.setattr(
        "vllm.v1.executor.multiproc_executor.time",
        SimpleNamespace(monotonic=lambda: clock.now),
    )
    success = WorkerProc.ResponseStatus.SUCCESS
    failure = WorkerProc.ResponseStatus.FAILURE
    executor = _make_executor([], [], [])
    drain_remaining = 4.0 if timeout == 8.0 else 6.0

    def first_reply(timeout: float | None) -> tuple:
        clock.now = 14.0
        return failure, "first worker error"

    def second_reply(timeout: float | None) -> tuple:
        assert timeout == drain_remaining
        clock.now += 2.0
        return failure, "second worker error"

    def third_reply(timeout: float | None) -> tuple:
        assert timeout == drain_remaining - 2.0
        return success, None

    for queue, reply in zip(
        executor.response_mqs, (first_reply, second_reply, third_reply), strict=True
    ):
        queue.dequeue.side_effect = reply
    with pytest.raises(RuntimeError, match="first worker error"):
        executor.collective_rpc("sleep", timeout=timeout)
    assert not executor.is_failed


@pytest.mark.cpu_test
@pytest.mark.parametrize("worker_error_first", [False, True])
@pytest.mark.parametrize("error_type", [TimeoutError, EOFError])
def test_rpc_receive_error_fails_pending_futures_without_reading(
    worker_error_first: bool, error_type: type[Exception]
) -> None:
    """An incomplete receive must stop queued and new RPCs from using stale replies."""
    success = WorkerProc.ResponseStatus.SUCCESS
    failure = WorkerProc.ResponseStatus.FAILURE
    receive_error = error_type("response receive failed")
    executor = _make_executor(
        [(failure, "worker error") if worker_error_first else (success, None)],
        [receive_error],
        [(success, "stale reply")],
    )

    def on_failure() -> None:
        with pytest.raises(RuntimeError, match="Executor failed"):
            executor.collective_rpc("reentrant")

    callback = Mock(side_effect=on_failure)
    executor.failure_callback = callback
    first = executor.collective_rpc("first", timeout=1.0, non_block=True)
    cancelled = executor.collective_rpc("cancelled", non_block=True)
    following = executor.collective_rpc("following", non_block=True)
    assert cancelled.cancel()

    with pytest.raises(RuntimeError, match="Executor failed"):
        following.result()
    with pytest.raises(error_type) as raised:
        first.result()
    if error_type is TimeoutError:
        assert str(raised.value) == "RPC call to first timed out."
        assert raised.value.__cause__ is receive_error
    else:
        assert raised.value is receive_error
    assert executor.is_failed
    assert cancelled.cancelled()
    assert not executor.futures_queue
    callback.assert_called_once_with()
    for queue in executor.response_mqs[:2]:
        assert queue.dequeue.call_count == 1
    executor.response_mqs[2].dequeue.assert_not_called()
    assert executor.rpc_broadcast_mq.enqueue.call_count == 3

    executor.rpc_broadcast_mq = None
    with pytest.raises(RuntimeError, match="Executor failed"):
        executor.collective_rpc("new")


@pytest.mark.cpu_test
def test_pending_rpc_checks_failure_before_receiving() -> None:
    executor = _make_executor([])
    pending = executor.collective_rpc("pending", non_block=True)
    executor.is_failed = True
    with pytest.raises(RuntimeError, match="Executor failed"):
        pending.result()
    executor.response_mqs[0].dequeue.assert_not_called()
