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


@pytest.mark.parametrize(
    "inherited_fds, local_world_size, is_cpu, numa_bind, expected",
    [
        (None, 8, False, False, True),  # spawn, GPU: start concurrently
        ([], 8, False, False, False),  # fork carries fds between workers
        (None, 1, False, False, False),  # nothing to overlap
        (None, 8, True, False, False),  # CPU backend sets OpenMP envs per worker
        (None, 8, False, True, False),  # NUMA wrapper patches the executable
    ],
)
def test_can_start_workers_concurrently(
    monkeypatch, inherited_fds, local_world_size, is_cpu, numa_bind, expected
):
    import vllm.v1.executor.multiproc_executor as mp_executor

    monkeypatch.setattr(
        mp_executor, "current_platform", SimpleNamespace(is_cpu=lambda: is_cpu)
    )
    executor: Any = SimpleNamespace(
        local_world_size=local_world_size,
        parallel_config=SimpleNamespace(numa_bind=numa_bind),
    )

    assert (
        MultiprocExecutor._can_start_workers_concurrently(executor, inherited_fds)
        is expected
    )


def test_start_workers_concurrently_keeps_started_workers_on_failure():
    """Workers that started must reach cleanup, in rank order, if one fails."""
    executor: Any = SimpleNamespace(local_world_size=4)
    handles = {rank: object() for rank in (0, 1, 3)}

    def start_worker(local_rank):
        if local_rank == 2:
            raise RuntimeError("boom")
        return handles[local_rank]

    unready_workers: list[Any] = []
    with pytest.raises(RuntimeError, match="boom"):
        MultiprocExecutor._start_workers_concurrently(
            executor, start_worker, unready_workers
        )

    assert unready_workers == [handles[0], handles[1], handles[3]]
