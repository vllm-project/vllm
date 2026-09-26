# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import pytest

from vllm.v1.executor import multiproc_executor
from vllm.v1.executor.multiproc_executor import MultiprocExecutor, WorkerProc


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


def test_collective_rpc_fails_cleanly_if_shutdown_during_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serialization_started = threading.Event()
    continue_serialization = threading.Event()

    def blocking_dumps(obj: Any, protocol: int) -> bytes:
        serialization_started.set()
        assert continue_serialization.wait(timeout=5)
        return b"serialized"

    monkeypatch.setattr(multiproc_executor.cloudpickle, "dumps", blocking_dumps)

    executor: Any = MultiprocExecutor.__new__(MultiprocExecutor)
    executor.rpc_broadcast_mq = SimpleNamespace(shutdown=lambda: None)
    executor.response_mqs = []
    executor.workers = []
    executor.is_failed = False

    with ThreadPoolExecutor(max_workers=1) as pool:
        rpc = pool.submit(executor.collective_rpc, lambda: None, non_block=True)
        assert serialization_started.wait(timeout=5)

        executor.is_failed = True
        executor.shutdown()
        assert executor.rpc_broadcast_mq is None
        continue_serialization.set()

        with pytest.raises(RuntimeError, match=r"Executor failed\."):
            rpc.result(timeout=5)
