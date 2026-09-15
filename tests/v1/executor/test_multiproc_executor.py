# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from vllm.v1.executor import multiproc_executor
from vllm.v1.executor.multiproc_executor import MultiprocExecutor, WorkerProc


def _monitor_executor(monkeypatch, cls=MultiprocExecutor):
    executor = cls.__new__(cls)
    executor.is_failed = False
    executor.failure_callback = None
    executor._failure_callback_lock = Lock()
    executor.shutdown = Mock()
    executor.workers = [
        SimpleNamespace(proc=SimpleNamespace(sentinel=1, name="worker", exitcode=1))
    ]
    monkeypatch.setattr(
        multiproc_executor.multiprocessing.connection, "wait", lambda _: [1]
    )
    return executor


@pytest.mark.parametrize("registration", ["before", "during_shutdown", "after"])
def test_worker_failure_notifies_registered_callback_once(monkeypatch, registration):
    executor = _monitor_executor(monkeypatch)

    def check_callback_runs_without_lock():
        assert executor._failure_callback_lock.acquire(blocking=False)
        executor._failure_callback_lock.release()

    callback = Mock(side_effect=check_callback_runs_without_lock)
    if registration == "before":
        executor.register_failure_callback(callback)
    elif registration == "during_shutdown":
        executor.shutdown.side_effect = lambda: executor.register_failure_callback(
            callback
        )

    executor.start_worker_monitor(inline=True)
    if registration == "after":
        executor.register_failure_callback(callback)

    callback.assert_called_once_with()
    executor.shutdown.assert_called_once_with()
    assert executor.is_failed
    assert executor.failure_callback is None


def test_worker_failure_during_callback_registration_is_not_lost(monkeypatch):
    """Worker death between the failure check and callback storage must notify."""
    registering = Event()
    resume_registration = Event()
    callback = Mock()
    lock = Lock()

    class PausedRegistrationExecutor(MultiprocExecutor):
        def __setattr__(self, name, value):
            if name == "failure_callback" and value is callback:
                registering.set()
                assert resume_registration.wait(5)
            super().__setattr__(name, value)

    class NotificationLock:
        def __enter__(self):
            if not lock.acquire(blocking=False):
                # The monitor reached the registration's critical section.
                resume_registration.set()
                assert lock.acquire(timeout=5)

        def __exit__(self, *exc):
            lock.release()

    executor = _monitor_executor(monkeypatch, PausedRegistrationExecutor)
    executor._failure_callback_lock = NotificationLock()

    with ThreadPoolExecutor(max_workers=1) as pool:
        registration = pool.submit(executor.register_failure_callback, callback)
        try:
            assert registering.wait(5)
            executor.start_worker_monitor(inline=True)
        finally:
            resume_registration.set()
        registration.result(timeout=5)

    callback.assert_called_once_with()
    assert executor.failure_callback is None


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
