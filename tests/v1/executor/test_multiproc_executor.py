# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing
import threading
import time
import weakref
from collections import deque
from types import SimpleNamespace
from typing import Any

import pytest

from vllm.v1.executor.multiproc_executor import (
    MultiprocExecutor,
    _PeerMonitor,
    WorkerProc,
    WorkerProcHandle,
)


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


class _BlockingBroadcastMQ:
    def enqueue(self, _message: object) -> None:
        pass


class _BlockingResponseMQ:
    def dequeue(self, timeout: float | None = None):
        time.sleep(min(timeout or 0.05, 0.05))
        raise TimeoutError("response did not arrive")


class _DeathWriter:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def _blocked_worker(stop: Any) -> None:
    stop.wait()


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


def test_peer_failure_interrupts_rpc_and_aborts_local_worker():
    context = multiprocessing.get_context("spawn")
    stop = context.Event()
    local_worker = context.Process(
        target=_blocked_worker, args=(stop,), name="local-worker"
    )
    local_worker.start()

    class FakeExecutor:
        pass

    def make_executor(node_rank: int):
        executor = FakeExecutor()
        executor.parallel_config = SimpleNamespace(
            node_rank_within_dp=node_rank,
            nnodes_within_dp=2,
            master_addr="127.0.0.1",
            master_port=0,
        )
        executor.workers = []
        return executor

    executor = MultiprocExecutor.__new__(MultiprocExecutor)
    executor.is_failed = False
    executor.workers_aborted = False
    executor.peer_failure_event = threading.Event()
    executor.peer_failure_info = None
    executor.failure_callback = None
    executor.parallel_config = make_executor(0).parallel_config
    death_writer = _DeathWriter()
    executor.workers = [
        WorkerProcHandle(
            proc=local_worker,
            rank=0,
            worker_response_mq=None,
            peer_worker_response_mqs=[],
            death_writer=death_writer,
        )
    ]
    executor.rpc_broadcast_mq = _BlockingBroadcastMQ()
    executor.response_mqs = [_BlockingResponseMQ()]
    executor.futures_queue = deque()

    follower_executor = make_executor(1)
    leader = _PeerMonitor(executor)
    follower = _PeerMonitor(follower_executor)
    leader.port = 0
    leader.start()
    # Resolve the ephemeral port before starting the client.
    deadline = time.monotonic() + 2
    while leader.server is None or leader.server.getsockname()[1] == 0:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    follower.port = leader.server.getsockname()[1]
    follower.start()
    rpc_errors: list[BaseException] = []
    rpc_thread = threading.Thread(
        target=lambda: _run_rpc(executor, rpc_errors), daemon=True
    )
    rpc_thread.start()
    try:
        deadline = time.monotonic() + 2
        while not follower.connections:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        follower.shutdown()
        deadline = time.monotonic() + 2
        while not executor.peer_failure_event.is_set():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        rpc_thread.join(timeout=2)
        assert not rpc_thread.is_alive()
        assert len(rpc_errors) == 1
        assert "Peer failure interrupted RPC" in str(rpc_errors[0])
        assert executor.is_failed
        assert executor.workers_aborted
        assert death_writer.closed
        local_worker.join(timeout=2)
        assert not local_worker.is_alive()
    finally:
        if rpc_thread.is_alive():
            rpc_thread.join(timeout=2)
        if local_worker.is_alive():
            local_worker.kill()
        local_worker.join(timeout=2)
        stop.set()
        follower.shutdown()
        leader.shutdown()


def _run_rpc(executor: MultiprocExecutor, errors: list[BaseException]) -> None:
    try:
        executor.collective_rpc("simulated_rpc", timeout=10)
    except BaseException as error:
        errors.append(error)
