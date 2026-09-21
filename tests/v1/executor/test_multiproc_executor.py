# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing
import os
import threading
import weakref
from collections import deque
from types import SimpleNamespace
from typing import Any

import pytest
import zmq

from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.v1.executor.multiproc_executor import MultiprocExecutor, WorkerProc
from vllm.v1.outputs import DraftTokenIds


def _exit_between_queue_handshakes(input_handle, pipe):
    input_mq = MessageQueue.create_from_handle(input_handle, 0)
    response_mq = MessageQueue(1, 1, max_chunk_bytes=1024, max_chunks=1)
    pipe.send(response_mq.export_handle())
    pipe.recv()  # Let the parent attach before exiting and releasing shared memory.
    input_mq.wait_until_ready()
    pipe.recv()  # Inject failure once the parent starts waiting for response READY.
    os._exit(0)


def test_worker_exit_cancels_response_queue_startup():
    """The real worker monitor must release a pending response MQ handshake."""
    context = multiprocessing.get_context("spawn")
    input_mq = MessageQueue(1, 1, max_chunk_bytes=1024, max_chunks=1)
    parent, child = context.Pipe()
    proc = context.Process(
        target=_exit_between_queue_handshakes,
        args=(input_mq.export_handle(), child),
    )
    response_mq = None
    waiter = None
    monitor = None
    proc.start()
    child.close()
    try:
        assert parent.poll(30), "Worker did not publish its response handle"
        response_mq = MessageQueue.create_from_handle(parent.recv(), 0)
        response_mq.local_socket.setsockopt(zmq.RCVTIMEO, 5000)
        executor: Any = MultiprocExecutor.__new__(MultiprocExecutor)
        executor.workers = [
            SimpleNamespace(
                proc=proc, death_writer=None, worker_response_mq=response_mq
            )
        ]
        executor.rpc_broadcast_mq = input_mq
        executor.response_mqs = [response_mq]
        executor.is_failed = False
        executor.failure_callback = None
        monitor = threading.Thread(
            target=executor.start_worker_monitor, kwargs={"inline": True}
        )
        monitor.start()
        parent.send("attached")
        input_mq.wait_until_ready()
        entered = threading.Event()
        finished = threading.Event()
        errors = []

        def wait():
            entered.set()
            try:
                response_mq.wait_until_ready()
            except Exception as exc:
                errors.append((type(exc), str(exc)))
            finally:
                finished.set()

        waiter = threading.Thread(target=wait)
        waiter.start()
        assert entered.wait(5)
        assert not finished.wait(0.05)
        parent.send("exit")
        assert finished.wait(3), "Worker death did not cancel startup"
        assert executor.is_failed
        assert response_mq.shutting_down
        assert errors == [(RuntimeError, "cancelled")]
    finally:
        if proc.is_alive():
            proc.kill()
        proc.join(5)
        if monitor is not None:
            monitor.join(5)
            assert not monitor.is_alive()
        if waiter is not None:
            waiter.join(6)
            assert not waiter.is_alive()
        parent.close()
        input_mq.local_socket.context.destroy(linger=0)
        if response_mq is not None:
            response_mq.local_socket.context.destroy(linger=0)
            # The fault-injected worker could not unlink its own ring buffer.
            response_mq.buffer.shared_memory.unlink()


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
