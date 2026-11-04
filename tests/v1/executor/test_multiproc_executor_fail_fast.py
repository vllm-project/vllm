# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for failing fast when a worker hits a fatal RPC failure."""

import multiprocessing
import queue
import threading
import time
import weakref
from collections import deque
from types import SimpleNamespace
from typing import Any

import pytest
import zmq

from vllm.config import DeviceConfig, VllmConfig
from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.v1.executor.multiproc_executor import (
    FutureWrapper,
    MultiprocExecutor,
    WorkerProc,
)

pytestmark = pytest.mark.timeout(60)

# Large enough that the reply is still being sent when the worker exits.
_OUTPUT_SIZE = 16 << 20


def _make_failing_worker_proc(rank: int, method: str) -> Any:
    def fail(*args, **kwargs):
        raise RuntimeError(f"{method} failed")

    worker_proc: Any = WorkerProc.__new__(WorkerProc)
    worker_proc.rank = rank
    worker_proc.worker = SimpleNamespace(**{method: fail})
    return worker_proc


@pytest.mark.parametrize(
    ("method", "exit_on_fatal_failure"),
    [("add_lora", True), ("execute_model", False)],
    ids=["recoverable_rpc", "fault_tolerance"],
)
def test_rpc_failure_is_only_returned_to_caller(
    monkeypatch, method, exit_on_fatal_failure
):
    """A failed utility RPC, e.g. loading an invalid LoRA adapter, must not
    take the worker down, and neither may any failure with fault tolerance."""
    monkeypatch.setattr(
        WorkerProc, "_exit_after_fatal_failure", lambda self: pytest.fail("exited")
    )
    worker_proc = _make_failing_worker_proc(0, method)
    worker_proc.exit_on_fatal_failure = exit_on_fatal_failure
    outputs: list[Any] = []
    worker_proc.handle_output = outputs.append

    worker_proc._execute_worker_rpc((method, (), {}, None))

    assert len(outputs) == 1
    assert isinstance(outputs[0], RuntimeError)


def _run_failing_worker(conn, method, output_rank, async_scheduling) -> None:
    # A remote reader, so that replies which aren't flushed before the exit are
    # lost rather than left in shared memory.
    response_mq: Any = MessageQueue(
        n_reader=1, n_local_reader=0, connect_ip="127.0.0.1"
    )
    conn.send(response_mq.export_handle())
    response_mq.wait_until_ready()

    worker_proc = _make_failing_worker_proc(1, method)
    worker_proc.exit_on_fatal_failure = True
    worker_proc.worker_response_mq = response_mq
    worker_proc.use_async_scheduling = async_scheduling
    if async_scheduling:
        enqueue = response_mq.enqueue

        def slow_enqueue(obj):
            time.sleep(0.2)
            enqueue(obj)

        # A slow output thread, which the worker must wait for before exiting.
        response_mq.enqueue = slow_enqueue
        worker_proc.async_output_queue = queue.Queue()
        threading.Thread(target=worker_proc.async_output_busy_loop, daemon=True).start()

    # The reply to an earlier RPC.
    worker_proc.handle_output(bytes(_OUTPUT_SIZE))
    worker_proc._execute_worker_rpc((method, (), {}, output_rank))


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize(
    ("method", "output_rank"),
    [("compile_or_warm_up_model", None), ("execute_model", 1), ("execute_model", 0)],
)
def test_fatal_rpc_failure_exits_worker(method, output_rank, async_scheduling):
    """A fatal RPC failure must take the worker down, even on a rank that
    doesn't reply, since its peers may be blocked in a collective with it.
    The worker's pending replies, including the error, must still reach the
    executor."""
    ctx = multiprocessing.get_context("fork")
    conn, child_conn = ctx.Pipe()
    proc = ctx.Process(
        target=_run_failing_worker,
        args=(child_conn, method, output_rank, async_scheduling),
        name="VllmWorker-1",
    )
    proc.start()
    try:
        assert conn.poll(30), "worker did not start"
        response_mq = MessageQueue.create_from_handle(conn.recv(), 0)
        response_mq.wait_until_ready()
        proc.join(timeout=30)
        assert proc.exitcode == 1

        status, output = response_mq.dequeue(timeout=5)
        assert status == WorkerProc.ResponseStatus.SUCCESS
        assert len(output) == _OUTPUT_SIZE
        if output_rank in (None, 1):
            assert response_mq.dequeue(timeout=5) == (
                WorkerProc.ResponseStatus.FAILURE,
                f"{method} failed",
            )
        else:
            with pytest.raises(TimeoutError):
                response_mq.dequeue(timeout=0.5)
    finally:
        if proc.is_alive():
            proc.kill()
        proc.join()
        conn.close()
        child_conn.close()


def _wait_for_parent(death_reader, ready):
    ready.set()
    try:
        death_reader.recv()
    except EOFError:
        pass
    finally:
        death_reader.close()


def _run_remote_publisher(conn):
    writer = MessageQueue(1, 0, connect_ip="127.0.0.1")
    writer.remote_socket.setsockopt(zmq.RCVTIMEO, 30000)
    try:
        conn.send(writer.export_handle())
        writer.wait_until_ready()
        conn.send("ready")
        command = conn.recv()
        if command == "fatal_rpc":
            worker = _make_failing_worker_proc(1, "execute_model")
            worker.exit_on_fatal_failure = True
            worker.use_async_scheduling = False
            worker.worker_response_mq = writer
            worker._execute_worker_rpc(("execute_model", (), {}, 0))
        elif command == "disconnect":
            writer.remote_socket.close(linger=0)
            conn.send("disconnected")
            conn.recv()
    except EOFError:
        pass
    finally:
        writer.remote_socket.context.destroy(linger=0)
        conn.close()


def _run_monitored_worker(handle, death_reader, ready, blocked):
    reader = MessageQueue.create_from_handle(handle, -1)
    reader.remote_socket.setsockopt(zmq.RCVTIMEO, 30000)
    worker: Any = WorkerProc.__new__(WorkerProc)
    worker.rank = 1
    worker.rpc_broadcast_mq = reader
    worker.worker_response_mq = None
    shutdown_requested = threading.Event()
    worker.monitor_death_pipe(death_reader, shutdown_requested)
    reader.wait_until_ready()
    ready.set()
    # Model code can be stuck in a collective and cannot service shutdown.
    if blocked:
        assert threading.Event().wait(30), "worker was not terminated"
    else:
        assert shutdown_requested.wait(30), "local parent EOF was not handled"
    death_reader.close()


def _stop_process(proc):
    proc.join(5)
    if proc.is_alive():
        proc.kill()
        proc.join(5)
    assert not proc.is_alive()
    proc.close()


def _worker_handle(proc, death_writer):
    return SimpleNamespace(
        proc=proc, death_writer=death_writer, worker_response_mq=None
    )


def _make_executor(workers, response_mqs, failed):
    executor: Any = MultiprocExecutor.__new__(MultiprocExecutor)
    executor.vllm_config = VllmConfig(device_config=DeviceConfig("cpu"))
    executor.is_failed = False
    executor.failure_callback = failed.set
    executor.workers = workers
    executor.response_mqs = response_mqs
    return executor


@pytest.fixture
def monitor_threads(monkeypatch):
    threads = []

    def make_thread(**kwargs):
        thread = threading.Thread(**kwargs)
        threads.append(thread)
        return thread

    monkeypatch.setattr("vllm.v1.executor.multiproc_executor.Thread", make_thread)
    yield threads
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive(), "executor monitor leaked after cleanup"


def _run_remote_monitor_collection(conn, handle):
    reader = MessageQueue.create_from_handle(handle, -1)
    reader.remote_socket.setsockopt(zmq.RCVTIMEO, 30000)
    reader_ref = weakref.ref(reader)
    failed = threading.Event()
    executor = _make_executor([], [reader], failed)
    executor.start_worker_monitor()
    monitor = next(
        thread
        for thread in threading.enumerate()
        if thread.name == "MultiprocWorkerMonitor"
    )
    reader.wait_until_ready()
    del reader
    conn.send("ready")
    monitor.join(10)
    conn.send(
        {
            "monitor_stopped": not monitor.is_alive(),
            "failed": failed.is_set(),
            "reader_collected": reader_ref() is None,
        }
    )
    conn.close()


def test_remote_monitor_releases_last_queue_reference_without_deadlock():
    """Queue collection must not wait on a monitor socket owned by itself."""
    ctx = multiprocessing.get_context("spawn")
    writer = MessageQueue(1, 0, connect_ip="127.0.0.1")
    writer.remote_socket.setsockopt(zmq.RCVTIMEO, 30000)
    conn, child_conn = ctx.Pipe()
    child = ctx.Process(
        target=_run_remote_monitor_collection,
        args=(child_conn, writer.export_handle()),
    )
    child.start()
    child_conn.close()
    try:
        writer.wait_until_ready()
        assert conn.poll(30) and conn.recv() == "ready"
        writer.remote_socket.close(linger=0)
        assert conn.poll(15), "remote disconnect did not complete monitor teardown"
        assert conn.recv() == {
            "monitor_stopped": True,
            "failed": True,
            "reader_collected": True,
        }
        child.join(5)
        assert child.exitcode == 0
    finally:
        conn.close()
        _stop_process(child)
        writer.remote_socket.context.destroy(linger=0)


@pytest.mark.parametrize("loss", ["fatal_rpc", "socket", "normal_shutdown"])
def test_remote_worker_loss_fails_executor_and_cancels_response(loss, monitor_threads):
    """Remote loss must fail the executor; normal shutdown must not.
    Both must cancel pending reads and clean up the local sibling."""
    ctx = multiprocessing.get_context("spawn")
    conn, publisher_conn = ctx.Pipe()
    publisher = ctx.Process(target=_run_remote_publisher, args=(publisher_conn,))
    death_reader, death_writer = ctx.Pipe(duplex=False)
    ready = ctx.Event()
    local_worker = ctx.Process(target=_wait_for_parent, args=(death_reader, ready))
    publisher.start()
    publisher_conn.close()
    local_worker.start()
    death_reader.close()
    executor = reader = receiver = None
    try:
        assert ready.wait(30), "local worker did not start"
        assert conn.poll(30), "remote publisher did not start"
        reader = MessageQueue.create_from_handle(conn.recv(), -1)
        reader.remote_socket.setsockopt(zmq.RCVTIMEO, 30000)
        failed = threading.Event()
        executor = _make_executor(
            [_worker_handle(local_worker, death_writer)], [reader], failed
        )
        executor.start_worker_monitor()
        reader.wait_until_ready()
        assert conn.poll(30) and conn.recv() == "ready"
        started = threading.Event()
        errors = []
        # A bounded fallback also releases the receiver on an unfixed parent.
        future = FutureWrapper(deque(), lambda: reader.dequeue(timeout=15))

        def receive():
            started.set()
            try:
                future.result()
            except Exception as error:
                errors.append(error)

        receiver = threading.Thread(target=receive)
        receiver.start()
        assert started.wait(5)
        assert local_worker.is_alive()
        if loss == "fatal_rpc":
            conn.send("fatal_rpc")
            publisher.join(5)
            assert publisher.exitcode == 1
        elif loss == "socket":
            conn.send("disconnect")
            assert conn.poll(5) and conn.recv() == "disconnected"
            assert publisher.is_alive(), "socket loss must not require process exit"
        else:
            executor.shutdown()

        if loss != "normal_shutdown":
            assert failed.wait(5), "remote worker loss did not reach the engine"
        receiver.join(5)
        local_worker.join(5)
        for thread in monitor_threads:
            thread.join(5)
            assert not thread.is_alive()
        assert executor.is_failed == (loss != "normal_shutdown")
        assert failed.is_set() == (loss != "normal_shutdown")
        assert local_worker.exitcode == 0, "local sibling was not shut down gracefully"
        assert not receiver.is_alive(), "pending response survived executor shutdown"
        assert len(errors) == 1 and isinstance(errors[0], RuntimeError)
    finally:
        if executor is not None:
            executor.shutdown()
        death_writer.close()
        if receiver is not None:
            receiver.join(20)
            assert not receiver.is_alive()
        conn.close()
        # The monitor and receiver must release their sockets before teardown.
        for thread in monitor_threads:
            thread.join(5)
            assert not thread.is_alive()
        for proc in (publisher, local_worker):
            _stop_process(proc)
        if reader is not None:
            reader.remote_socket.context.destroy(linger=0)


@pytest.mark.parametrize("shutdown", ["coordinator_loss", "local_parent_shutdown"])
def test_worker_death_monitor_cleans_up_headless_workers(shutdown, monitor_threads):
    """A lost coordinator must kill a blocked worker and wake its local parent;
    a normal local-parent shutdown must remain graceful and nonfatal."""
    ctx = multiprocessing.get_context("spawn")
    writer = MessageQueue(1, 0, connect_ip="127.0.0.1")
    writer.remote_socket.setsockopt(zmq.RCVTIMEO, 30000)
    death_reader, death_writer = ctx.Pipe(duplex=False)
    ready = ctx.Event()
    blocked = shutdown == "coordinator_loss"
    worker = ctx.Process(
        target=_run_monitored_worker,
        args=(writer.export_handle(), death_reader, ready, blocked),
    )
    sibling_reader, sibling_writer = ctx.Pipe(duplex=False)
    sibling_ready = ctx.Event()
    sibling = ctx.Process(target=_wait_for_parent, args=(sibling_reader, sibling_ready))
    worker.start()
    death_reader.close()
    sibling.start()
    sibling_reader.close()
    failed = threading.Event()
    executor = _make_executor(
        [_worker_handle(worker, death_writer), _worker_handle(sibling, sibling_writer)],
        [],
        failed,
    )
    try:
        executor.start_worker_monitor()
        writer.wait_until_ready()
        assert ready.wait(30), "worker did not arm its death monitor"
        assert sibling_ready.wait(30), "local sibling did not start"
        if blocked:
            writer.remote_socket.close(linger=0)
            assert failed.wait(5), "coordinator loss did not reach the headless parent"
        else:
            executor.shutdown()
        worker.join(5)
        sibling.join(5)
        assert worker.exitcode == (1 if blocked else 0)
        assert sibling.exitcode == 0, "headless parent did not clean up its sibling"
        for thread in monitor_threads:
            thread.join(5)
            assert not thread.is_alive()
        assert executor.is_failed == blocked
        assert failed.is_set() == blocked
    finally:
        executor.shutdown()
        death_writer.close()
        sibling_writer.close()
        for thread in monitor_threads:
            thread.join(5)
            assert not thread.is_alive()
        for proc in (worker, sibling):
            _stop_process(proc)
        writer.remote_socket.context.destroy(linger=0)


def _run_replying_worker(conn, request_handle, async_scheduling):
    requests = MessageQueue.create_from_handle(request_handle, -1)
    replies = MessageQueue(1, 0, connect_ip="127.0.0.1")
    conn.send(replies.export_handle())
    requests.wait_until_ready()
    replies.wait_until_ready()
    worker = _make_failing_worker_proc(0, "execute_model")
    worker.worker.check_health = lambda: "healthy"
    worker.exit_on_fatal_failure = True
    worker.use_async_scheduling = async_scheduling
    worker.worker_response_mq = replies
    if async_scheduling:
        worker.async_output_queue = queue.Queue()
        threading.Thread(target=worker.async_output_busy_loop, daemon=True).start()
    for _ in range(2):
        worker._execute_worker_rpc(requests.dequeue(timeout=10))


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_remote_failure_preserves_pending_replies(async_scheduling, monitor_threads):
    ctx = multiprocessing.get_context("spawn")
    requests = MessageQueue(1, 0, connect_ip="127.0.0.1")
    conn, child_conn = ctx.Pipe()
    worker = ctx.Process(
        target=_run_replying_worker,
        args=(child_conn, requests.export_handle(), async_scheduling),
    )
    worker.start()
    child_conn.close()
    replies = executor = None
    try:
        assert conn.poll(30), "remote worker did not start"
        replies = MessageQueue.create_from_handle(conn.recv(), -1)
        failed = threading.Event()
        executor = _make_executor([], [replies], failed)
        executor.rpc_broadcast_mq = requests
        executor.futures_queue = deque()
        executor.start_worker_monitor()
        requests.wait_until_ready()
        replies.wait_until_ready()

        healthy = executor.collective_rpc(
            "check_health", non_block=True, unique_reply_rank=0
        )
        fatal = executor.collective_rpc(
            "execute_model", non_block=True, unique_reply_rank=0
        )
        assert failed.wait(10), "remote worker failure was not propagated"
        worker.join(5)
        assert worker.exitcode == 1
        # Consume only after disconnect-driven shutdown has completed.
        assert healthy.result() == "healthy"
        with pytest.raises(RuntimeError, match="execute_model failed"):
            fatal.result()
    finally:
        if executor is not None:
            executor.shutdown()
        conn.close()
        for thread in monitor_threads:
            thread.join(5)
            assert not thread.is_alive()
        _stop_process(worker)
        if replies is not None:
            replies.remote_socket.context.destroy(linger=0)
        requests.remote_socket.context.destroy(linger=0)
