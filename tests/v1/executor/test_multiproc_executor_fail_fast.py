# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for failing fast when a worker hits a fatal RPC failure."""

import multiprocessing
import queue
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
from vllm.v1.executor.multiproc_executor import MultiprocExecutor, WorkerProc

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

    calls: list[str] = []
    executor: Any = MultiprocExecutor.__new__(MultiprocExecutor)
    executor.is_failed = False
    executor.shutdown = lambda: calls.append("shutdown")
    executor.failure_callback = lambda: calls.append("failure_callback")
    executor.workers = [SimpleNamespace(proc=proc)]
    executor.start_worker_monitor(inline=True)

    assert executor.is_failed
    assert calls == ["shutdown", "failure_callback"]
