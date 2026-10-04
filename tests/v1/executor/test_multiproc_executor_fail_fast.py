# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for failing fast when a worker hits a fatal RPC failure."""

import multiprocessing
import os
import struct
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from vllm.v1.executor.multiproc_executor import MultiprocExecutor, WorkerProc


@pytest.fixture
def error_pipe():
    reader, writer = multiprocessing.Pipe(duplex=False)
    yield reader, writer
    reader.close()
    writer.close()


def _make_failing_worker_proc(rank: int, method: str, error_writer) -> Any:
    def fail(*args, **kwargs):
        raise RuntimeError(f"{method} failed")

    worker_proc: Any = WorkerProc.__new__(WorkerProc)
    worker_proc.rank = rank
    worker_proc.worker = SimpleNamespace(**{method: fail})
    worker_proc.error_writer = error_writer
    worker_proc.outputs = []
    worker_proc.handle_output = worker_proc.outputs.append
    return worker_proc


def test_recoverable_rpc_failure_is_only_returned_to_caller(error_pipe):
    """A failed utility RPC, e.g. loading an invalid LoRA adapter, must not
    fail the executor."""
    reader, writer = error_pipe
    worker_proc = _make_failing_worker_proc(0, "add_lora", writer)

    worker_proc._execute_worker_rpc(("add_lora", (), {}, None))

    assert len(worker_proc.outputs) == 1
    assert isinstance(worker_proc.outputs[0], RuntimeError)
    assert not reader.poll(0)


@pytest.mark.parametrize(
    ("method", "output_rank"),
    [("compile_or_warm_up_model", None), ("execute_model", 0)],
)
def test_fatal_rpc_failure_notifies_monitor_once(error_pipe, method, output_rank):
    """A fatal RPC failure must wake up the executor monitor even on a rank
    that doesn't reply, since its peers may be blocked in a collective."""
    reader, writer = error_pipe
    worker_proc = _make_failing_worker_proc(1, method, writer)

    for _ in range(2):
        worker_proc._execute_worker_rpc((method, (), {}, output_rank))

    assert len(worker_proc.outputs) == (2 if output_rank is None else 0)
    assert reader.poll(0)
    reader.recv_bytes()
    assert not reader.poll(0)


def _send_notification(writer) -> None:
    worker_proc: Any = WorkerProc.__new__(WorkerProc)
    worker_proc.error_writer = writer
    worker_proc._report_fatal_failure()


def _write_partial_message(writer) -> None:
    # A worker dying mid-send leaves a torn frame, for which Connection.recv()
    # raises OSError rather than EOFError.
    os.write(writer.fileno(), struct.pack("!i", 1 << 20) + b"partial")
    writer.close()


@pytest.mark.parametrize(
    "signal_failure",
    [_send_notification, _write_partial_message, lambda writer: writer.close()],
    ids=["notification", "partial_message", "closed"],
)
def test_monitor_fails_executor_on_error_pipe_event(error_pipe, signal_failure):
    """Teardown and the failure callback must not depend on what, if anything,
    can be read from the error pipe."""
    reader, writer = error_pipe
    # Never readable while both ends are open, i.e. the worker is alive.
    sentinel_reader, sentinel_writer = os.pipe()
    calls: list[str] = []
    executor: Any = MultiprocExecutor.__new__(MultiprocExecutor)
    executor.is_failed = False
    executor.shutdown = lambda: calls.append("shutdown")
    executor.failure_callback = lambda: calls.append("failure_callback")
    executor.workers = [
        SimpleNamespace(
            proc=SimpleNamespace(
                sentinel=sentinel_reader, name="VllmWorker-1", exitcode=None
            ),
            error_reader=reader,
        )
    ]

    signal_failure(writer)
    monitor = threading.Thread(
        target=executor.start_worker_monitor, kwargs={"inline": True}, daemon=True
    )
    monitor.start()
    monitor.join(timeout=10)
    # Check before closing the sentinel, which would wake up a stuck monitor.
    alive = monitor.is_alive()
    os.close(sentinel_reader)
    os.close(sentinel_writer)

    assert not alive, "monitor did not react to the error pipe"
    assert executor.is_failed
    assert calls == ["shutdown", "failure_callback"]
