# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import multiprocessing
import os
from types import SimpleNamespace

import pytest

from vllm.v1.engine.utils import (
    STARTUP_FAILURE,
    StartupErrorPayload,
    WorkerStartupMessage,
    build_startup_error_payload,
)
from vllm.v1.executor.multiproc_executor import UnreadyWorkerProcHandle, WorkerProc


def _send_worker_startup_failure(ready_pipe, rank: int):
    try:
        raise ValueError("simulated worker startup failure")
    except Exception as exc:
        ready_pipe.send(
            WorkerStartupMessage(
                status=STARTUP_FAILURE,
                error=build_startup_error_payload(
                    exc,
                    source_process=multiprocessing.current_process().name,
                    source_rank=rank,
                    pid=os.getpid(),
                ),
            )
        )
    finally:
        ready_pipe.close()


def test_worker_startup_failure_surfaces_root_cause():
    ctx = multiprocessing.get_context("spawn")
    ready_reader, ready_writer = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_send_worker_startup_failure,
        args=(ready_writer, 7),
        name="VllmWorker-7",
    )
    proc.start()
    ready_writer.close()

    try:
        handle = UnreadyWorkerProcHandle(proc=proc, rank=7, ready_pipe=ready_reader)
        with pytest.raises(RuntimeError) as exc_info:
            WorkerProc.wait_for_ready([handle])

        message = str(exc_info.value)
        assert "WorkerProc initialization failed." in message
        assert "Failed worker proc(s):" in message
        assert "VllmWorker-7" in message
        assert f"pid={proc.pid}" in message
        assert "rank=7" in message
        assert "Root cause: ValueError: simulated worker startup failure" in message
        assert "Child traceback:" in message
    finally:
        ready_reader.close()
        proc.join(timeout=10)


class FakeReadyPipe:
    def __init__(self, response: WorkerStartupMessage):
        self.response = response

    def recv(self) -> WorkerStartupMessage:
        return self.response

    def close(self) -> None:
        pass


def test_worker_startup_failure_dedupes_and_preserves_exitcode(monkeypatch):
    ready_pipe = FakeReadyPipe(
        WorkerStartupMessage(
            status=STARTUP_FAILURE,
            error=StartupErrorPayload(
                error_type="ValueError",
                message="simulated worker startup failure",
                traceback="Traceback line 1\nTraceback line 2",
                source_process="VllmWorker-7",
                source_rank=7,
                pid=4321,
            ),
        )
    )
    handle = UnreadyWorkerProcHandle(
        proc=SimpleNamespace(name="VllmWorker-7", pid=4321, exitcode=1),
        rank=7,
        ready_pipe=ready_pipe,
    )
    monkeypatch.setattr(
        multiprocessing.connection,
        "wait",
        lambda _pipes: [ready_pipe],
    )

    with pytest.raises(RuntimeError) as exc_info:
        WorkerProc.wait_for_ready([handle])

    message = str(exc_info.value)
    assert message.count("VllmWorker-7(") == 1
    assert "Failed worker proc(s): VllmWorker-7(pid=4321, exitcode=1)" in message
    assert "Source: VllmWorker-7 (rank=7, pid=4321)" in message
    assert "Root cause: ValueError: simulated worker startup failure" in message
