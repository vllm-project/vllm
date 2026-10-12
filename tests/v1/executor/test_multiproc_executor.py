# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import subprocess
import sys
import textwrap
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
    ("method", "delay", "hold_gil", "expected_dumps"),
    [
        ("execute_model", 0.75, False, 1),
        ("execute_model", 0.75, True, 1),
        ("sample_tokens", 0, False, 0),
        ("load_model", 0.75, False, 0),
    ],
)
def test_slow_runtime_rpc_dumps_stacks_without_interrupting_work(
    method, delay, hold_gil, expected_dumps
):
    """Dump even with a held GIL; preserve results and cancel on RPC exit."""
    script = textwrap.dedent(
        f"""\
        import ctypes
        import time
        from types import SimpleNamespace
        from vllm.v1.executor.multiproc_executor import WorkerProc

        def runtime_call():
            if {hold_gil}:
                # PyDLL keeps the GIL held across the native call.
                ctypes.PyDLL(None).usleep({int(delay * 1_000_000)})
            else:
                time.sleep({delay})
            return 42

        worker = WorkerProc.__new__(WorkerProc)
        worker.rank = 0
        worker.worker = SimpleNamespace(**{{{method!r}: runtime_call}})
        results = []
        worker.handle_output = results.append
        worker._execute_worker_rpc(({method!r}, (), {{}}, None))
        assert results == [42]
        time.sleep(0.5)
        print("rpc completed")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "VLLM_WORKER_RPC_DUMP_TIMEOUT_S": "0.25"},
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert "rpc completed" in result.stdout
    assert result.stderr.count("Timeout (") == expected_dumps
    if expected_dumps:
        assert "in runtime_call" in result.stderr
