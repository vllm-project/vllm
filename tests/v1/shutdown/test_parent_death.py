# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test that EngineCore and its workers exit when their parent process dies
without shutting them down (API server SIGKILLed, OOM-killed or crashed)."""

import contextlib
import multiprocessing
import os
import signal
import time

import psutil
import pytest

from tests.utils import wait_for_gpu_memory_to_clear
from tests.v1.shutdown.utils import (
    SHUTDOWN_TEST_THRESHOLD_BYTES,
    SHUTDOWN_TEST_TIMEOUT_SEC,
)
from vllm.platforms import current_platform
from vllm.v1.engine.core import PARENT_DEATH_SHUTDOWN_TIMEOUT_S

MODELS = ["hmellor/tiny-random-LlamaForCausalLM"]

# Cooperative shutdown budget plus a margin for the force-kill fallback.
ORPHAN_EXIT_TIMEOUT_SEC = PARENT_DEATH_SHUTDOWN_TIMEOUT_S + 30


def _serve_until_killed(model: str, tensor_parallel_size: int, conn) -> None:
    """Child process: start an LLM, report its engine process tree, then idle
    until the test kills this process."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        enforce_eager=True,
        tensor_parallel_size=tensor_parallel_size,
    )
    llm.generate("Hello my name is", SamplingParams(max_tokens=1))
    conn.send([child.pid for child in psutil.Process().children(recursive=True)])
    conn.close()
    while True:
        time.sleep(1)


def _is_running(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _wait_until_gone(pids: list[int], timeout: float) -> list[int]:
    """Return the pids still running after ``timeout`` seconds."""
    deadline = time.monotonic() + timeout
    while True:
        alive = [pid for pid in pids if _is_running(pid)]
        if not alive or time.monotonic() >= deadline:
            return alive
        time.sleep(0.5)


def _kill_all(pids: list[int]) -> None:
    for pid in pids:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)


@pytest.mark.timeout(SHUTDOWN_TEST_TIMEOUT_SEC * 2)
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("tensor_parallel_size", [2, 1])
def test_engine_exits_after_parent_sigkill(model: str, tensor_parallel_size: int):
    """SIGKILL the process that owns a running LLM; EngineCore and every
    worker must exit on their own and release GPU memory."""
    if current_platform.device_count() < tensor_parallel_size:
        pytest.skip(reason="Not enough CUDA devices")

    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_serve_until_killed, args=(model, tensor_parallel_size, child_conn)
    )
    proc.start()
    child_conn.close()
    assert proc.pid is not None
    parent_pid = proc.pid
    engine_pids: list[int] = []
    try:
        assert parent_conn.poll(SHUTDOWN_TEST_TIMEOUT_SEC), "LLM did not start"
        engine_pids = parent_conn.recv()
        # EngineCore, plus one worker per rank when TP > 1.
        expected = 1 + (tensor_parallel_size if tensor_parallel_size > 1 else 0)
        assert len(engine_pids) >= expected, engine_pids
        assert all(_is_running(pid) for pid in engine_pids)

        os.kill(parent_pid, signal.SIGKILL)
        proc.join(10)
        assert proc.exitcode == -signal.SIGKILL

        orphans = _wait_until_gone(engine_pids, ORPHAN_EXIT_TIMEOUT_SEC)
        assert not orphans, f"engine processes outlived their parent: {orphans}"
    finally:
        if proc.is_alive():
            proc.kill()
        _kill_all(engine_pids)

    wait_for_gpu_memory_to_clear(
        devices=list(range(tensor_parallel_size)),
        threshold_bytes=SHUTDOWN_TEST_THRESHOLD_BYTES,
    )
