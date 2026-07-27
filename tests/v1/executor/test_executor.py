# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import os
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any

import pytest

from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.executor import multiproc_executor as multiproc_executor_module
from vllm.v1.executor.abstract import Executor
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.executor.uniproc_executor import (
    ExecutorWithExternalLauncher,
    UniProcExecutor,
)


class Mock: ...


def test_supports_async_scheduling_base_executor():
    assert Executor.supports_async_scheduling() is False


def test_supports_async_scheduling_uniproc_executor():
    assert UniProcExecutor.supports_async_scheduling() is True


def test_supports_async_scheduling_executor_with_external_launcher():
    # ExecutorWithExternalLauncher inherits from UniProcExecutor and does not
    # override supports_async_scheduling, so it should return True.
    assert ExecutorWithExternalLauncher.supports_async_scheduling() is True


def test_supports_async_scheduling_multiproc_executor():
    assert MultiprocExecutor.supports_async_scheduling() is True


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class _FakeProcess:
    def __init__(self, clock: _FakeClock, exits_at: float) -> None:
        self.clock = clock
        self.exits_at = exits_at
        self.terminate_called = False

    def is_alive(self) -> bool:
        return self.clock.time() < self.exits_at

    def terminate(self) -> None:
        self.terminate_called = True


@pytest.mark.parametrize(
    ("timeout", "exits_at", "expected_terminate"),
    [
        pytest.param(6, 5, False, id="worker-exits-before-timeout"),
        pytest.param(6, 7, True, id="worker-exceeds-timeout"),
    ],
)
def test_multiproc_executor_worker_termination_timeout(
    monkeypatch, timeout, exits_at, expected_terminate
):
    monkeypatch.setenv("VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS", str(timeout))
    clock = _FakeClock()
    monkeypatch.setattr(multiproc_executor_module.time, "time", clock.time)
    monkeypatch.setattr(multiproc_executor_module.time, "sleep", clock.sleep)
    executor = MultiprocExecutor.__new__(MultiprocExecutor)
    proc = _FakeProcess(clock, exits_at=exits_at)
    executor._ensure_worker_termination([proc])
    assert proc.terminate_called is expected_terminate


class _TimingOutExecutor(Executor):
    """Minimal Executor whose sleep/wake collective RPC always times out.

    Mirrors the MultiprocExecutor contract: ``is_failed`` exists and gates
    ``collective_rpc`` into the "Executor failed." fast-fail, ``shutdown`` and a
    one-shot ``failure_callback`` exist exactly like the real worker-death path
    (``start_worker_monitor``). Used to prove that a bounded sleep/wake RPC
    timeout tears the executor down (callback fires → EXECUTOR_FAILED enqueued)
    instead of a silent ``is_sleeping`` split-brain. No GPU/torch.
    """

    def __init__(self) -> None:
        # Bypass the heavy base __init__ (workers, RPC mq, etc.).
        self.is_sleeping = False
        self.sleeping_tags = set()
        self.is_failed = False
        self.failure_callback = None
        self.shutdown_called = False

    def _init_executor(self) -> None:  # pragma: no cover - not exercised
        ...

    def check_health(self) -> None:  # pragma: no cover - not exercised
        ...

    def shutdown(self) -> None:
        self.shutdown_called = True

    def register_failure_callback(self, callback) -> None:
        if self.is_failed:
            callback()
        else:
            self.failure_callback = callback

    def collective_rpc(
        self, method, timeout=None, args=(), kwargs=None, non_block=False
    ):
        # A non-zero timeout was threaded through: simulate a dead/hung worker
        # whose reply never arrives, exactly like MultiprocExecutor raises.
        if self.is_failed:
            raise RuntimeError("Executor failed.")
        assert timeout is not None, "expected a bounded timeout to be threaded through"
        raise TimeoutError(f"RPC call to {method} timed out.")


def test_wake_up_timeout_marks_executor_failed_not_silent_sleeping(monkeypatch):
    """Regression: a timed-out wake must tear the executor down, not silently wedge.

    Pre-fix, ``wake_up`` only set ``is_failed=True`` on the timeout. That is NOT
    enough to recover: ``wake_up`` re-raises before ``EngineCore.wake_up`` reaches
    ``resume_scheduler()``, so the scheduler stays paused, ``has_work()`` is false,
    no engine step ever runs and ``is_failed`` is never consulted — the engine
    sits alive forever with ``is_sleeping()==True`` and ``/health`` lying 200. The
    only thing that actually tears the engine down is the registered
    ``failure_callback`` (it enqueues ``EXECUTOR_FAILED`` →
    ``RuntimeError("Executor failed.")`` → ``_send_engine_dead``).

    Post-fix, ``_fail_on_sleep_wake_timeout`` mirrors the real worker-death path:
    ``is_failed=True`` + ``shutdown()`` + ``failure_callback()``. This test
    asserts the callback fired (engine-dead path triggered) and the executor was
    shut down — both FAIL pre-fix (callback never invoked, only is_failed set) and
    PASS post-fix — without falsely flipping ``is_sleeping``.
    """
    monkeypatch.setenv("VLLM_SLEEP_WAKE_TIMEOUT_SECONDS", "30")

    executor = _TimingOutExecutor()
    executor.is_sleeping = True
    executor.sleeping_tags = {"weights", "kv_cache"}

    # Engine registers a failure callback (here: a stand-in for the lambda that
    # enqueues EXECUTOR_FAILED on the engine input queue).
    callback_fired: list[bool] = []
    executor.register_failure_callback(lambda: callback_fired.append(True))

    with pytest.raises(TimeoutError):
        executor.wake_up()

    # The engine-dead path MUST have been triggered: the failure callback fired
    # (EXECUTOR_FAILED would be enqueued) and the executor was shut down. This is
    # the core of the fix — setting is_failed alone left the engine wedged.
    assert callback_fired == [True], (
        "wake_up timeout must invoke the failure callback (EXECUTOR_FAILED "
        "enqueued → engine teardown), not merely set is_failed"
    )
    assert executor.shutdown_called is True
    # The callback is one-shot (mirrors monitor_workers): cleared after firing.
    assert executor.failure_callback is None
    # Still a detectable FAILED state...
    assert executor.is_failed is True
    # ...and it must NOT lie that the workers are awake (is_sleeping flipped to
    # False) — workers may never have woken.
    assert executor.is_sleeping is True
    # The failed flag means a subsequent RPC fast-fails (recovery trigger).
    with pytest.raises(RuntimeError, match="Executor failed."):
        executor.collective_rpc("noop")


def test_sleep_timeout_marks_executor_failed(monkeypatch):
    """A timed-out sleep also tears the executor down (failure callback fired +
    shutdown) and re-raises, instead of falsely recording is_sleeping=True for
    dead workers."""
    monkeypatch.setenv("VLLM_SLEEP_WAKE_TIMEOUT_SECONDS", "30")

    executor = _TimingOutExecutor()
    assert executor.is_sleeping is False

    callback_fired: list[bool] = []
    executor.register_failure_callback(lambda: callback_fired.append(True))

    with pytest.raises(TimeoutError):
        executor.sleep()

    assert callback_fired == [True], (
        "sleep timeout must invoke the failure callback, not merely set is_failed"
    )
    assert executor.shutdown_called is True
    assert executor.failure_callback is None
    assert executor.is_failed is True
    # Did NOT falsely flip to sleeping — the sleep RPC never completed.
    assert executor.is_sleeping is False
    assert executor.sleeping_tags == set()


def test_sleep_wake_timeout_without_callback_still_fails_safe(monkeypatch):
    """Executors that never registered a failure callback (or lack one) must
    still avoid the lying state update and end up failed + shut down — the
    teardown degrades gracefully when no callback is present."""
    monkeypatch.setenv("VLLM_SLEEP_WAKE_TIMEOUT_SECONDS", "30")

    executor = _TimingOutExecutor()  # no register_failure_callback call
    assert executor.failure_callback is None

    with pytest.raises(TimeoutError):
        executor.sleep()

    assert executor.is_failed is True
    assert executor.shutdown_called is True
    assert executor.is_sleeping is False


def test_sleep_wake_default_disabled_threads_none_timeout(monkeypatch):
    """When VLLM_SLEEP_WAKE_TIMEOUT_SECONDS=0 (default), the RPC timeout is
    None — i.e. the bounded-timeout behavior is fully dormant and the
    sleep/wake path is unchanged from upstream (no teardown, no callback)."""
    monkeypatch.setenv("VLLM_SLEEP_WAKE_TIMEOUT_SECONDS", "0")

    captured: dict[str, object] = {}

    class _CapturingExecutor(_TimingOutExecutor):
        def collective_rpc(
            self, method, timeout=None, args=(), kwargs=None, non_block=False
        ):
            captured["timeout"] = timeout
            return []

    executor = _CapturingExecutor()
    callback_fired: list[bool] = []
    executor.register_failure_callback(lambda: callback_fired.append(True))
    executor.sleep()
    assert captured["timeout"] is None
    assert executor.is_sleeping is True
    assert executor.is_failed is False
    # Dormant path: no teardown, callback untouched.
    assert callback_fired == []
    assert executor.shutdown_called is False
    assert executor.failure_callback is not None


class CustomMultiprocExecutor(MultiprocExecutor):
    def collective_rpc(
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        unique_reply_rank: int | None = None,
        kv_output_aggregator: KVOutputAggregator = None,
    ) -> Any | list[Any] | Future[Any | list[Any]]:
        # Drop marker to show that this was run
        with open(".marker", "w"):
            ...
        return super().collective_rpc(
            method,
            timeout,
            args,
            kwargs,
            non_block,
            unique_reply_rank,
            kv_output_aggregator,
        )


CustomMultiprocExecutorAsync = CustomMultiprocExecutor
MODEL = "Qwen/Qwen3-0.6B"


def test_custom_executor_type_checking():
    with pytest.raises(ValueError):
        engine_args = EngineArgs(
            model=MODEL,
            gpu_memory_utilization=0.2,
            max_model_len=8192,
            distributed_executor_backend=Mock,
        )
        LLMEngine.from_engine_args(engine_args)
    with pytest.raises(ValueError):
        engine_args = AsyncEngineArgs(
            model=MODEL,
            gpu_memory_utilization=0.2,
            max_model_len=8192,
            distributed_executor_backend=Mock,
        )
        AsyncLLM.from_engine_args(engine_args)


@pytest.mark.parametrize(
    "distributed_executor_backend",
    [
        CustomMultiprocExecutor,
        "tests.v1.executor.test_executor.CustomMultiprocExecutor",
    ],
)
def test_custom_executor(distributed_executor_backend, tmp_path):
    cwd = os.path.abspath(".")
    os.chdir(tmp_path)
    try:
        assert not os.path.exists(".marker")

        engine_args = EngineArgs(
            model=MODEL,
            gpu_memory_utilization=0.2,
            max_model_len=8192,
            distributed_executor_backend=distributed_executor_backend,
            enforce_eager=True,  # reduce test time
        )
        engine = LLMEngine.from_engine_args(engine_args)
        sampling_params = SamplingParams(max_tokens=1)

        engine.add_request("0", "foo", sampling_params)
        engine.step()

        assert os.path.exists(".marker")
    finally:
        os.chdir(cwd)


@pytest.mark.parametrize(
    "distributed_executor_backend",
    [
        CustomMultiprocExecutorAsync,
        "tests.v1.executor.test_executor.CustomMultiprocExecutorAsync",
    ],
)
def test_custom_executor_async(distributed_executor_backend, tmp_path):
    cwd = os.path.abspath(".")
    os.chdir(tmp_path)
    try:
        assert not os.path.exists(".marker")

        engine_args = AsyncEngineArgs(
            model=MODEL,
            gpu_memory_utilization=0.2,
            max_model_len=8192,
            distributed_executor_backend=distributed_executor_backend,
            enforce_eager=True,  # reduce test time
        )
        engine = AsyncLLM.from_engine_args(engine_args)
        sampling_params = SamplingParams(max_tokens=1)

        async def t():
            stream = engine.generate(
                request_id="0", prompt="foo", sampling_params=sampling_params
            )
            async for x in stream:
                ...

        asyncio.run(t())

        assert os.path.exists(".marker")
    finally:
        os.chdir(cwd)
