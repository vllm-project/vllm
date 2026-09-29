# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import os
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any
from unittest.mock import MagicMock

import pytest

from vllm.distributed.ec_transfer.ec_connector.utils import ECOutputAggregator
from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.executor import multiproc_executor as multiproc_executor_module
from vllm.v1.executor.abstract import Executor
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.executor.uniproc_executor import (
    ExecutorWithExternalLauncher,
    UniProcExecutor,
)


class Mock: ...


class _SleepExecutor(Executor):
    """Exercise executor transitions with worker RPCs as the failure boundary."""

    def _init_executor(self) -> None:
        self.worker_rpc = MagicMock(return_value=[None])

    def collective_rpc(
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
    ) -> Any:
        return self.worker_rpc(method, timeout, args, kwargs, non_block)

    def check_health(self) -> None:
        pass


@pytest.mark.cpu_test
@pytest.mark.parametrize("level", [1, 2])
@pytest.mark.parametrize("error_type", [RuntimeError, TimeoutError])
@pytest.mark.parametrize("deferred", [False, True])
def test_failed_sleep_can_restore_before_resuming(
    level: int, error_type: type[Exception], deferred: bool
) -> None:
    """Retain sleep state on failure; resume only after a successful restore."""
    executor = _SleepExecutor(MagicMock())
    failure = error_type("injected sleep failure after unmapping")
    executor.worker_rpc.side_effect = failure
    core = object.__new__(EngineCore)
    core.model_executor = executor
    pause_future: Future[None] = Future()
    core.pause_scheduler = MagicMock(return_value=pause_future if deferred else None)
    core.resume_scheduler = MagicMock()

    if deferred:
        result = core.sleep(level=level)
        assert isinstance(result, Future)
        pause_future.set_result(None)
    with pytest.raises(error_type) as raised:
        if deferred:
            result.result()
        else:
            core.sleep(level=level)
    assert raised.value is failure
    assert executor.is_sleeping

    assert core.wake_up(tags=["scheduling"]) is False
    core.resume_scheduler.assert_not_called()

    executor.worker_rpc.side_effect = RuntimeError("incomplete restore")
    with pytest.raises(RuntimeError, match="incomplete restore"):
        core.wake_up()
    assert executor.is_sleeping
    core.resume_scheduler.assert_not_called()

    executor.worker_rpc.reset_mock()
    executor.worker_rpc.side_effect = None
    assert core.wake_up() is True
    assert not executor.is_sleeping
    core.resume_scheduler.assert_called_once_with()
    executor.worker_rpc.assert_called_once()
    assert executor.worker_rpc.call_args.args[0] == "wake_up"


@pytest.mark.cpu_test
@pytest.mark.parametrize("level", [1, 2])
def test_successful_sleep_allows_partial_wake(level: int) -> None:
    """Successful transitions still resume only after every tag is restored."""
    executor = _SleepExecutor(MagicMock())
    core = object.__new__(EngineCore)
    core.model_executor = executor
    core.pause_scheduler = MagicMock(return_value=None)
    core.resume_scheduler = MagicMock()

    core.sleep(level=level)
    assert core.wake_up(tags=["weights"]) is False
    core.resume_scheduler.assert_not_called()
    assert core.wake_up(tags=["kv_cache"]) is True
    assert not executor.is_sleeping
    core.resume_scheduler.assert_called_once_with()


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
        ec_output_aggregator: ECOutputAggregator | None = None,
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
            ec_output_aggregator,
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
