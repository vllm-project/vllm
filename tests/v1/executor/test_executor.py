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
