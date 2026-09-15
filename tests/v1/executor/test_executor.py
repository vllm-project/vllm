# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import os
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any

import pytest

from vllm.distributed.ec_transfer.ec_connector.utils import ECOutputAggregator
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


def test_failed_checkpoint_keeps_wake_available(monkeypatch):
    """A failed sleep or restore must not advertise resident memory."""
    from types import SimpleNamespace

    monkeypatch.setattr("vllm.envs.VLLM_SLEEP_OFFLOAD_CUDA_CONTEXT", True)
    executor = SimpleNamespace(is_sleeping=False, sleeping_tags=set())

    def fail(method, **kwargs):
        raise RuntimeError("driver failure")

    executor.collective_rpc = fail
    with pytest.raises(RuntimeError, match="driver failure"):
        Executor.sleep(executor)
    assert executor.is_sleeping
    assert executor.sleeping_tags == {"weights", "kv_cache"}
    with pytest.raises(RuntimeError, match="driver failure"):
        Executor.wake_up(executor)
    assert executor.is_sleeping
    executor.collective_rpc = lambda *args, **kwargs: None
    Executor.wake_up(executor)
    assert not executor.is_sleeping
    assert executor.sleeping_tags == set()


@pytest.mark.parametrize(
    "cuda, dp_backend, worker_cls",
    [(False, "mp", "auto"), (True, "ray", "auto"), (True, "mp", "custom.Worker")],
)
def test_checkpoint_rejects_unsupported_engine_process(
    monkeypatch, cuda, dp_backend, worker_cls
):
    """Reject unsupported platforms and actor launchers before worker startup."""
    from types import SimpleNamespace

    from vllm.platforms import current_platform

    monkeypatch.setattr("vllm.envs.VLLM_SLEEP_OFFLOAD_CUDA_CONTEXT", True)
    monkeypatch.setattr(current_platform, "is_cuda", lambda: cuda)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_backend=dp_backend, worker_cls=worker_cls
        )
    )
    with pytest.raises(ValueError, match="CUDA and a local engine"):
        Executor.__init__(SimpleNamespace(), config)


class Mock: ...


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("parallel_config", "world_size", 2),
        ("parallel_config", "data_parallel_size", 2),
        ("parallel_config", "distributed_executor_backend", "external_launcher"),
        ("model_config", "enable_sleep_mode", False),
        ("model_config", "sleep_mode_backend", "custom"),
        (None, "use_v2_model_runner", False),
        (None, "multiprocessing", False),
    ],
)
def test_checkpoint_requires_isolated_v2_cumem_worker(
    monkeypatch, section, field, value
):
    from types import SimpleNamespace

    from vllm.platforms import current_platform

    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            world_size=1,
            data_parallel_size=1,
            data_parallel_backend="mp",
            worker_cls="vllm.v1.worker.gpu_worker.Worker",
            distributed_executor_backend="uni",
        ),
        model_config=SimpleNamespace(
            enable_sleep_mode=True, sleep_mode_backend="cumem"
        ),
        use_v2_model_runner=True,
    )
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr("vllm.envs.VLLM_SLEEP_OFFLOAD_CUDA_CONTEXT", True)
    monkeypatch.setattr(
        "vllm.envs.VLLM_ENABLE_V1_MULTIPROCESSING", field != "multiprocessing"
    )
    if field != "multiprocessing":
        setattr(getattr(config, section) if section else config, field, value)
    with pytest.raises(ValueError, match="cuMem sleep mode"):
        Executor.__init__(SimpleNamespace(), config)


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
