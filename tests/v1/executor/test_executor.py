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


class _RecordingExecutor(Executor):
    """Minimal concrete Executor that records collective_rpc calls.

    Constructed via __new__ to bypass __init__ (which needs a VllmConfig);
    sleep/wake_up state is set directly. This keeps the test pure-CPU and
    exercises the inherited Executor.wake_up tag-handling logic in isolation.
    """

    def _init_executor(self) -> None: ...

    def collective_rpc(
        self, method, timeout=None, args=(), kwargs=None, non_block=False
    ):
        self.rpc_calls.append((method, (kwargs or {}).get("tags")))
        return []

    def check_health(self) -> None: ...


def _make_sleeping_executor() -> _RecordingExecutor:
    executor = _RecordingExecutor.__new__(_RecordingExecutor)
    executor.rpc_calls = []
    # Mirror Executor.sleep() state: both tags asleep.
    executor.sleeping_tags = {"weights", "kv_cache"}
    executor.is_sleeping = True
    return executor


def test_wake_up_mixed_tags_wakes_still_sleeping_subset():
    # Regression for the mixed-tag partial-wake silent no-op: when wake_up is
    # called with one already-awake tag and one still-asleep tag, the asleep
    # tag must still be woken (an RPC is issued for the sleeping subset),
    # instead of early-returning and leaving it unmapped.
    executor = _make_sleeping_executor()

    # Wake only "weights" first -> "kv_cache" remains asleep.
    executor.wake_up(tags=["weights"])
    assert executor.sleeping_tags == {"kv_cache"}
    assert executor.is_sleeping is True

    executor.rpc_calls.clear()

    # Mixed request: "weights" already awake, "kv_cache" still asleep.
    executor.wake_up(tags=["weights", "kv_cache"])

    # An RPC must be issued, and it must include the still-sleeping tag.
    assert len(executor.rpc_calls) == 1, executor.rpc_calls
    method, rpc_tags = executor.rpc_calls[0]
    assert method == "wake_up"
    assert "kv_cache" in (rpc_tags or [])
    # The already-awake tag is filtered out of the RPC.
    assert "weights" not in (rpc_tags or [])
    # State fully reconciled: everything awake.
    assert executor.sleeping_tags == set()
    assert executor.is_sleeping is False


def test_wake_up_all_already_awake_is_clean_no_op():
    # When every requested tag is already awake, wake_up is a clean no-op:
    # no RPC is issued and state is unchanged.
    executor = _make_sleeping_executor()
    executor.wake_up(tags=["weights"])
    assert executor.sleeping_tags == {"kv_cache"}

    executor.rpc_calls.clear()
    executor.wake_up(tags=["weights"])  # already awake

    assert executor.rpc_calls == []
    assert executor.sleeping_tags == {"kv_cache"}
    assert executor.is_sleeping is True


def test_wake_up_no_tags_wakes_everything():
    # tags=None wakes all sleeping tags and clears the sleeping state.
    executor = _make_sleeping_executor()
    executor.wake_up()
    assert len(executor.rpc_calls) == 1
    method, rpc_tags = executor.rpc_calls[0]
    assert method == "wake_up"
    assert rpc_tags is None
    assert executor.sleeping_tags == set()
    assert executor.is_sleeping is False
