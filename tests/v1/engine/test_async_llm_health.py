# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import gc
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core import DPEngineCoreProc, EngineCore
from vllm.v1.engine.exceptions import EngineDeadError


@pytest.fixture
def async_llm_for_health_check(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("vllm.envs.VLLM_HEALTH_CHECK_GPU_TIMEOUT", 1)

    async_llm = AsyncLLM.__new__(AsyncLLM)
    async_llm.engine_core = SimpleNamespace(
        resources=SimpleNamespace(engine_dead=False),
        check_ready_async=AsyncMock(return_value=True),
        shutdown=Mock(),
    )
    async_llm.output_handler = None
    async_llm._gpu_health_probe = None

    return async_llm


def block_probe(async_llm: AsyncLLM, exc: Exception | None = None):
    """Make the engine-side probe hang until the returned event is set."""
    release = asyncio.Event()

    async def check_ready() -> bool:
        await release.wait()
        if exc is not None:
            raise exc
        return True

    async_llm.engine_core.check_ready_async.side_effect = check_ready
    return release


@pytest.mark.asyncio
@pytest.mark.parametrize("engine_ready", [True, False])
async def test_check_health_gpu_reports_engine_readiness(
    async_llm_for_health_check, engine_ready
):
    """The engine reports not ready while sleeping or paused."""
    engine_core = async_llm_for_health_check.engine_core
    engine_core.check_ready_async.return_value = engine_ready

    assert await async_llm_for_health_check.check_health_gpu() is engine_ready
    engine_core.check_ready_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_check_health_gpu_fails_when_engine_dead(async_llm_for_health_check):
    async_llm_for_health_check.engine_core.resources.engine_dead = True

    with pytest.raises(EngineDeadError):
        await async_llm_for_health_check.check_health_gpu()

    engine_core = async_llm_for_health_check.engine_core
    engine_core.check_ready_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_check_health_gpu_not_ready_when_dummy_batch_fails(
    async_llm_for_health_check,
):
    engine_core = async_llm_for_health_check.engine_core
    engine_core.check_ready_async.side_effect = RuntimeError("GPU failed")

    assert not await async_llm_for_health_check.check_health_gpu()


@pytest.mark.asyncio
async def test_check_health_gpu_concurrent_callers_share_one_probe(
    async_llm_for_health_check,
):
    release = block_probe(async_llm_for_health_check)
    callers = [
        asyncio.create_task(async_llm_for_health_check.check_health_gpu())
        for _ in range(4)
    ]
    await asyncio.sleep(0.01)
    release.set()

    assert all(await asyncio.gather(*callers))
    engine_core = async_llm_for_health_check.engine_core
    engine_core.check_ready_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_check_health_gpu_timeout_does_not_enqueue_second_probe(
    async_llm_for_health_check, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("vllm.envs.VLLM_HEALTH_CHECK_GPU_TIMEOUT", 0.01)
    release = block_probe(async_llm_for_health_check)
    engine_core = async_llm_for_health_check.engine_core

    assert not await async_llm_for_health_check.check_health_gpu()
    assert not await async_llm_for_health_check.check_health_gpu()
    engine_core.check_ready_async.assert_awaited_once()

    # The timed-out probe keeps running and is cleared once it finishes.
    probe = async_llm_for_health_check._gpu_health_probe
    release.set()
    assert await probe
    await asyncio.sleep(0)
    assert async_llm_for_health_check._gpu_health_probe is None

    assert await async_llm_for_health_check.check_health_gpu()
    assert engine_core.check_ready_async.await_count == 2


@pytest.mark.asyncio
async def test_check_health_gpu_retrieves_exception_of_abandoned_probe(
    async_llm_for_health_check, monkeypatch: pytest.MonkeyPatch
):
    """No "Task exception was never retrieved" once every caller timed out."""
    monkeypatch.setattr("vllm.envs.VLLM_HEALTH_CHECK_GPU_TIMEOUT", 0.01)
    release = block_probe(async_llm_for_health_check, RuntimeError("boom"))
    loop = asyncio.get_running_loop()
    unhandled: list[dict] = []
    old_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _, context: unhandled.append(context))
    try:
        assert not await async_llm_for_health_check.check_health_gpu()
        probe = async_llm_for_health_check._gpu_health_probe
        release.set()
        await asyncio.wait([probe])
        del probe
        gc.collect()
    finally:
        loop.set_exception_handler(old_handler)

    assert async_llm_for_health_check._gpu_health_probe is None
    assert not unhandled


def make_engine_core(cls: type[EngineCore], sleeping: bool) -> EngineCore:
    engine_core = object.__new__(cls)
    engine_core.is_sleeping = Mock(return_value=sleeping)
    engine_core.scheduler = Mock()
    engine_core.scheduler.has_requests.return_value = False
    engine_core.batch_queue = None
    engine_core.model_executor = Mock()
    return engine_core


@pytest.mark.parametrize(
    ("sleeping", "has_requests", "batch_queue", "ready", "ran_dummy_batch"),
    [
        (False, False, None, True, True),
        (False, True, None, True, False),
        (False, False, deque([Mock()]), True, False),
        (True, False, None, False, False),
    ],
    ids=["idle", "requests", "batch_in_flight", "sleeping"],
)
def test_engine_core_check_ready(
    sleeping, has_requests, batch_queue, ready, ran_dummy_batch
):
    """The dummy batch runs only on an idle, awake engine."""
    engine_core = make_engine_core(EngineCore, sleeping)
    engine_core.scheduler.has_requests.return_value = has_requests
    engine_core.batch_queue = batch_queue

    assert engine_core.check_ready() is ready
    assert engine_core.model_executor.execute_dummy_batch.called is ran_dummy_batch


@pytest.mark.parametrize("sleeping", [False, True])
def test_dp_engine_core_check_ready_skips_dummy_batch(sleeping):
    """A MoE DP rank must not enter the DP all-reduce without its peers."""
    engine_core = make_engine_core(DPEngineCoreProc, sleeping)

    assert engine_core.check_ready() is not sleeping
    engine_core.model_executor.execute_dummy_batch.assert_not_called()
