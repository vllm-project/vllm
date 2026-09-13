# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.exceptions import EngineDeadError


@pytest.fixture
def async_llm_for_health_check(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("vllm.envs.VLLM_HEALTH_CHECK_GPU_TIMEOUT", 1)

    async_llm = AsyncLLM.__new__(AsyncLLM)
    async_llm.engine_core = SimpleNamespace(
        resources=SimpleNamespace(engine_dead=False),
        execute_dummy_batch_async=AsyncMock(),
        shutdown=Mock(),
    )
    async_llm.output_handler = None
    async_llm.output_processor = Mock()
    async_llm.output_processor.has_unfinished_requests.return_value = False

    return async_llm


@pytest.mark.asyncio
async def test_check_health_gpu_runs_dummy_batch_when_idle(async_llm_for_health_check):
    await async_llm_for_health_check.check_health_gpu()

    output_processor = async_llm_for_health_check.output_processor
    output_processor.has_unfinished_requests.assert_called_once()
    engine_core = async_llm_for_health_check.engine_core
    engine_core.execute_dummy_batch_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_check_health_gpu_skips_dummy_batch_when_busy(async_llm_for_health_check):
    output_processor = async_llm_for_health_check.output_processor
    output_processor.has_unfinished_requests.return_value = True

    await async_llm_for_health_check.check_health_gpu()

    output_processor.has_unfinished_requests.assert_called_once()
    engine_core = async_llm_for_health_check.engine_core
    engine_core.execute_dummy_batch_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_check_health_gpu_fails_when_engine_dead(async_llm_for_health_check):
    async_llm_for_health_check.engine_core.resources.engine_dead = True

    with pytest.raises(EngineDeadError):
        await async_llm_for_health_check.check_health_gpu()

    output_processor = async_llm_for_health_check.output_processor
    output_processor.has_unfinished_requests.assert_not_called()
    engine_core = async_llm_for_health_check.engine_core
    engine_core.execute_dummy_batch_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_check_health_gpu_fails_when_dummy_batch_fails(
    async_llm_for_health_check,
):
    engine_core = async_llm_for_health_check.engine_core
    engine_core.execute_dummy_batch_async.side_effect = RuntimeError("GPU failed")

    with pytest.raises(EngineDeadError):
        await async_llm_for_health_check.check_health_gpu()

    engine_core.execute_dummy_batch_async.assert_awaited_once()
