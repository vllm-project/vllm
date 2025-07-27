# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.v1.engine.async_llm import AsyncEngineArgs, AsyncLLM
from vllm.v1.metrics.ray_wrappers import RayPrometheusStatLogger

DEFAULT_ENGINE_ARGS = AsyncEngineArgs(
    model="distilbert/distilgpt2",
    dtype="half",
    disable_log_stats=False,
    enforce_eager=True,
)


@pytest.mark.asyncio
async def test_async_llm_replace_default_loggers():
    # Empty stat_loggers removes default loggers
    engine = AsyncLLM.from_engine_args(DEFAULT_ENGINE_ARGS, stat_loggers=[])
    await engine.add_logger(RayPrometheusStatLogger)

    # Verify that only this logger is present in shared loggers
    assert len(engine.logger_manager.shared_loggers) == 1
    assert isinstance(engine.logger_manager.shared_loggers[0],
                      RayPrometheusStatLogger)


@pytest.mark.asyncio
async def test_async_llm_add_to_default_loggers():
    # Start with default loggers, including PrometheusStatLogger
    engine = AsyncLLM.from_engine_args(DEFAULT_ENGINE_ARGS)

    # Add another PrometheusStatLogger subclass
    await engine.add_logger(RayPrometheusStatLogger)

    assert len(engine.logger_manager.shared_loggers) == 2
