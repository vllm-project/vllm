# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.v1.engine.async_llm import AsyncEngineArgs, AsyncLLM
from vllm.v1.metrics.loggers import PrometheusStatLogger


@pytest.mark.asyncio
async def test_async_llm_add_logger():
    # Minimal model config for test
    model_name = "distilbert/distilgpt2"
    dtype = "half"
    engine_args = AsyncEngineArgs(
        model=model_name,
        dtype=dtype,
        disable_log_stats=False,
        enforce_eager=True,
    )

    # Force empty list to avoid default loggers
    engine = AsyncLLM.from_engine_args(engine_args, stat_loggers=[])

    # Add PrometheusStatLogger and verify no exception is raised
    await engine.add_logger(PrometheusStatLogger)

    # Verify that logger is present in the first DP rank
    assert len(engine.stat_loggers[0]) == 1
    assert isinstance(engine.stat_loggers[0][0], PrometheusStatLogger)