# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy

import pytest

from tests.plugins.vllm_add_dummy_stat_logger.dummy_stat_logger.dummy_stat_logger import (  # noqa E501
    DummyStatLogger,
)
from vllm.v1.engine.async_llm import AsyncEngineArgs, AsyncLLM
from vllm.v1.metrics.ray_wrappers import RayPrometheusStatLogger


@pytest.fixture
def log_stats_enabled_engine_args():
    """
    Shared fixture providing common AsyncEngineArgs configuration
    used across multiple tests.
    """
    return AsyncEngineArgs(
        model="distilbert/distilgpt2",
        dtype="half",
        disable_log_stats=False,
        enforce_eager=True,
    )


@pytest.mark.asyncio
async def test_async_llm_replace_default_loggers(log_stats_enabled_engine_args):
    """
    RayPrometheusStatLogger should replace the default PrometheusStatLogger
    """

    engine = AsyncLLM.from_engine_args(
        log_stats_enabled_engine_args, stat_loggers=[RayPrometheusStatLogger]
    )
    assert isinstance(engine.logger_manager.stat_loggers[0], RayPrometheusStatLogger)
    engine.shutdown()


@pytest.mark.asyncio
async def test_async_llm_add_to_default_loggers(log_stats_enabled_engine_args):
    """
    It's still possible to use custom stat loggers exclusively by passing
    disable_log_stats=True in addition to a list of custom stat loggers.
    """
    # Create engine_args with disable_log_stats=True for this test
    disabled_log_engine_args = copy.deepcopy(log_stats_enabled_engine_args)
    disabled_log_engine_args.disable_log_stats = True

    # Disable default loggers; pass custom stat logger to the constructor
    engine = AsyncLLM.from_engine_args(
        disabled_log_engine_args, stat_loggers=[DummyStatLogger]
    )

    assert len(engine.logger_manager.stat_loggers) == 2
    assert len(engine.logger_manager.stat_loggers[0].per_engine_stat_loggers) == 1
    assert isinstance(
        engine.logger_manager.stat_loggers[0].per_engine_stat_loggers[0],
        DummyStatLogger,
    )

    # log_stats is still True, since custom stat loggers are used
    assert engine.log_stats

    engine.shutdown()
