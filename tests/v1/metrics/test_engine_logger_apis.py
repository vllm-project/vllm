# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy

import pytest

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


@pytest.fixture
def default_dp_shared_loggers_len(log_stats_enabled_engine_args):
    """
    Fixture to provide the length of the default dp_shared_loggers
    for AsyncLLM with no custom stat loggers.
    """
    engine = AsyncLLM.from_engine_args(log_stats_enabled_engine_args,
                                       stat_loggers=[])
    length = len(engine.logger_manager.dp_shared_loggers)
    engine.shutdown()
    return length


@pytest.mark.asyncio
async def test_async_llm_replace_default_loggers(
        log_stats_enabled_engine_args, default_dp_shared_loggers_len):
    """
    The default stats loggers should be used regardless of whether additional
    custom ones are added.
    """

    engine = AsyncLLM.from_engine_args(log_stats_enabled_engine_args,
                                       stat_loggers=[RayPrometheusStatLogger])
    assert len(engine.logger_manager.dp_shared_loggers
               ) == default_dp_shared_loggers_len + 1
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

    # Disable default loggers whilst passing a custom stat logger
    engine = AsyncLLM.from_engine_args(disabled_log_engine_args,
                                       stat_loggers=[RayPrometheusStatLogger])

    # Only RayPrometheusStatLogger is available
    assert len(engine.logger_manager.dp_shared_loggers) == 1
    assert isinstance(engine.logger_manager.dp_shared_loggers[0],
                      RayPrometheusStatLogger)

    # log_stats is still True, since custom stat loggers are used
    assert engine.log_stats

    engine.shutdown()
