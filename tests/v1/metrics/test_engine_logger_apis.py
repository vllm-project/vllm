# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
from unittest.mock import Mock

import pytest

from tests.plugins.vllm_add_dummy_stat_logger.dummy_stat_logger.dummy_stat_logger import (  # noqa E501
    DummyStatLogger,
)
from tests.utils import wait_for_memory_to_settle
from vllm.config import DeviceConfig, ObservabilityConfig, VllmConfig
from vllm.v1.engine.async_llm import AsyncEngineArgs, AsyncLLM
from vllm.v1.metrics.loggers import LoggingStatLogger, PerEngineStatLoggerAdapter
from vllm.v1.metrics.ray_wrappers import RayPrometheusStatLogger
from vllm.v1.metrics.stats import SchedulerIterationDetails, SchedulerStats


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize(
    "verbose,batched", [(False, True), (True, False), (True, True)]
)
def test_iteration_logging_preserves_records_without_inventing_timing(
    monkeypatch, verbose, batched
):
    stat_logger = LoggingStatLogger(
        VllmConfig(
            device_config=DeviceConfig("cpu"),
            observability_config=ObservabilityConfig(
                enable_logging_iteration_details=verbose
            ),
        )
    )
    info = Mock()
    monkeypatch.setattr("vllm.v1.metrics.loggers.logger.info", info)
    untimed = SchedulerIterationDetails(
        7,
        1,
        16,
        0,
        0,
        elapsed_ms=None,
        num_encoder_inputs=1,
        num_encoder_output_tokens=32,
    )
    timed = SchedulerIterationDetails(
        8, 0, 0, 0, 0, elapsed_ms=1.25, is_dummy=True, timing_scope="model_step_cuda"
    )
    stats = SchedulerStats(
        kv_cache_usage=0.9,
        iteration_details=untimed,
        iteration_records=[untimed, timed] if batched else [],
    )

    stat_logger.record(stats, None, engine_idx=2)

    messages = [args[0] % args[1:] for args, _ in info.call_args_list]
    if not verbose:
        assert messages == []
        return
    assert len(messages) == (2 if batched else 1)
    assert (
        "Engine 002: Iteration(7): 1 context requests, 16 context tokens" in messages[0]
    )
    assert "elapsed" not in messages[0]
    assert "encoder inputs: 1, encoder output embeddings: 32" in messages[0]
    assert all("KV cache usage" not in message for message in messages)
    if batched:
        assert "Iteration(8)" in messages[1]
        assert "iteration elapsed time: 1.25 ms (dummy)" in messages[1]


@pytest.fixture
def log_stats_enabled_engine_args():
    """Shared fixture providing common AsyncEngineArgs configuration
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
    """RayPrometheusStatLogger should replace the default PrometheusStatLogger."""
    engine = AsyncLLM.from_engine_args(
        log_stats_enabled_engine_args, stat_loggers=[RayPrometheusStatLogger]
    )
    try:
        assert engine.logger_manager is not None
        assert isinstance(
            engine.logger_manager.stat_loggers[0], RayPrometheusStatLogger
        )
    finally:
        engine.shutdown()
        wait_for_memory_to_settle(
            threshold_ratio=1.0 - log_stats_enabled_engine_args.gpu_memory_utilization
        )


@pytest.mark.asyncio
async def test_async_llm_add_to_default_loggers(log_stats_enabled_engine_args):
    """It's still possible to use custom stat loggers exclusively by passing
    disable_log_stats=True in addition to a list of custom stat loggers.
    """
    # Create engine_args with disable_log_stats=True for this test
    disabled_log_engine_args = copy.deepcopy(log_stats_enabled_engine_args)
    disabled_log_engine_args.disable_log_stats = True

    # Disable default loggers; pass custom stat logger to the constructor
    engine = AsyncLLM.from_engine_args(
        disabled_log_engine_args, stat_loggers=[DummyStatLogger]
    )

    try:
        assert engine.logger_manager is not None
        assert len(engine.logger_manager.stat_loggers) == 2
        # DummyStatLogger is a per-engine logger, so the manager wraps it.
        per_engine_adapter = engine.logger_manager.stat_loggers[0]
        assert isinstance(per_engine_adapter, PerEngineStatLoggerAdapter)
        assert len(per_engine_adapter.per_engine_stat_loggers) == 1
        assert isinstance(
            per_engine_adapter.per_engine_stat_loggers[0],
            DummyStatLogger,
        )

        # log_stats is still True, since custom stat loggers are used
        assert engine.log_stats
    finally:
        engine.shutdown()
        wait_for_memory_to_settle(
            threshold_ratio=1.0 - disabled_log_engine_args.gpu_memory_utilization
        )
