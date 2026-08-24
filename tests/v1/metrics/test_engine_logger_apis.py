# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.plugins.vllm_add_dummy_stat_logger.dummy_stat_logger.dummy_stat_logger import (  # noqa E501
    DummyStatLogger,
)
from vllm.v1.engine.async_llm import AsyncEngineArgs, AsyncLLM
from vllm.v1.metrics.loggers import (
    ITERATION_PHASE_DECODE,
    ITERATION_PHASE_PREFILL,
    LoggingStatLogger,
    PrometheusStatLogger,
    StatLoggerRequirements,
    get_stat_logger_requirements,
)
from vllm.v1.metrics.ray_wrappers import RayPrometheusStatLogger
from vllm.v1.metrics.stats import SchedulerIterationDetails, SchedulerStats


class IterationDetailsStatLogger(DummyStatLogger):
    @classmethod
    def get_requirements(cls, vllm_config) -> StatLoggerRequirements:
        return StatLoggerRequirements(iteration_details=True)


class InvalidRequirementsStatLogger(DummyStatLogger):
    @classmethod
    def get_requirements(cls, vllm_config):
        return object()


def make_vllm_config(*, enable_prometheus_iteration_metrics=False):
    return SimpleNamespace(
        observability_config=SimpleNamespace(
            enable_prometheus_iteration_metrics=enable_prometheus_iteration_metrics,
        )
    )


def make_prometheus_iteration_logger():
    stat_logger = object.__new__(PrometheusStatLogger)
    stat_logger.iteration_metrics_enabled = True
    request_histograms = {
        phase: {0: Mock()}
        for phase in (ITERATION_PHASE_PREFILL, ITERATION_PHASE_DECODE)
    }
    token_histograms = {
        phase: {0: Mock()}
        for phase in (ITERATION_PHASE_PREFILL, ITERATION_PHASE_DECODE)
    }
    stat_logger.histogram_iteration_requests = request_histograms
    stat_logger.histogram_iteration_tokens_by_phase = token_histograms
    return stat_logger, request_histograms, token_histograms


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


def test_get_stat_logger_requirements_merges_plugin_requests():
    requirements = get_stat_logger_requirements(
        [DummyStatLogger, IterationDetailsStatLogger], make_vllm_config()
    )

    assert requirements == StatLoggerRequirements(iteration_details=True)


def test_get_stat_logger_requirements_keeps_existing_plugins_compatible():
    requirements = get_stat_logger_requirements([DummyStatLogger], make_vllm_config())

    assert requirements == StatLoggerRequirements()


def test_get_stat_logger_requirements_rejects_invalid_result():
    with pytest.raises(
        TypeError,
        match=r"get_requirements\(\) must return StatLoggerRequirements",
    ):
        get_stat_logger_requirements(
            [InvalidRequirementsStatLogger], make_vllm_config()
        )


@pytest.mark.parametrize("enabled", [False, True])
def test_prometheus_requests_iteration_details_only_when_enabled(enabled):
    requirements = PrometheusStatLogger.get_requirements(
        make_vllm_config(enable_prometheus_iteration_metrics=enabled)
    )

    assert requirements.iteration_details is enabled


def test_prometheus_records_phase_aware_iteration_distributions():
    stat_logger, request_histograms, token_histograms = (
        make_prometheus_iteration_logger()
    )
    scheduler_stats = SchedulerStats(
        iteration_details=SchedulerIterationDetails(
            iteration_index=1,
            num_ctx_requests=2,
            num_ctx_tokens=3,
            num_generation_requests=4,
            num_generation_tokens=5,
            elapsed_ms=6,
        )
    )

    stat_logger._record_iteration_metrics(scheduler_stats, 0)

    request_histograms[ITERATION_PHASE_PREFILL][0].observe.assert_called_once_with(2)
    token_histograms[ITERATION_PHASE_PREFILL][0].observe.assert_called_once_with(3)
    request_histograms[ITERATION_PHASE_DECODE][0].observe.assert_called_once_with(4)
    token_histograms[ITERATION_PHASE_DECODE][0].observe.assert_called_once_with(5)


@pytest.mark.parametrize(
    "details",
    [None, SchedulerIterationDetails(0, 0, 0, 0, 0, 0, is_dummy=True)],
)
def test_prometheus_skips_non_model_iterations(details):
    stat_logger, request_histograms, token_histograms = (
        make_prometheus_iteration_logger()
    )

    stat_logger._record_iteration_metrics(SchedulerStats(iteration_details=details), 0)

    for histogram in (*request_histograms.values(), *token_histograms.values()):
        histogram[0].observe.assert_not_called()


def test_prometheus_disabled_path_does_not_access_iteration_details():
    stat_logger = object.__new__(PrometheusStatLogger)
    stat_logger.iteration_metrics_enabled = False

    stat_logger._record_iteration_metrics(SchedulerStats(), 0)


def test_iteration_collection_does_not_enable_iteration_logging(monkeypatch):
    log_info = Mock()
    monkeypatch.setattr("vllm.v1.metrics.loggers.logger.info", log_info)
    stat_logger = object.__new__(LoggingStatLogger)
    stat_logger.vllm_config = SimpleNamespace(
        observability_config=SimpleNamespace(
            enable_logging_iteration_details=False,
        )
    )
    scheduler_stats = SchedulerStats(
        iteration_details=SchedulerIterationDetails(
            iteration_index=1,
            num_ctx_requests=2,
            num_ctx_tokens=3,
            num_generation_requests=4,
            num_generation_tokens=5,
            elapsed_ms=6,
        )
    )

    LoggingStatLogger._log_iteration_details(stat_logger, scheduler_stats, 0)

    log_info.assert_not_called()
