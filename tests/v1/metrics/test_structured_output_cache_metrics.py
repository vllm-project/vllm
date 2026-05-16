# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm.config import ObservabilityConfig
from vllm.v1.metrics.loggers import LoggingStatLogger, PrometheusStatLogger
from vllm.v1.metrics.reader import Counter, get_metrics_snapshot
from vllm.v1.metrics.stats import SchedulerStats, StructuredOutputCacheStats


pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


@pytest.fixture
def stats_vllm_config():
    return SimpleNamespace(
        observability_config=ObservabilityConfig(),
        kv_transfer_config=None,
        speculative_config=None,
        lora_config=None,
        compilation_config=SimpleNamespace(
            cudagraph_mode=None,
            cudagraph_capture_sizes=None,
        ),
        model_config=SimpleNamespace(
            served_model_name="test-model",
            max_model_len=1024,
        ),
    )


def _get_counter_value(metrics: list[Counter], name: str) -> int:
    metric = next(m for m in metrics if m.name == name)
    assert isinstance(metric, Counter)
    return metric.value


def test_logging_stat_logger_logs_structured_output_cache_hit_rate(
    stats_vllm_config,
    monkeypatch,
):
    logger = LoggingStatLogger(stats_vllm_config)
    scheduler_stats = SchedulerStats(
        structured_output_cache_stats=StructuredOutputCacheStats(
            requests=5,
            queries=5,
            hits=4,
        )
    )

    logger.record(scheduler_stats=scheduler_stats, iteration_stats=None)
    info_log = Mock()
    debug_log = Mock()
    monkeypatch.setattr("vllm.v1.metrics.loggers.logger.info", info_log)
    monkeypatch.setattr("vllm.v1.metrics.loggers.logger.debug", debug_log)

    logger.log()

    assert info_log.called or debug_log.called
    log_call = info_log.call_args if info_log.called else debug_log.call_args
    assert log_call is not None
    assert "Structured output cache hit rate" in log_call.args[0]


def test_prometheus_stat_logger_records_structured_output_cache_counters(
    stats_vllm_config,
):
    logger = PrometheusStatLogger(stats_vllm_config)
    scheduler_stats = SchedulerStats(
        structured_output_cache_stats=StructuredOutputCacheStats(
            requests=3,
            queries=3,
            hits=2,
        )
    )

    logger.record(scheduler_stats=scheduler_stats, iteration_stats=None)

    metrics = get_metrics_snapshot()
    assert _get_counter_value(metrics, "vllm:structured_output_cache_queries") == 3
    assert _get_counter_value(metrics, "vllm:structured_output_cache_hits") == 2