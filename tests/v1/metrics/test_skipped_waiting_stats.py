# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for the num_skipped_waiting_reqs metric introduced in:
  vllm/v1/metrics/stats.py          -- SchedulerStats dataclass field
  vllm/v1/metrics/loggers.py        -- PrometheusStatLogger gauge +
                                       AggregatedLoggingStatLogger aggregation +
                                       LoggingStatLogger log-line

These tests are pure Python (no torch, no GPU) and run in the normal
CI tier.  They exercise:
  1.  SchedulerStats dataclass semantics
  2.  AggregatedLoggingStatLogger.aggregate_scheduler_stats() summation
  3.  LoggingStatLogger.log() conditional "Deferred:" line
  4.  PrometheusStatLogger gauge registration and record() wiring
"""

import logging
from unittest.mock import MagicMock, call, patch

import pytest

from vllm.v1.metrics.stats import SchedulerStats


# ---------------------------------------------------------------------------
# 1. SchedulerStats dataclass
# ---------------------------------------------------------------------------


def test_scheduler_stats_default_zero():
    """New field must default to 0 so existing callers need no changes."""
    stats = SchedulerStats()
    assert stats.num_skipped_waiting_reqs == 0


def test_scheduler_stats_explicit_construction():
    """Field must be settable at construction time like all other stats."""
    stats = SchedulerStats(
        num_running_reqs=4,
        num_waiting_reqs=10,
        num_skipped_waiting_reqs=3,
    )
    assert stats.num_running_reqs == 4
    assert stats.num_waiting_reqs == 10
    assert stats.num_skipped_waiting_reqs == 3


def test_scheduler_stats_skipped_is_subset_of_waiting():
    """
    By definition skipped_waiting is a sub-population of waiting
    (skipped requests are still counted in num_waiting_reqs).
    This test documents that invariant explicitly.
    """
    waiting = 7
    skipped = 3
    stats = SchedulerStats(
        num_waiting_reqs=waiting,
        num_skipped_waiting_reqs=skipped,
    )
    # The "pure backlog" count (fresh arrivals) must be non-negative.
    assert stats.num_waiting_reqs - stats.num_skipped_waiting_reqs >= 0


# ---------------------------------------------------------------------------
# 2. AggregatedLoggingStatLogger — multi-engine summation
# ---------------------------------------------------------------------------


def _make_minimal_vllm_config(engine_indexes: list[int]):
    """
    Build the smallest possible VllmConfig needed to instantiate a
    LoggingStatLogger without touching torch or Prometheus.
    We mock the sub-objects that LoggingStatLogger probes at init time.
    """
    from unittest.mock import MagicMock

    cfg = MagicMock()
    cfg.model_config.served_model_name = "test-model"
    cfg.model_config.max_model_len = 8192
    cfg.cache_config.num_gpu_blocks = 1024
    cfg.kv_transfer_config = None
    cfg.observability_config.enable_mfu_metrics = False
    cfg.observability_config.cudagraph_metrics = False
    cfg.speculative_config = None
    return cfg


@pytest.fixture()
def aggregated_logger():
    """
    Create an AggregatedLoggingStatLogger with two fake engines using
    a fully-mocked VllmConfig so no torch/GPU is required.
    """
    from vllm.v1.metrics.loggers import AggregatedLoggingStatLogger

    vllm_config = _make_minimal_vllm_config([0, 1])
    return AggregatedLoggingStatLogger(vllm_config, engine_indexes=[0, 1])


def test_aggregated_logger_sums_skipped_waiting(aggregated_logger):
    """
    aggregate_scheduler_stats() must sum num_skipped_waiting_reqs across
    all engines, mirroring how num_waiting_reqs and num_running_reqs are
    aggregated.
    """
    # Simulate two DP engines each reporting different skipped counts.
    aggregated_logger.last_scheduler_stats_dict[0] = SchedulerStats(
        num_running_reqs=2,
        num_waiting_reqs=5,
        num_skipped_waiting_reqs=2,
    )
    aggregated_logger.last_scheduler_stats_dict[1] = SchedulerStats(
        num_running_reqs=3,
        num_waiting_reqs=8,
        num_skipped_waiting_reqs=4,
    )

    aggregated_logger.aggregate_scheduler_stats()

    agg = aggregated_logger.last_scheduler_stats
    # Sanity-check existing fields are still summed correctly.
    assert agg.num_running_reqs == 5   # 2 + 3
    assert agg.num_waiting_reqs == 13  # 5 + 8
    # The new field must also be summed, not averaged or dropped.
    assert agg.num_skipped_waiting_reqs == 6  # 2 + 4


def test_aggregated_logger_zero_when_no_skipped(aggregated_logger):
    """
    If no engine has deferred requests the aggregate must be 0 (not None
    or some default that could break comparisons in log()).
    """
    aggregated_logger.last_scheduler_stats_dict[0] = SchedulerStats(
        num_waiting_reqs=3,
        num_skipped_waiting_reqs=0,
    )
    aggregated_logger.last_scheduler_stats_dict[1] = SchedulerStats(
        num_waiting_reqs=2,
        num_skipped_waiting_reqs=0,
    )

    aggregated_logger.aggregate_scheduler_stats()

    assert aggregated_logger.last_scheduler_stats.num_skipped_waiting_reqs == 0


# ---------------------------------------------------------------------------
# 3. LoggingStatLogger — "Deferred:" line in log output
# ---------------------------------------------------------------------------


@pytest.fixture()
def logging_logger():
    """LoggingStatLogger with a mocked VllmConfig (no torch needed)."""
    from vllm.v1.metrics.loggers import LoggingStatLogger

    return LoggingStatLogger(_make_minimal_vllm_config([0]))


def test_log_deferred_line_omitted_when_zero(logging_logger, caplog):
    """
    The "Deferred:" token must NOT appear when num_skipped_waiting_reqs==0
    so idle-system logs stay clean.
    """
    logging_logger.last_scheduler_stats = SchedulerStats(
        num_running_reqs=2,
        num_waiting_reqs=4,
        num_skipped_waiting_reqs=0,
    )

    with caplog.at_level(logging.DEBUG):
        logging_logger.log()

    full_output = " ".join(caplog.messages)
    assert "Deferred" not in full_output


def test_log_deferred_line_present_when_nonzero(logging_logger, caplog):
    """
    The "Deferred:" token must appear in the log line when there are
    constraint-blocked requests, so operators can see it at a glance.
    """
    logging_logger.last_scheduler_stats = SchedulerStats(
        num_running_reqs=2,
        num_waiting_reqs=7,
        num_skipped_waiting_reqs=3,
    )

    with caplog.at_level(logging.DEBUG):
        logging_logger.log()

    full_output = " ".join(caplog.messages)
    assert "Deferred" in full_output
    # Verify the count itself appears in the output.
    assert "3" in full_output


# ---------------------------------------------------------------------------
# 4. PrometheusStatLogger — gauge registration and record() wiring
# ---------------------------------------------------------------------------


class _MockGauge:
    """
    Minimal stand-in for prometheus_client.Gauge.
    Captures the constructor kwargs and the most-recent .set() call so
    tests can assert on them without touching the real Prometheus registry.
    """

    def __init__(self, name, documentation, multiprocess_mode, labelnames, **_):
        self.name = name
        self.documentation = documentation
        self._label_instances: dict[tuple, "_MockGauge"] = {}
        self._last_set_value: float | None = None

    def labels(self, **kwargs) -> "_MockGauge":
        # Return a child instance keyed by the label values so that
        # create_metric_per_engine() gets distinct objects per engine.
        key = tuple(sorted(kwargs.items()))
        if key not in self._label_instances:
            child = _MockGauge.__new__(_MockGauge)
            child.name = self.name
            child._label_instances = {}
            child._last_set_value = None
            self._label_instances[key] = child
        return self._label_instances[key]

    def set(self, value: float) -> None:
        self._last_set_value = value


class _MockCounter:
    """Minimal stand-in for prometheus_client.Counter."""

    def __init__(self, name, documentation, labelnames, **_):
        self.name = name
        self._label_instances: dict[tuple, "_MockCounter"] = {}

    def labels(self, *args, **kwargs) -> "_MockCounter":
        key = tuple(args) + tuple(sorted(kwargs.items()))
        if key not in self._label_instances:
            child = _MockCounter.__new__(_MockCounter)
            child.name = self.name
            child._label_instances = {}
            child._last_inc = 0.0
            self._label_instances[key] = child
        return self._label_instances[key]

    def inc(self, amount: float = 1) -> None:
        self._last_inc = amount


class _MockHistogram:
    """Minimal stand-in for prometheus_client.Histogram."""

    def __init__(self, name, documentation, buckets, labelnames, **_):
        self.name = name
        self._label_instances: dict[tuple, "_MockHistogram"] = {}

    def labels(self, *args, **kwargs) -> "_MockHistogram":
        key = tuple(args) + tuple(sorted(kwargs.items()))
        if key not in self._label_instances:
            child = _MockHistogram.__new__(_MockHistogram)
            child.name = self.name
            child._label_instances = {}
            self._label_instances[key] = child
        return self._label_instances[key]

    def observe(self, value: float) -> None:
        pass


@pytest.fixture()
def prom_logger():
    """
    PrometheusStatLogger with mock Gauge/Counter/Histogram so no real
    Prometheus registry is touched.  Uses the class-level attribute
    override pattern that PrometheusStatLogger already supports.
    """
    from vllm.v1.metrics.loggers import PrometheusStatLogger
    from vllm.v1.metrics.prometheus import unregister_vllm_metrics

    # Patch unregister so we don't hit the real global registry.
    with patch("vllm.v1.metrics.loggers.unregister_vllm_metrics"):

        class MockPrometheusStatLogger(PrometheusStatLogger):
            # PrometheusStatLogger uses these class attributes for all metric
            # construction; replacing them redirects every Gauge/Counter/
            # Histogram creation to our lightweight mocks.
            _gauge_cls = _MockGauge
            _counter_cls = _MockCounter
            _histogram_cls = _MockHistogram

            # These sub-loggers also create Prometheus objects; stub them out.
            _spec_decoding_cls = MagicMock(return_value=MagicMock())
            _kv_connector_cls = MagicMock(return_value=MagicMock())
            _perf_metrics_cls = MagicMock(return_value=MagicMock())

        vllm_config = _make_minimal_vllm_config([0])
        # Some fields PrometheusStatLogger probes that aren't in the minimal config.
        vllm_config.observability_config.show_hidden_metrics = False
        vllm_config.observability_config.kv_cache_metrics = False

        return MockPrometheusStatLogger(vllm_config, engine_indexes=[0])


def test_prometheus_skipped_waiting_gauge_registered(prom_logger):
    """
    The new gauge must be registered on the logger so it can be set.
    A missing attribute would mean record() silently does nothing.
    """
    assert hasattr(prom_logger, "gauge_scheduler_skipped_waiting"), (
        "gauge_scheduler_skipped_waiting missing from PrometheusStatLogger — "
        "the gauge was never registered in __init__"
    )
    # The dict must contain an entry for engine 0.
    assert 0 in prom_logger.gauge_scheduler_skipped_waiting


def test_prometheus_skipped_waiting_gauge_set_on_record(prom_logger):
    """
    record() must call .set() on the skipped_waiting gauge with the value
    from scheduler_stats.num_skipped_waiting_reqs.
    """
    stats = SchedulerStats(
        num_running_reqs=2,
        num_waiting_reqs=9,
        num_skipped_waiting_reqs=4,
    )

    prom_logger.record(
        scheduler_stats=stats,
        iteration_stats=None,
        engine_idx=0,
    )

    gauge = prom_logger.gauge_scheduler_skipped_waiting[0]
    assert gauge._last_set_value == 4, (
        f"Expected gauge to be set to 4 (num_skipped_waiting_reqs), "
        f"got {gauge._last_set_value}"
    )


def test_prometheus_waiting_gauge_unchanged(prom_logger):
    """
    The existing vllm:num_requests_waiting gauge must still reflect the
    combined waiting count (backward-compat guarantee).
    """
    stats = SchedulerStats(
        num_running_reqs=1,
        num_waiting_reqs=10,
        num_skipped_waiting_reqs=3,
    )

    prom_logger.record(
        scheduler_stats=stats,
        iteration_stats=None,
        engine_idx=0,
    )

    # num_requests_waiting must still equal the combined count.
    waiting_gauge = prom_logger.gauge_scheduler_waiting[0]
    assert waiting_gauge._last_set_value == 10


def test_prometheus_zero_skipped(prom_logger):
    """
    When no requests are deferred the gauge must be set to 0, not omitted.
    Omitting it would leave a stale non-zero value from a previous scrape.
    """
    stats = SchedulerStats(
        num_running_reqs=5,
        num_waiting_reqs=5,
        num_skipped_waiting_reqs=0,
    )

    prom_logger.record(
        scheduler_stats=stats,
        iteration_stats=None,
        engine_idx=0,
    )

    gauge = prom_logger.gauge_scheduler_skipped_waiting[0]
    assert gauge._last_set_value == 0
