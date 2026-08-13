# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import prometheus_client

from vllm.distributed.eplb.metrics import EplbMetricsSnapshot
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.metrics.loggers import PrometheusStatLogger
from vllm.v1.metrics.stats import PrefixCacheStats, SchedulerStats


class NoopProm:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def observe(self, *args, **kwargs) -> None:
        pass


def make_prometheus_logger(monkeypatch, *, pipeline_parallel_size: int = 1):
    registry = prometheus_client.CollectorRegistry()

    class TestGauge:
        def __new__(cls, *args, **kwargs):
            return prometheus_client.Gauge(*args, registry=registry, **kwargs)

    class TestCounter:
        def __new__(cls, *args, **kwargs):
            return prometheus_client.Counter(*args, registry=registry, **kwargs)

    class TestHistogram:
        def __new__(cls, *args, **kwargs):
            return prometheus_client.Histogram(*args, registry=registry, **kwargs)

    monkeypatch.setattr(PrometheusStatLogger, "_gauge_cls", TestGauge)
    monkeypatch.setattr(PrometheusStatLogger, "_counter_cls", TestCounter)
    monkeypatch.setattr(PrometheusStatLogger, "_histogram_cls", TestHistogram)
    monkeypatch.setattr(PrometheusStatLogger, "_spec_decoding_cls", NoopProm)
    monkeypatch.setattr(PrometheusStatLogger, "_kv_connector_cls", NoopProm)
    monkeypatch.setattr(PrometheusStatLogger, "_perf_metrics_cls", NoopProm)

    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            served_model_name="served-model",
            max_model_len=1024,
            is_diffusion=False,
        ),
        parallel_config=SimpleNamespace(
            enable_eplb=True,
            pipeline_parallel_size=pipeline_parallel_size,
        ),
        observability_config=SimpleNamespace(
            show_hidden_metrics=False,
            kv_cache_metrics=False,
        ),
        speculative_config=None,
        kv_transfer_config=None,
        lora_config=None,
    )
    logger = PrometheusStatLogger(vllm_config, engine_indexes=[0, 1])
    return logger, registry


def test_eplb_metrics_are_recorded_once_without_engine_label(monkeypatch):
    logger, registry = make_prometheus_logger(monkeypatch)
    snapshot = EplbMetricsSnapshot(
        rebalancing=True,
        rebalance_events=3,
    )
    scheduler_stats = SchedulerStats(eplb_metrics=snapshot)
    labels = {"model_name": "served-model"}

    logger.record(scheduler_stats, None, engine_idx=1)
    assert registry.get_sample_value("vllm:eplb_rebalancing", labels) == 0.0
    assert registry.get_sample_value("vllm:eplb_rebalance_events_total", labels) == 0.0

    logger.record(scheduler_stats, None, engine_idx=0)

    assert registry.get_sample_value("vllm:eplb_rebalancing", labels) == 1.0
    assert registry.get_sample_value("vllm:eplb_rebalance_events_total", labels) == 3.0
    rendered_metrics = prometheus_client.generate_latest(registry).decode()
    eplb_lines = [
        line for line in rendered_metrics.splitlines() if line.startswith("vllm:eplb_")
    ]
    assert eplb_lines
    assert all("engine=" not in line for line in eplb_lines)
    assert "vllm:eplb_avg_tokens_per_rank" not in rendered_metrics
    assert "vllm:eplb_max_tokens_per_rank" not in rendered_metrics


def test_eplb_counter_adds_worker_event_deltas(monkeypatch):
    logger, registry = make_prometheus_logger(monkeypatch)
    snapshot = EplbMetricsSnapshot(
        rebalancing=True,
        rebalance_events=3,
    )
    scheduler_stats = SchedulerStats(eplb_metrics=snapshot)
    labels = {"model_name": "served-model"}

    logger.record(scheduler_stats, None, engine_idx=0)
    snapshot.rebalance_events = 0
    logger.record(scheduler_stats, None, engine_idx=0)
    assert registry.get_sample_value("vllm:eplb_rebalance_events_total", labels) == 3.0

    snapshot.rebalancing = False
    snapshot.rebalance_events = 1
    logger.record(scheduler_stats, None, engine_idx=0)

    assert registry.get_sample_value("vllm:eplb_rebalancing", labels) == 0.0
    assert registry.get_sample_value("vllm:eplb_rebalance_events_total", labels) == 4.0


def test_eplb_metrics_fail_closed_with_pipeline_parallelism(monkeypatch):
    logger, registry = make_prometheus_logger(
        monkeypatch,
        pipeline_parallel_size=2,
    )
    snapshot = EplbMetricsSnapshot(rebalance_events=1)

    logger.record(
        SchedulerStats(eplb_metrics=snapshot),
        None,
        engine_idx=0,
    )

    assert logger.eplb_metrics_enabled is False
    assert (
        registry.get_sample_value(
            "vllm:eplb_rebalance_events_total",
            {"model_name": "served-model"},
        )
        is None
    )


def test_scheduler_includes_eplb_metrics_in_stats():
    snapshot = EplbMetricsSnapshot()
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.log_stats = True
    scheduler.kv_cache_manager = SimpleNamespace(
        make_prefix_cache_stats=lambda: PrefixCacheStats(),
        usage=0.25,
    )
    scheduler.connector_prefix_cache_stats = None
    scheduler.kv_metrics_collector = None
    scheduler.running = []
    scheduler.waiting = []
    scheduler.skipped_waiting = []

    stats = Scheduler.make_stats(scheduler, eplb_metrics=snapshot)

    assert stats is not None
    assert stats.eplb_metrics is snapshot
