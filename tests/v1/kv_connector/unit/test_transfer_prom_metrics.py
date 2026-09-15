# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the per-transfer Prometheus plumbing.

`KVTransferPromMetrics` owns `observe()` for every connector that records one
row per transfer, so its contract is covered once here through a synthetic
subclass rather than once per connector. `NixlPromMetrics` is covered alongside
it as the reference implementation the base class was factored out of.
"""

from types import SimpleNamespace
from typing import Any

import pytest
from prometheus_client import REGISTRY, Counter, Gauge, Histogram

from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVTransferPromMetrics,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.stats import NixlPromMetrics
from vllm.v1.metrics.prometheus import unregister_vllm_metrics

MODEL_NAME = "test-model"
LABELNAMES = ["model_name", "engine"]
METRIC_TYPES = {Gauge: Gauge, Counter: Counter, Histogram: Histogram}


@pytest.fixture(autouse=True)
def _clean_registry():
    """vLLM metrics share the process-wide registry, so drop them per test."""
    unregister_vllm_metrics()
    yield
    unregister_vllm_metrics()


def _vllm_config() -> Any:
    return SimpleNamespace(kv_transfer_config=SimpleNamespace())


def _labels(engine_idx: int = 0) -> dict[str, str]:
    return {"model_name": MODEL_NAME, "engine": str(engine_idx)}


def _sample(series: str, engine_idx: int = 0) -> float | None:
    return REGISTRY.get_sample_value(series, _labels(engine_idx))


def _bucket(le: str) -> float | None:
    return REGISTRY.get_sample_value(
        "vllm:test_transfer_seconds_bucket", {**_labels(), "le": le}
    )


class _PromMetricsUnderTest(KVTransferPromMetrics):
    """Synthetic connector exporting one histogram and one counter."""

    def __init__(self, num_engines: int = 1):
        super().__init__(
            vllm_config=_vllm_config(),  # type: ignore[arg-type]
            metric_types=METRIC_TYPES,
            labelnames=LABELNAMES,
            per_engine_labelvalues={
                idx: [MODEL_NAME, str(idx)] for idx in range(num_engines)
            },
        )
        self.histogram_latency = self.declare_histogram(
            name="vllm:test_transfer_seconds",
            documentation="Test histogram.",
            stats_key="transfer_duration",
            buckets=(0.1, 1.0),
        )
        self.counter_failures = self.declare_counter(
            name="vllm:test_failures",
            documentation="Test counter.",
            stats_key="num_failures",
        )


def test_declared_histograms_export_every_recorded_transfer():
    metrics = _PromMetricsUnderTest()

    metrics.observe({"transfer_duration": [0.05, 0.5, 1.5], "num_failures": []})

    assert _sample("vllm:test_transfer_seconds_count") == 3
    assert _sample("vllm:test_transfer_seconds_sum") == 0.05 + 0.5 + 1.5
    # The declared edges are the ones users get: 0.05 falls in the 0.1 bucket,
    # 0.05 and 0.5 in the 1.0 bucket, and all three in +Inf.
    assert _bucket("0.1") == 1
    assert _bucket("1.0") == 2


def test_declared_counters_export_the_event_count():
    metrics = _PromMetricsUnderTest()

    metrics.observe({"transfer_duration": [], "num_failures": [1] * 7})

    assert _sample("vllm:test_failures_total") == 7


def test_observe_routes_samples_to_the_given_engine():
    metrics = _PromMetricsUnderTest(num_engines=2)

    metrics.observe({"transfer_duration": [0.5], "num_failures": []}, engine_idx=1)

    assert _sample("vllm:test_transfer_seconds_count", engine_idx=0) == 0
    assert _sample("vllm:test_transfer_seconds_count", engine_idx=1) == 1


def test_observe_rejects_a_snapshot_missing_a_declared_key():
    metrics = _PromMetricsUnderTest()

    with pytest.raises(KeyError):
        metrics.observe({"num_failures": []}, engine_idx=0)
    with pytest.raises(KeyError):
        metrics.observe({"transfer_duration": [0.5]}, engine_idx=0)


NIXL_HISTOGRAMS = {
    "vllm:nixl_xfer_time_seconds": "transfer_duration",
    "vllm:nixl_post_time_seconds": "post_duration",
    "vllm:nixl_bytes_transferred": "bytes_transferred",
    "vllm:nixl_num_descriptors": "num_descriptors",
}

NIXL_COUNTERS = {
    "vllm:nixl_num_failed_transfers": "num_failed_transfers",
    "vllm:nixl_num_failed_notifications": "num_failed_notifications",
    "vllm:nixl_num_kv_expired_reqs": "num_kv_expired_reqs",
}


def _nixl_prom_metrics() -> NixlPromMetrics:
    return NixlPromMetrics(
        vllm_config=_vllm_config(),  # type: ignore[arg-type]
        metric_types=METRIC_TYPES,
        labelnames=LABELNAMES,
        per_engine_labelvalues={0: [MODEL_NAME, "0"]},
    )


def test_nixl_histograms_export_their_observed_values():
    prom_metrics = _nixl_prom_metrics()
    # One distinct observation per stats key, so a series fed by the wrong key
    # cannot pass.
    observations = {
        stats_key: [float(index)]
        for index, stats_key in enumerate(NIXL_HISTOGRAMS.values())
    }
    snapshot: dict[str, list[float | int]] = {
        stats_key: [] for stats_key in NIXL_COUNTERS.values()
    }
    snapshot.update(observations)

    prom_metrics.observe(snapshot, engine_idx=0)

    for series, stats_key in NIXL_HISTOGRAMS.items():
        assert _sample(f"{series}_count") == 1
        assert _sample(f"{series}_sum") == observations[stats_key][0]


def test_nixl_counters_export_their_event_counts():
    prom_metrics = _nixl_prom_metrics()
    counts = {
        stats_key: index + 1 for index, stats_key in enumerate(NIXL_COUNTERS.values())
    }
    snapshot: dict[str, list[float | int]] = {
        stats_key: [] for stats_key in NIXL_HISTOGRAMS.values()
    }
    snapshot.update({stats_key: [1] * count for stats_key, count in counts.items()})

    prom_metrics.observe(snapshot, engine_idx=0)

    for series, stats_key in NIXL_COUNTERS.items():
        assert _sample(f"{series}_total") == counts[stats_key]
