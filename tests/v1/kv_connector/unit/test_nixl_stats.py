# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prometheus metrics for the NIXL connector transfer stats."""

from typing import Any

import pytest
from prometheus_client import REGISTRY, Counter, Gauge, Histogram

from vllm.config import KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import NixlConnector
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.stats import (
    NixlKVConnectorStats,
    NixlPromMetrics,
)
from vllm.v1.metrics.prometheus import unregister_vllm_metrics

TEST_MODEL = "test-model"


@pytest.fixture(autouse=True)
def _clean_registry():
    """VLLM metrics share the process-wide registry, so drop them per test."""
    unregister_vllm_metrics()
    yield
    unregister_vllm_metrics()


def _labels(engine_idx: int = 0) -> dict[str, str]:
    return {"model_name": TEST_MODEL, "engine": str(engine_idx)}


def _sample(series: str, engine_idx: int = 0) -> float | None:
    return REGISTRY.get_sample_value(series, _labels(engine_idx))


def _build_prom_metrics(
    per_engine_labelvalues: dict[int, list[object]],
) -> NixlPromMetrics:
    prom_metrics = NixlConnector.build_prom_metrics(
        vllm_config=VllmConfig(
            kv_transfer_config=KVTransferConfig(
                kv_connector="NixlConnector", kv_role="kv_producer"
            )
        ),
        metric_types={Gauge: Gauge, Counter: Counter, Histogram: Histogram},
        labelnames=["model_name", "engine"],
        per_engine_labelvalues=per_engine_labelvalues,
    )
    assert isinstance(prom_metrics, NixlPromMetrics)
    return prom_metrics


def _telemetry(
    xfer_duration_s: float, post_duration_s: float, total_bytes: int, num_descs: int
) -> Any:
    class AttributeDict(dict):
        __slots__ = ()
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

    # nixlXferTelemetry is read-only and NIXL may not be installed, so feed the
    # stats container the attributes it reads.
    return AttributeDict(
        xferDuration=xfer_duration_s * 1e6,
        postDuration=post_duration_s * 1e6,
        totalBytes=total_bytes,
        descCount=num_descs,
    )


def _recorded_stats() -> dict[str, list[float | int]]:
    """Drive the real stats container, so the keys observed below are the ones
    the connector actually produces rather than keys restated in this test."""
    stats = NixlKVConnectorStats()
    stats.record_transfer(_telemetry(0.01, 0.02, total_bytes=1024, num_descs=4))
    stats.record_transfer(_telemetry(2.0, 3.0, total_bytes=2048, num_descs=6))
    stats.record_failed_transfer()
    stats.record_failed_handshake()
    stats.record_failed_notification()
    stats.record_kv_expired_req()
    stats.record_notification_after_expiry()
    return stats.clone_and_reset().to_dict()


@pytest.fixture
def exported_series():
    """Record one interval from the real stats container into the real registry."""
    _build_prom_metrics({0: [TEST_MODEL, "0"]}).observe(_recorded_stats(), engine_idx=0)
    yield


@pytest.mark.usefixtures("exported_series")
def test_transfer_histograms_are_exported():
    assert _sample("vllm:nixl_xfer_time_seconds_count") == 2
    assert _sample("vllm:nixl_xfer_time_seconds_sum") == pytest.approx(2.01)
    assert _sample("vllm:nixl_post_time_seconds_sum") == pytest.approx(3.02)
    assert _sample("vllm:nixl_bytes_transferred_sum") == 1024 + 2048
    assert _sample("vllm:nixl_num_descriptors_sum") == 4 + 6


@pytest.mark.usefixtures("exported_series")
def test_failure_counters_are_exported():
    # Transfer, handshake and notification failures share one counter; KV
    # expiry and late notifications keep their own series.
    assert _sample("vllm:nixl_num_failed_transfers_total") == 3
    assert _sample("vllm:nixl_num_failed_notifications_total") == 1
    assert _sample("vllm:nixl_num_kv_expired_reqs_total") == 1
    assert _sample("vllm:nixl_num_notifications_after_expiry_total") == 1


@pytest.mark.usefixtures("exported_series")
def test_only_the_post_time_histogram_has_a_sub_millisecond_bucket():
    """Post time is measured separately from the transfer itself, so it is the
    only timing series with an edge below 5ms."""
    post_time_edges = REGISTRY.get_sample_value(
        "vllm:nixl_post_time_seconds_bucket", {**_labels(), "le": "0.001"}
    )
    xfer_time_edges = REGISTRY.get_sample_value(
        "vllm:nixl_xfer_time_seconds_bucket", {**_labels(), "le": "0.001"}
    )

    assert post_time_edges == 0
    assert xfer_time_edges is None
