# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Rust-frontend-gated ``_metrics_descriptor`` emission."""

from types import SimpleNamespace
from typing import Any

import pytest
from prometheus_client import Counter, Gauge, Histogram

from vllm.distributed.kv_transfer.kv_connector.v1.hf3fs.hf3fs_connector import (
    HF3FSKVConnectorStats,
    HF3FSPromMetrics,
)
from vllm.distributed.kv_transfer.kv_connector.v1.hisparse.stats import (
    _HISPARSE_COUNTERS,
    HiSparseKVConnectorStats,
    build_hisparse_metrics_descriptor,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics_descriptor import (
    INC_BY_F64,
    INC_BY_SUM_U64,
    INC_BY_U64,
    METRICS_DESCRIPTOR_KEY,
    OBSERVE_EACH_F64,
    SET_F64,
    reset_metrics_descriptor_emission_for_tests,
    strip_metrics_descriptor,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
    OffloadPromMetrics,
    _TransferMetricName,
    build_offloading_metrics_descriptor,
    get_connector_metric_definitions,
)
from vllm.v1.kv_offload.base import (
    OffloadingCounterMetadata,
    OffloadingGaugeMetadata,
    OffloadingHistogramMetadata,
)
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec


class _RecordingMetric:
    """Stand-in Prometheus metric that records constructor kwargs and kind."""

    kind: str = "unknown"
    _created: list["_RecordingMetric"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.kind = type(self).kind
        type(self)._created.append(self)

    def labels(self, *_args):
        return self


class _RecordingHistogram(_RecordingMetric):
    kind = "histogram"
    _created: list["_RecordingMetric"] = []


class _RecordingCounter(_RecordingMetric):
    kind = "counter"
    _created: list["_RecordingMetric"] = []


class _RecordingGauge(_RecordingMetric):
    kind = "gauge"
    _created: list["_RecordingMetric"] = []


def _reset_recording_metrics() -> None:
    _RecordingHistogram._created = []
    _RecordingCounter._created = []
    _RecordingGauge._created = []


class _FakeOffloadVllmConfig:
    def __init__(self, store_threshold: int = 2):
        self.kv_transfer_config = SimpleNamespace(
            kv_connector_extra_config={"store_threshold": store_threshold}
        )


@pytest.mark.parametrize(
    ("stats_cls", "connector_id", "build_descriptor"),
    [
        (
            OffloadingConnectorStats,
            "OffloadingConnector",
            build_offloading_metrics_descriptor,
        ),
        (
            HiSparseKVConnectorStats,
            "HiSparseConnector",
            build_hisparse_metrics_descriptor,
        ),
        (
            HF3FSKVConnectorStats,
            "HF3FSKVConnector",
            HF3FSKVConnectorStats._metrics_descriptor,
        ),
    ],
)
@pytest.mark.parametrize("rust_frontend", [False, True])
def test_metrics_descriptor_follows_rust_frontend_flag(
    monkeypatch, stats_cls, connector_id, build_descriptor, rust_frontend
):
    """Shared helper gates one-shot emission for every adopting stats class."""
    reset_metrics_descriptor_emission_for_tests()
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.metrics_descriptor.envs."
        "VLLM_USE_RUST_FRONTEND",
        rust_frontend,
    )
    stats = stats_cls()
    first = stats.to_dict()
    if rust_frontend:
        assert first[METRICS_DESCRIPTOR_KEY] == build_descriptor()
        assert first[METRICS_DESCRIPTOR_KEY]["connector_id"] == connector_id
        second = stats.to_dict()
        assert METRICS_DESCRIPTOR_KEY not in second
    else:
        assert METRICS_DESCRIPTOR_KEY not in first


def test_hisparse_zero_snapshot_keeps_zero_counts_in_to_dict(monkeypatch):
    """Zero samples stay in the payload. Emission gating is parametrized above.

    Worker bootstrap of the zero snapshot lives in ``tests/v1/worker/test_utils.py``.
    """
    reset_metrics_descriptor_emission_for_tests()
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.metrics_descriptor.envs."
        "VLLM_USE_RUST_FRONTEND",
        True,
    )
    stats = HiSparseKVConnectorStats()
    stats.record_snapshot(0, 0, 0)
    payload = stats.to_dict()
    assert payload["cache_hits"] == [0]
    assert payload["cache_misses"] == [0]
    assert payload["host_to_device_bytes"] == [0]


def test_strip_metrics_descriptor_when_rebuilding_stats_from_wire_dict():
    """``__post_init__`` drops a wire ``_metrics_descriptor`` before accumulating."""
    assert strip_metrics_descriptor(None) is None
    assert strip_metrics_descriptor({"save_duration": []}) == {"save_duration": []}

    hf3fs_wire = {
        "save_duration": [0.01],
        "load_duration": [],
        "num_failed_save": 1,
        "num_failed_load": 0,
        "num_transfer_task": 1,
        METRICS_DESCRIPTOR_KEY: {"connector_id": "HF3FSKVConnector"},
    }
    hf3fs = HF3FSKVConnectorStats(data=hf3fs_wire)
    assert METRICS_DESCRIPTOR_KEY not in hf3fs.data
    assert hf3fs.data["save_duration"] == [0.01]
    assert hf3fs.data["num_failed_save"] == 1

    hisparse_wire = {
        "cache_hits": [1],
        "cache_misses": [2],
        "host_to_device_bytes": [3],
        METRICS_DESCRIPTOR_KEY: {"connector_id": "HiSparseConnector"},
    }
    hisparse = HiSparseKVConnectorStats(data=hisparse_wire)
    assert METRICS_DESCRIPTOR_KEY not in hisparse.data
    assert hisparse.data == {
        "cache_hits": [1],
        "cache_misses": [2],
        "host_to_device_bytes": [3],
    }


def test_offloading_metrics_descriptor_derived_from_metadata():
    """Descriptor names/kinds match OffloadingMetricMetadata."""
    descriptor = build_offloading_metrics_descriptor()
    assert descriptor["descriptor_version"] == 1
    assert descriptor["connector_id"] == "OffloadingConnector"

    by_name = {m["name"]: m for m in descriptor["metrics"]}
    expected = {
        **CPUOffloadingSpec.build_metric_definitions({"store_threshold": 2}),
        **get_connector_metric_definitions(),
    }
    assert set(by_name) == set(expected)

    # Float counters on the wire (seconds); all other counters are integer-valued.
    float_counter_names = frozenset(
        {
            _TransferMetricName.LOAD_TIME,
            _TransferMetricName.STORE_TIME,
        }
    )
    for name, metadata in expected.items():
        entry = by_name[name]
        assert entry["samples_path"] == f"data.{name}"
        if isinstance(metadata, OffloadingHistogramMetadata):
            assert entry["type"] == "histogram"
            assert entry["sample_kind"] == OBSERVE_EACH_F64
            assert entry["buckets"] == list(metadata.buckets or ())
        elif isinstance(metadata, OffloadingGaugeMetadata):
            assert entry["type"] == "gauge"
            assert entry["sample_kind"] == SET_F64
        elif isinstance(metadata, OffloadingCounterMetadata):
            assert entry["type"] == "counter"
            want = INC_BY_F64 if name in float_counter_names else INC_BY_U64
            assert entry["sample_kind"] == want
        else:
            raise AssertionError(f"unexpected metadata: {metadata}")


def _prom_fields(metric: _RecordingMetric) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "name": metric.kwargs["name"],
        "type": metric.kind,
        "documentation": metric.kwargs["documentation"],
    }
    if "buckets" in metric.kwargs:
        fields["buckets"] = list(metric.kwargs["buckets"])
    return fields


def _descriptor_fields(entry: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "name": entry["name"],
        "type": entry["type"],
        "documentation": entry["documentation"],
    }
    if "buckets" in entry:
        fields["buckets"] = list(entry["buckets"])
    return fields


def test_hf3fs_prom_metrics_match_descriptor():
    """Every HF3FS Prom metric matches the descriptor entry and vice versa."""
    _reset_recording_metrics()
    HF3FSPromMetrics(
        vllm_config=SimpleNamespace(kv_transfer_config=None),  # type: ignore[arg-type]
        metric_types={
            Gauge: _RecordingGauge,
            Counter: _RecordingCounter,
            Histogram: _RecordingHistogram,
        },
        labelnames=["engine"],
        per_engine_labelvalues={0: ["0"]},
    )
    # Parent constructions only (labels() returns self without re-appending).
    created = (
        _RecordingHistogram._created
        + _RecordingCounter._created
        + _RecordingGauge._created
    )
    prom_by_name = {m.kwargs["name"]: m for m in created}
    descriptor = HF3FSKVConnectorStats._metrics_descriptor()
    assert descriptor["connector_id"] == "HF3FSKVConnector"
    desc_by_name = {m["name"]: m for m in descriptor["metrics"]}

    assert set(prom_by_name) == set(desc_by_name)
    for name in desc_by_name:
        assert _prom_fields(prom_by_name[name]) == _descriptor_fields(
            desc_by_name[name]
        )


def test_offloading_prom_metrics_match_descriptor():
    """Every Offloading Prom metric matches the descriptor entry and vice versa."""
    _reset_recording_metrics()
    prom = OffloadPromMetrics(
        vllm_config=_FakeOffloadVllmConfig(store_threshold=2),  # type: ignore[arg-type]
        metric_types={
            Gauge: _RecordingGauge,
            Counter: _RecordingCounter,
            Histogram: _RecordingHistogram,
        },
        labelnames=["model_name", "engine"],
        per_engine_labelvalues={0: ["model", "0"]},
    )
    # Flat (non-deprecated) defs registered in _offloading_metric_defs.
    prom_by_name = {
        name: metric for name, metric in prom._offloading_metric_defs.items()
    }
    # Ensure they really came from the recording classes.
    for metric in prom_by_name.values():
        assert isinstance(metric, _RecordingMetric)

    descriptor = build_offloading_metrics_descriptor()
    assert descriptor["connector_id"] == "OffloadingConnector"
    desc_by_name = {m["name"]: m for m in descriptor["metrics"]}

    assert set(prom_by_name) == set(desc_by_name)
    for name in desc_by_name:
        assert _prom_fields(prom_by_name[name]) == _descriptor_fields(
            desc_by_name[name]
        )


def test_hisparse_metrics_descriptor_derived_from_counters():
    """Descriptor matches ``_HISPARSE_COUNTERS`` used by Prom."""
    descriptor = build_hisparse_metrics_descriptor()
    assert descriptor["connector_id"] == "HiSparseConnector"
    assert len(descriptor["metrics"]) == len(_HISPARSE_COUNTERS)
    for entry, (wire_key, documentation) in zip(
        descriptor["metrics"], _HISPARSE_COUNTERS, strict=True
    ):
        assert entry["name"] == f"vllm:hisparse_{wire_key}"
        assert entry["type"] == "counter"
        assert entry["samples_path"] == wire_key
        assert entry["sample_kind"] == INC_BY_SUM_U64
        assert entry["documentation"] == documentation
