# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SimpleCPUOffloadConnector metrics."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

import pytest
from prometheus_client import Counter, Gauge, Histogram

from tests.v1.kv_connector.unit.test_simple_cpu_offload_connector import (
    _make_connector,
)
from tests.v1.simple_kv_offload.test_scheduler import (
    make_request,
    make_scheduler,
    make_scheduler_output,
)
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (
    SimpleCPUOffloadConnector,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.simple_kv_offload.manager import (
    BoundaryStoreStats,
    LoadRequestState,
    TransferMeta,
)
from vllm.v1.simple_kv_offload.metrics import (
    INFO_LABELS,
    OUTCOME_TO_FIELD,
    MetricName,
    SimpleCPUOffloadStats,
    _MetricType,
    _StatsKey,
)

SAVE_OUTCOMES = MetricName.SAVE_OUTCOMES
LOAD_BLOCKS = MetricName.LOAD_BLOCKS
USED_BLOCKS = MetricName.USED_BLOCKS
PENDING_STORE_BLOCKS = MetricName.PENDING_STORE_BLOCKS
INFO = MetricName.INFO


class _FakeMetric:
    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        self.children: list[_FakeMetric] = []
        self.observed: list[int | float] = []
        self.increments: list[int | float] = []
        self.set_values: list[int | float] = []
        self.labelvalues: tuple[object, ...] = ()

    def labels(self, *labelvalues: object) -> _FakeMetric:
        child = _FakeMetric(**self.kwargs)
        child.labelvalues = labelvalues
        self.children.append(child)
        return child

    def observe(self, value: int | float) -> None:
        self.observed.append(value)

    def inc(self, value: int | float) -> None:
        self.increments.append(value)

    def set(self, value: int | float) -> None:
        self.set_values.append(value)


def _fake_metric_types() -> dict[type, type]:
    return {Gauge: _FakeMetric, Counter: _FakeMetric, Histogram: _FakeMetric}


def _make_prom_metrics(per_engine: dict[int, list[object]] | None = None) -> Any:
    from vllm.v1.simple_kv_offload.metrics import SimpleCPUOffloadPromMetrics

    return SimpleCPUOffloadPromMetrics(
        vllm_config=SimpleNamespace(kv_transfer_config=None),
        metric_types=_fake_metric_types(),
        labelnames=["model_name", "engine"],
        per_engine_labelvalues=per_engine or {0: ["model", "0"]},
    )


def _stats_payload(types: dict, data: dict) -> dict[str, Any]:
    return {_StatsKey.TYPES: types, _StatsKey.DATA: data}


def _values(stats: SimpleCPUOffloadStats, name: str) -> dict:
    return stats.data[_StatsKey.DATA][name]


# ---------------------------------------------------------------------------
# Stats class
# ---------------------------------------------------------------------------


def test_build_kv_connector_stats_none_and_empty() -> None:
    stats = SimpleCPUOffloadConnector.build_kv_connector_stats(data=None)
    assert isinstance(stats, SimpleCPUOffloadStats)
    assert stats.is_empty()

    stats = SimpleCPUOffloadConnector.build_kv_connector_stats(data={})
    assert isinstance(stats, SimpleCPUOffloadStats)
    assert stats.is_empty()


def test_stats_round_trip_through_serialized_dict() -> None:
    stats = SimpleCPUOffloadStats()
    stats.increase_counter(SAVE_OUTCOMES, 3, ("stored",))
    stats.increase_counter(LOAD_BLOCKS, 4, ("issued",))
    stats.set_gauge(USED_BLOCKS, 7)
    stats.set_gauge(INFO, 1, ("cpu", "false", "false", "64"))

    rebuilt = SimpleCPUOffloadConnector.build_kv_connector_stats(data=stats.to_dict())
    assert isinstance(rebuilt, SimpleCPUOffloadStats)
    assert _values(rebuilt, SAVE_OUTCOMES) == {("stored",): 3}
    assert _values(rebuilt, LOAD_BLOCKS) == {("issued",): 4}
    assert _values(rebuilt, USED_BLOCKS) == {(): 7}
    assert _values(rebuilt, INFO) == {("cpu", "false", "false", "64"): 1}


def test_stats_aggregate_sums_counters_and_keeps_latest_gauge() -> None:
    stats1 = SimpleCPUOffloadStats()
    stats1.increase_counter(SAVE_OUTCOMES, 3, ("stored",))
    stats1.increase_counter(SAVE_OUTCOMES, 1, ("dropped_cpu_full",))
    stats1.set_gauge(USED_BLOCKS, 7)

    stats2 = SimpleCPUOffloadStats()
    stats2.increase_counter(SAVE_OUTCOMES, 4, ("stored",))
    stats2.set_gauge(USED_BLOCKS, 2)

    result = stats1.aggregate(stats2)
    assert result is stats1
    outcomes = _values(stats1, SAVE_OUTCOMES)
    assert outcomes == {("stored",): 7, ("dropped_cpu_full",): 1}
    assert _values(stats1, USED_BLOCKS) == {(): 2}


def test_stats_aggregate_merges_types_from_other() -> None:
    stats1 = SimpleCPUOffloadStats()
    stats1.set_gauge(USED_BLOCKS, 1)

    stats2 = SimpleCPUOffloadStats()
    stats2.increase_counter(LOAD_BLOCKS, 2, ("issued",))

    stats1.aggregate(stats2)
    assert stats1.data[_StatsKey.TYPES][LOAD_BLOCKS] == _MetricType.COUNTER
    assert _values(stats1, LOAD_BLOCKS) == {("issued",): 2}


def test_stats_reduce_and_reset() -> None:
    stats = SimpleCPUOffloadStats()
    stats.increase_counter(SAVE_OUTCOMES, 3, ("stored",))
    stats.set_gauge(PENDING_STORE_BLOCKS, 4)
    stats.increase_counter(LOAD_BLOCKS, 9, ("issued",))
    stats.set_gauge(INFO, 1, ("cpu", "false", "false", "64"))

    reduced = stats.reduce()
    assert reduced[f"{SAVE_OUTCOMES}:{('stored',)}"] == 3
    assert reduced[PENDING_STORE_BLOCKS] == 4
    assert reduced[f"{LOAD_BLOCKS}:{('issued',)}"] == 9
    assert not any(key.startswith(INFO) for key in reduced)

    stats.reset()
    assert stats.is_empty()


def test_outcome_labels_match_boundary_store_stats_fields() -> None:
    """Every outcome label maps 1:1 to a BoundaryStoreStats counter field."""
    boundary_fields = {
        f.name for f in dataclasses.fields(BoundaryStoreStats) if f.name != "published"
    }
    assert set(OUTCOME_TO_FIELD.values()) == boundary_fields
    assert len(set(OUTCOME_TO_FIELD.values())) == len(OUTCOME_TO_FIELD)


# ---------------------------------------------------------------------------
# Prom metrics class
# ---------------------------------------------------------------------------


def test_prom_metrics_registers_tier1_metrics() -> None:
    prom = _make_prom_metrics()
    defs = prom._defs
    assert set(defs) == {
        SAVE_OUTCOMES,
        LOAD_BLOCKS,
        USED_BLOCKS,
        PENDING_STORE_BLOCKS,
        INFO,
    }
    # The prom class registers literal names (required by the docs
    # generator); they must stay in sync with the emission-side constants.
    assert {m.kwargs["name"] for m in defs.values()} == {
        SAVE_OUTCOMES,
        LOAD_BLOCKS,
        USED_BLOCKS,
        PENDING_STORE_BLOCKS,
        INFO,
    }
    for metric in defs.values():
        assert metric.kwargs["documentation"]
        assert metric.kwargs["labelnames"][:2] == ["model_name", "engine"]

    assert defs[SAVE_OUTCOMES].kwargs["labelnames"] == [
        "model_name",
        "engine",
        "outcome",
    ]
    assert defs[LOAD_BLOCKS].kwargs["labelnames"] == [
        "model_name",
        "engine",
        "phase",
    ]
    assert defs[INFO].kwargs["labelnames"] == [
        "model_name",
        "engine",
        *INFO_LABELS,
    ]
    assert defs[USED_BLOCKS].kwargs["labelnames"] == ["model_name", "engine"]

    # mostrecent: multiprocess scrapes expose one series per label set, not
    # one pid-tagged series per writer process.
    for gauge in (USED_BLOCKS, PENDING_STORE_BLOCKS, INFO):
        assert defs[gauge].kwargs["multiprocess_mode"] == "mostrecent"


def test_prom_metrics_observe_routes_each_series() -> None:
    prom = _make_prom_metrics()
    prom.observe(
        _stats_payload(
            types={
                SAVE_OUTCOMES: _MetricType.COUNTER,
                LOAD_BLOCKS: _MetricType.COUNTER,
                USED_BLOCKS: _MetricType.GAUGE,
                PENDING_STORE_BLOCKS: _MetricType.GAUGE,
                INFO: _MetricType.GAUGE,
            },
            data={
                SAVE_OUTCOMES: {("stored",): 3},
                LOAD_BLOCKS: {("issued",): 4, ("completed",): 2},
                USED_BLOCKS: {(): 5},
                PENDING_STORE_BLOCKS: {(): 1},
                INFO: {("cpu", "false", "false", "64"): 1},
            },
        )
    )

    assert prom._metrics[(0, SAVE_OUTCOMES, ("stored",))].increments == [3]
    assert prom._metrics[(0, LOAD_BLOCKS, ("issued",))].increments == [4]
    assert prom._metrics[(0, LOAD_BLOCKS, ("completed",))].increments == [2]
    assert prom._metrics[(0, USED_BLOCKS, ())].set_values == [5]
    assert prom._metrics[(0, PENDING_STORE_BLOCKS, ())].set_values == [1]
    assert prom._metrics[(0, INFO, ("cpu", "false", "false", "64"))].set_values == [1]
    assert prom._metrics[(0, SAVE_OUTCOMES, ("stored",))].labelvalues == (
        "model",
        "0",
        "stored",
    )
    assert prom._metrics[(0, INFO, ("cpu", "false", "false", "64"))].labelvalues == (
        "model",
        "0",
        "cpu",
        "false",
        "false",
        "64",
    )


def test_prom_metrics_reuses_bound_children() -> None:
    prom = _make_prom_metrics()
    payload = _stats_payload(
        types={SAVE_OUTCOMES: _MetricType.COUNTER},
        data={SAVE_OUTCOMES: {("stored",): 1}},
    )
    prom.observe(payload)
    prom.observe(payload)

    child = prom._metrics[(0, SAVE_OUTCOMES, ("stored",))]
    assert child.increments == [1, 1]
    assert len(prom._defs[SAVE_OUTCOMES].children) == 1


def test_prom_metrics_rejects_unknown_metric() -> None:
    prom = _make_prom_metrics()
    with pytest.raises(AssertionError, match="Unknown"):
        prom.observe(
            _stats_payload(
                types={"vllm:not_a_metric": _MetricType.COUNTER},
                data={"vllm:not_a_metric": {(): 1}},
            )
        )


def test_prom_metrics_rejects_wrong_label_count() -> None:
    prom = _make_prom_metrics()
    with pytest.raises(AssertionError, match="labels"):
        prom.observe(
            _stats_payload(
                types={SAVE_OUTCOMES: _MetricType.COUNTER},
                data={SAVE_OUTCOMES: {("stored", "extra"): 1}},
            )
        )


def test_prom_metrics_routes_per_engine() -> None:
    prom = _make_prom_metrics(per_engine={0: ["model", "0"], 1: ["model", "1"]})
    prom.observe(
        _stats_payload(
            types={USED_BLOCKS: _MetricType.GAUGE}, data={USED_BLOCKS: {(): 3}}
        ),
        engine_idx=1,
    )
    assert (0, USED_BLOCKS, ()) not in prom._metrics
    assert prom._metrics[(1, USED_BLOCKS, ())].set_values == [3]


def test_connector_build_prom_metrics() -> None:
    from vllm.v1.simple_kv_offload.metrics import SimpleCPUOffloadPromMetrics

    prom = SimpleCPUOffloadConnector.build_prom_metrics(
        vllm_config=SimpleNamespace(kv_transfer_config=None),
        metric_types=_fake_metric_types(),
        labelnames=["model_name", "engine"],
        per_engine_labelvalues={0: ["model", "0"]},
    )
    assert isinstance(prom, SimpleCPUOffloadPromMetrics)


# ---------------------------------------------------------------------------
# Manager: interval stats
# ---------------------------------------------------------------------------


def test_manager_get_stats_reports_outcome_deltas_per_interval() -> None:
    fixture = make_scheduler()
    sched = fixture.scheduler

    sched.boundary_store_stats.stored += 3
    sched.boundary_store_stats.dropped_cpu_full += 2
    sched.boundary_store_stats.skipped_in_flight += 1

    stats = sched.get_stats()
    assert _values(stats, SAVE_OUTCOMES) == {
        ("stored",): 3,
        ("dropped_cpu_full",): 2,
        ("skipped_in_flight",): 1,
    }

    # Deltas are per-interval: a second drain reports no new outcomes.
    stats2 = sched.get_stats()
    assert SAVE_OUTCOMES not in stats2.data[_StatsKey.DATA]


def test_manager_reset_preserves_undrained_outcome_deltas() -> None:
    fixture = make_scheduler()
    sched = fixture.scheduler

    sched.boundary_store_stats.stored += 3
    sched.reset()

    stats = sched.get_stats()
    assert _values(stats, SAVE_OUTCOMES) == {("stored",): 3}


def test_manager_get_stats_gauges_and_info_labels() -> None:
    fixture = make_scheduler(num_cpu_blocks=8)
    sched = fixture.scheduler

    sched._store_event_to_blocks[0] = TransferMeta([10], [11])
    sched._pending_finished_stores.append(TransferMeta([12], [13]))
    sched._abandoned_store_event_to_blocks[1] = TransferMeta([14], [15])

    stats = sched.get_stats()
    assert _values(stats, PENDING_STORE_BLOCKS) == {(): 3}
    used = _values(stats, USED_BLOCKS)[()]
    assert used == sched.num_cpu_blocks - sched.cpu_block_pool.get_num_free_blocks()

    info = _values(stats, INFO)
    assert info == {("cpu", "false", "false", str(sched.num_cpu_blocks)): 1}

    # Gauges are re-reported every drain.
    stats2 = sched.get_stats()
    assert _values(stats2, PENDING_STORE_BLOCKS) == {(): 3}


def test_manager_info_labels_reflect_disk_mode() -> None:
    connector = _make_connector(
        extra_config={
            "kv_offload_backend": "disk",
            "disk_path": "/tmp/fake-disk-metrics",
            "disk_capacity_bytes": 1024**3,
            "use_page_cache": True,
        }
    )
    sched = connector.scheduler_manager
    assert sched is not None
    assert sched._info_labelvalues[0] == "disk"
    assert sched._info_labelvalues[1] == "true"
    stats = sched.get_stats()
    assert set(_values(stats, INFO)) == {sched._info_labelvalues}


def test_manager_info_labels_page_cache_false_for_cpu_backend() -> None:
    """CPU backend ignores use_page_cache; the info gauge must not claim it."""
    connector = _make_connector(extra_config={"use_page_cache": True})
    sched = connector.scheduler_manager
    assert sched is not None
    assert sched._info_labelvalues[:2] == ("cpu", "false")
    stats = sched.get_stats()
    assert set(_values(stats, INFO)) == {sched._info_labelvalues}


def test_manager_counts_load_blocks_issued_and_completed() -> None:
    fixture = make_scheduler(num_cpu_blocks=8)
    sched = fixture.scheduler

    request = make_request(num_blocks=2)
    gpu_blocks = fixture.gpu_block_pool.get_new_blocks(2)
    cpu_blocks = sched.cpu_block_pool.get_new_blocks(2)
    sched._reqs_to_load[request.request_id] = LoadRequestState(
        request=request,
        transfer_meta=TransferMeta(
            [b.block_id for b in gpu_blocks], [b.block_id for b in cpu_blocks]
        ),
    )

    sched.build_connector_meta(make_scheduler_output({}))
    stats = sched.get_stats()
    assert _values(stats, LOAD_BLOCKS) == {("issued",): 2}

    sched.update_connector_output(
        KVConnectorOutput(finished_recving={request.request_id})
    )
    stats = sched.get_stats()
    assert _values(stats, LOAD_BLOCKS) == {("completed",): 2}


def test_manager_abandoned_load_counts_as_completed_after_reset() -> None:
    """Pins issued/completed balance: loads abandoned by reset() that still
    finish on the worker are credited to the completed counter.
    """
    fixture = make_scheduler(num_cpu_blocks=8)
    sched = fixture.scheduler

    request = make_request(num_blocks=2)
    gpu_blocks = fixture.gpu_block_pool.get_new_blocks(2)
    cpu_blocks = sched.cpu_block_pool.get_new_blocks(2)
    sched._reqs_to_load[request.request_id] = LoadRequestState(
        request=request,
        transfer_meta=TransferMeta(
            [b.block_id for b in gpu_blocks], [b.block_id for b in cpu_blocks]
        ),
    )

    sched.build_connector_meta(make_scheduler_output({}))
    stats = sched.get_stats()
    assert _values(stats, LOAD_BLOCKS) == {("issued",): 2}

    assert sched.reset() is False
    assert set(sched._abandoned_reqs_to_load) == {request.request_id}

    sched.update_connector_output(
        KVConnectorOutput(finished_recving={request.request_id})
    )
    assert sched._abandoned_reqs_to_load == {}
    stats = sched.get_stats()
    assert _values(stats, LOAD_BLOCKS) == {("completed",): 2}
