# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SimpleCPUOffloadConnector metrics."""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
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


STARTUP_GAUGES = """
import os
import sys
from types import SimpleNamespace
from prometheus_client import Counter, Gauge, Histogram
from vllm.v1.metrics.prometheus import (
    get_prometheus_registry, set_gauge_initial_value, unregister_vllm_metrics,
)

if sys.argv[1] == "threaded":
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    args = ("vllm:simple_kv_offload_used_blocks", "Used blocks", ["engine"])
    writer = Gauge(*args, registry=None, multiprocess_mode=sys.argv[2]).labels("0")
    initializer = Gauge(*args, registry=None, multiprocess_mode=sys.argv[2]).labels("0")
    registry = get_prometheus_registry()
    if sys.argv[3] == "before":
        writer.set(7)
        set_gauge_initial_value(initializer, 0)
    else:
        initializing = Event()
        updated = Event()

        class InitialValue(float):
            def __float__(self):
                initializing.set()
                assert updated.wait(10), "Writer did not finish"
                return 0.0

        def update():
            assert initializing.wait(10), "Initializer did not start"
            writer.set(7)
            updated.set()

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(update)
            set_gauge_initial_value(initializer, InitialValue(0))
            future.result(timeout=10)

    samples = [sample for metric in registry.collect() for sample in metric.samples
               if sample.name == args[0]]
    assert [(sample.labels, sample.value) for sample in samples] == [
        ({"engine": "0"}, 7)
    ]
    writer.set(0)
    assert registry.get_sample_value(args[0], {"engine": "0"}) == 0
    sys.exit(0)

if sys.argv[1] == "cleanup":
    name = "vllm:simple_kv_offload_used_blocks"
    gauge = Gauge(name, "Used blocks", ["engine"], multiprocess_mode="mostrecent")
    plugin_name = "plugin_pending_stores"
    plugin = Gauge(plugin_name, "Plugin stores", multiprocess_mode="mostrecent")
    registry = get_prometheus_registry()
    set_gauge_initial_value(gauge.labels("0"), 0)
    set_gauge_initial_value(plugin, 0)
    assert registry.get_sample_value(name, {"engine": "0"}) == 0
    assert registry.get_sample_value(plugin_name) == 0
    unregister_vllm_metrics()
    assert registry.get_sample_value(name, {"engine": "0"}) is None
    assert registry.get_sample_value(plugin_name) == 0, "Plugin default was removed"
    sys.exit(0)

if sys.argv[1] == "native":
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
        OffloadPromMetrics as PromMetrics, OffloadingConnectorStats as Stats,
    )
    from vllm.v1.kv_offload.cpu.common import CPUOffloadingMetrics as Names
    names = (Names.CPU_CACHE_USAGE_PERC, Names.CPU_CACHE_WRITE_USAGE_PERC,
             Names.CPU_CACHE_READ_USAGE_PERC)
else:
    from vllm.v1.simple_kv_offload.metrics import (
        SimpleCPUOffloadPromMetrics as PromMetrics, SimpleCPUOffloadStats as Stats,
        MetricName as Names,
    )
    names = (Names.USED_BLOCKS, Names.PENDING_STORE_BLOCKS)

config = SimpleNamespace(
    kv_transfer_config=SimpleNamespace(kv_connector_extra_config={})
)
args = (config, {Gauge: Gauge, Counter: Counter, Histogram: Histogram},
        ["model_name", "engine"], {0: ["model", "0"], 3: ["model", "3"]})
prom = PromMetrics(*args)
initial = float(sys.argv[2]) if len(sys.argv) > 2 else 0
updated = float(sys.argv[3]) if len(sys.argv) > 3 else 1
registry = get_prometheus_registry()
for engine in (0, 3):
    labels = {"model_name": "model", "engine": str(engine)}
    if sys.argv[1] == "native" and os.getenv("PROMETHEUS_MULTIPROC_DIR"):
        labels["pid"] = str(os.getpid())
    for name in names:
        assert registry.get_sample_value(name, labels) == initial, (name, engine)
    stats = Stats()
    for name in names:
        stats.set_gauge(name, updated)
    prom.observe(stats.data, engine)
    for name in names:
        assert registry.get_sample_value(name, labels) == updated, (name, engine)

if len(sys.argv) > 4 and sys.argv[4] == "recreate":
    unregister_vllm_metrics()
    prom = PromMetrics(*args)
    registry = get_prometheus_registry()
    for engine in (0, 3):
        labels = {"model_name": "model", "engine": str(engine)}
        if sys.argv[1] == "native":
            labels["pid"] = str(os.getpid())
        for name in names:
            assert registry.get_sample_value(name, labels) == updated, (name, engine)
        stats = Stats()
        for name in names:
            stats.set_gauge(name, 0)
        prom.observe(stats.data, engine)
        for name in names:
            assert registry.get_sample_value(name, labels) == 0, (name, engine)
"""


def _run_usage_probe(tmp_path, multiprocess, *args):
    env = os.environ.copy()
    env.pop("PROMETHEUS_MULTIPROC_DIR", None)
    env.pop("prometheus_multiproc_dir", None)
    if multiprocess:
        env["PROMETHEUS_MULTIPROC_DIR"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, "-c", STARTUP_GAUGES, *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("multiprocess", [False, True])
@pytest.mark.parametrize("connector", ["native", "simple"])
def test_usage_gauges_exported_before_first_stats(tmp_path, multiprocess, connector):
    """Idle offload gauges exist before stats, which replace their startup zeros."""
    _run_usage_probe(tmp_path, multiprocess, connector)


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
def test_late_metrics_writer_preserves_usage_gauges(tmp_path):
    """A new metrics writer must not replace existing stats with startup zeros."""
    _run_usage_probe(tmp_path, True, "simple", "0", "7")
    _run_usage_probe(tmp_path, True, "simple", "7", "9")


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("connector,updated", [("native", "0.7"), ("simple", "7")])
def test_usage_gauges_survive_recorder_recreation(tmp_path, connector, updated):
    """Recreating a recorder preserves its multiprocess storage's current stats."""
    _run_usage_probe(tmp_path, True, connector, "0", updated, "recreate")


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("mode", ["mostrecent", "livemostrecent"])
@pytest.mark.parametrize("update", ["before", "during"])
def test_usage_gauge_initialization_preserves_same_process_updates(
    tmp_path, mode, update
):
    """Startup defaults cannot overwrite updates through another gauge object."""
    _run_usage_probe(tmp_path, True, "threaded", mode, update)


@pytest.mark.cpu_test
@pytest.mark.skip_global_cleanup
def test_usage_gauge_unregister_only_removes_vllm_defaults(tmp_path):
    """VLLM metric cleanup preserves other collectors' startup defaults."""
    _run_usage_probe(tmp_path, True, "cleanup")


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
    stats.increase_counter(LOAD_BLOCKS, 4)
    stats.set_gauge(USED_BLOCKS, 7)
    stats.set_gauge(INFO, 1, ("cpu", "false", "false", "64"))

    rebuilt = SimpleCPUOffloadConnector.build_kv_connector_stats(data=stats.to_dict())
    assert isinstance(rebuilt, SimpleCPUOffloadStats)
    assert _values(rebuilt, SAVE_OUTCOMES) == {("stored",): 3}
    assert _values(rebuilt, LOAD_BLOCKS) == {(): 4}
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
    stats2.increase_counter(LOAD_BLOCKS, 2)

    stats1.aggregate(stats2)
    assert stats1.data[_StatsKey.TYPES][LOAD_BLOCKS] == _MetricType.COUNTER
    assert _values(stats1, LOAD_BLOCKS) == {(): 2}


def test_stats_reduce_and_reset() -> None:
    stats = SimpleCPUOffloadStats()
    stats.increase_counter(SAVE_OUTCOMES, 3, ("stored",))
    stats.set_gauge(PENDING_STORE_BLOCKS, 4)
    stats.increase_counter(LOAD_BLOCKS, 9)
    stats.set_gauge(INFO, 1, ("cpu", "false", "false", "64"))

    reduced = stats.reduce()
    assert reduced[f"{SAVE_OUTCOMES}:{('stored',)}"] == 3
    assert reduced[PENDING_STORE_BLOCKS] == 4
    assert reduced[LOAD_BLOCKS] == 9
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
    assert defs[LOAD_BLOCKS].kwargs["labelnames"] == ["model_name", "engine"]
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
                LOAD_BLOCKS: {(): 4},
                USED_BLOCKS: {(): 5},
                PENDING_STORE_BLOCKS: {(): 1},
                INFO: {("cpu", "false", "false", "64"): 1},
            },
        )
    )

    assert prom._metrics[(0, SAVE_OUTCOMES, ("stored",))].increments == [3]
    assert prom._metrics[(0, LOAD_BLOCKS, ())].increments == [4]
    assert prom._metrics[(0, USED_BLOCKS, ())].set_values == [0, 5]
    assert prom._metrics[(0, PENDING_STORE_BLOCKS, ())].set_values == [0, 1]
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
    assert prom._metrics[(0, USED_BLOCKS, ())].set_values == [0]
    assert prom._metrics[(1, USED_BLOCKS, ())].set_values == [0, 3]


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


def test_manager_counts_load_blocks_completed() -> None:
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
    assert LOAD_BLOCKS not in stats.data[_StatsKey.DATA]

    sched.update_connector_output(
        KVConnectorOutput(finished_recving={request.request_id})
    )
    stats = sched.get_stats()
    assert _values(stats, LOAD_BLOCKS) == {(): 2}


def test_manager_abandoned_load_counts_as_completed_after_reset() -> None:
    """Loads abandoned by reset() that still finish on the worker are
    counted as loaded blocks.
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
    assert LOAD_BLOCKS not in stats.data[_StatsKey.DATA]

    assert sched.reset() is False
    assert set(sched._abandoned_reqs_to_load) == {request.request_id}

    sched.update_connector_output(
        KVConnectorOutput(finished_recving={request.request_id})
    )
    assert sched._abandoned_reqs_to_load == {}
    stats = sched.get_stats()
    assert _values(stats, LOAD_BLOCKS) == {(): 2}
