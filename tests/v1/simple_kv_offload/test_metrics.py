# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SimpleCPUOffloadConnector metrics: stats payload
construction/aggregation, worker poll-and-reset, scheduler gauges,
worker+scheduler aggregation, IPC (msgpack) round trip, and Prometheus
observation. All CPU-only.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from prometheus_client import Counter, Gauge, Histogram

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    TransferStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
    _MetricType,
    _StatsKey,
    _TransferMetricName,
)
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (
    SimpleCPUOffloadConnector,
    SimpleCPUOffloadPromMetrics,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.kv_offload.factory import OffloadingSpecFactory
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder
from vllm.v1.simple_kv_offload.copy_backend import DmaCopyEvent
from vllm.v1.simple_kv_offload.metadata import SimpleCPUOffloadWorkerMetadata
from vllm.v1.simple_kv_offload.metrics import SimpleCPUMetricName
from vllm.v1.simple_kv_offload.worker import SimpleCPUOffloadWorker

from .test_scheduler import (
    BLOCK_SIZE,
    _alloc_and_register,
    make_request,
    make_scheduler,
    make_scheduler_output,
    simulate_store_completion,
)

# ---------------------------------------------------------------------------
# 6. build_kv_connector_stats
# ---------------------------------------------------------------------------


def test_build_kv_connector_stats_none_returns_empty():
    stats = SimpleCPUOffloadConnector.build_kv_connector_stats(data=None)

    assert isinstance(stats, OffloadingConnectorStats)
    assert stats.is_empty()


def test_build_kv_connector_stats_rehydrates_from_payload():
    payload = {
        _StatsKey.TYPES: {
            _TransferMetricName.LOAD_BYTES: _MetricType.COUNTER,
            SimpleCPUMetricName.TOTAL_BLOCKS: _MetricType.GAUGE,
        },
        _StatsKey.DATA: {
            _TransferMetricName.LOAD_BYTES: {(): 128},
            SimpleCPUMetricName.TOTAL_BLOCKS: {(): 64},
        },
    }

    stats = SimpleCPUOffloadConnector.build_kv_connector_stats(data=payload)

    assert isinstance(stats, OffloadingConnectorStats)
    values = stats.data[_StatsKey.DATA]
    assert values[_TransferMetricName.LOAD_BYTES][()] == 128
    assert values[SimpleCPUMetricName.TOTAL_BLOCKS][()] == 64


# ---------------------------------------------------------------------------
# 7. Aggregation semantics via the SimpleCPU payload: counters sum, gauges
#    last-write-wins, histograms extend, labelvalues survive.
# ---------------------------------------------------------------------------


def test_aggregation_counters_sum_gauges_latest_histograms_extend():
    stats1 = OffloadingConnectorStats()
    stats1.increase_counter(_TransferMetricName.LOAD_BYTES, 100)
    stats1.observe_histogram(_TransferMetricName.LOAD_SIZE, 100)
    stats1.set_gauge(SimpleCPUMetricName.PENDING_LOADS, 3)
    stats1.increase_counter("labeled_counter", 5, ("gpu0",))

    stats2 = OffloadingConnectorStats()
    stats2.increase_counter(_TransferMetricName.LOAD_BYTES, 50)
    stats2.observe_histogram(_TransferMetricName.LOAD_SIZE, 50)
    stats2.set_gauge(SimpleCPUMetricName.PENDING_LOADS, 7)
    stats2.increase_counter("labeled_counter", 2, ("gpu0",))
    stats2.increase_counter("labeled_counter", 9, ("gpu1",))

    stats1.aggregate(stats2)

    values = stats1.data[_StatsKey.DATA]
    assert values[_TransferMetricName.LOAD_BYTES][()] == 150  # counter: sum
    assert values[_TransferMetricName.LOAD_SIZE][()] == [100, 50]  # histogram: extend
    assert values[SimpleCPUMetricName.PENDING_LOADS][()] == 7  # gauge: last write wins
    assert values["labeled_counter"][("gpu0",)] == 7  # labelvalues preserved
    assert values["labeled_counter"][("gpu1",)] == 9


# ---------------------------------------------------------------------------
# 8. Serialized (IPC) stats path: round-trip through the real MsgpackEncoder/
#    Decoder used to send SchedulerStats.kv_connector_stats to the frontend.
# ---------------------------------------------------------------------------


def _sample_stats() -> OffloadingConnectorStats:
    stats = OffloadingConnectorStats()
    stats.increase_counter(_TransferMetricName.LOAD_BYTES, 4096)
    stats.increase_counter(_TransferMetricName.LOAD_TIME, 0.02)
    stats.observe_histogram(_TransferMetricName.LOAD_SIZE, 4096)
    stats.increase_counter(_TransferMetricName.STORE_BYTES, 8192)
    stats.increase_counter(_TransferMetricName.STORE_TIME, 0.04)
    stats.observe_histogram(_TransferMetricName.STORE_SIZE, 8192)
    stats.set_gauge(SimpleCPUMetricName.TOTAL_BLOCKS, 100)
    stats.set_gauge(SimpleCPUMetricName.FREE_BLOCKS, 40)
    stats.set_gauge(SimpleCPUMetricName.USED_BLOCKS, 60)
    stats.set_gauge(SimpleCPUMetricName.USAGE_PERC, 0.6)
    stats.set_gauge(SimpleCPUMetricName.PENDING_LOADS, 2)
    stats.set_gauge(SimpleCPUMetricName.PENDING_STORES, 1)
    # A labeled key exercises the tuple-labelvalues-as-dict-key shape that
    # the self-describing {types, data} payload relies on.
    stats.increase_counter("some_labeled_metric", 3, ("engine0", "gpu"))
    return stats


def test_stats_data_survives_msgpack_round_trip():
    """Round-trip the raw ``.data`` dict exactly as it travels: scheduler.py
    ``make_stats()`` sets ``SchedulerStats.kv_connector_stats = stats.data``
    (see vllm/v1/core/sched/scheduler.py), and that dict is what gets
    msgpack-encoded for IPC to the frontend process."""
    stats = _sample_stats()

    encoder = MsgpackEncoder()
    decoder = MsgpackDecoder()
    encoded = encoder.encode(stats.data)
    decoded = decoder.decode(encoded)

    assert decoded == stats.data

    # Tuple labelvalue keys must survive as tuples (used as dict keys
    # throughout OffloadingConnectorStats), not silently become lists.
    for metric_name, label_map in decoded[_StatsKey.DATA].items():
        for labelvalues in label_map:
            assert isinstance(labelvalues, tuple), (
                f"{metric_name} labelvalues {labelvalues!r} did not survive "
                "msgpack round trip as a tuple"
            )

    rehydrated = SimpleCPUOffloadConnector.build_kv_connector_stats(data=decoded)
    assert isinstance(rehydrated, OffloadingConnectorStats)
    assert rehydrated.data == stats.data


def test_stats_survive_msgpack_round_trip_via_scheduler_stats():
    """End-to-end variant: wrap in the real SchedulerStats dataclass (as
    produced by Scheduler.make_stats()) to prove the field-level path, not
    just the bare dict, round-trips."""
    stats = _sample_stats()
    sched_stats = SchedulerStats(kv_connector_stats=stats.data)

    encoder = MsgpackEncoder()
    decoder = MsgpackDecoder()
    encoded = encoder.encode(sched_stats)
    decoded = decoder.decode(encoded)

    assert decoded["kv_connector_stats"] == stats.data
    rehydrated = SimpleCPUOffloadConnector.build_kv_connector_stats(
        data=decoded["kv_connector_stats"]
    )
    assert rehydrated.data == stats.data


# ---------------------------------------------------------------------------
# 5 (worker half). Empty-batch events (num_bytes=0) must not affect stats.
# ---------------------------------------------------------------------------


def test_record_copy_event_skips_stats_for_zero_bytes():
    worker = SimpleCPUOffloadWorker(
        vllm_config=None, kv_cache_config=None, cpu_capacity_bytes=0
    )
    ev = DmaCopyEvent(
        event_idx=3,
        start_event=Mock(elapsed_time=Mock(return_value=999.0)),
        end_event=Mock(),
        num_bytes=0,
        is_store=False,
        release=lambda: None,
    )

    worker._record_copy_event(ev)

    assert worker._transfer_stats.is_empty()
    assert worker._transfer_stats.load.bytes == 0
    assert worker._transfer_stats.load.time == 0.0
    # An empty batch has nothing to time; elapsed_time must not even be
    # consulted (the method returns before reaching it).
    ev.start_event.elapsed_time.assert_not_called()


# ---------------------------------------------------------------------------
# 9. Worker poll-and-reset via build_connector_worker_meta().
# ---------------------------------------------------------------------------


def _make_dma_event(
    event_idx: int, num_bytes: int, is_store: bool, elapsed_ms: float
) -> DmaCopyEvent:
    return DmaCopyEvent(
        event_idx=event_idx,
        start_event=Mock(elapsed_time=Mock(return_value=elapsed_ms)),
        end_event=Mock(),
        num_bytes=num_bytes,
        is_store=is_store,
        release=lambda: None,
    )


def test_worker_meta_poll_and_reset():
    worker = SimpleCPUOffloadWorker(
        vllm_config=None, kv_cache_config=None, cpu_capacity_bytes=0
    )

    load_ev = _make_dma_event(0, num_bytes=1024, is_store=False, elapsed_ms=10.0)
    worker._record_copy_event(load_ev)

    meta = worker.build_connector_worker_meta()

    assert meta is not None
    assert not meta.transfer_stats.is_empty()
    assert meta.transfer_stats.load.bytes == 1024
    assert meta.transfer_stats.load.time == pytest.approx(0.01)

    # poll-and-reset: nothing new since the last call -> None.
    assert worker.build_connector_worker_meta() is None


def test_worker_meta_built_with_only_transfer_stats_no_store_events():
    """meta must be returned even when completed_store_events is empty but
    transfer_stats has data (build_connector_worker_meta's early-return must
    check both, not just completed_store_events)."""
    worker = SimpleCPUOffloadWorker(
        vllm_config=None, kv_cache_config=None, cpu_capacity_bytes=0
    )
    assert worker._completed_store_events == {}

    store_ev = _make_dma_event(1, num_bytes=2048, is_store=True, elapsed_ms=5.0)
    worker._record_copy_event(store_ev)
    assert worker._completed_store_events == {}

    meta = worker.build_connector_worker_meta()

    assert meta is not None
    assert meta.completed_store_events == {}
    assert meta.transfer_stats.store.bytes == 2048
    assert meta.transfer_stats.store.time == pytest.approx(0.005)


# ---------------------------------------------------------------------------
# 10. Scheduler-only gauges (no worker input).
# ---------------------------------------------------------------------------


def test_scheduler_only_gauges_no_worker_activity():
    fix = make_scheduler(num_cpu_blocks=8, num_gpu_blocks=16, lazy=False)
    sched = fix.scheduler

    stats = sched.get_stats()

    assert stats is not None
    values = stats.data[_StatsKey.DATA]
    total_blocks = sched.cpu_block_pool.num_gpu_blocks - 1
    assert values[SimpleCPUMetricName.TOTAL_BLOCKS][()] == total_blocks
    assert values[SimpleCPUMetricName.FREE_BLOCKS][()] == total_blocks
    assert values[SimpleCPUMetricName.USED_BLOCKS][()] == 0
    usage = values[SimpleCPUMetricName.USAGE_PERC][()]
    assert 0.0 <= usage <= 1.0
    assert usage == 0.0
    assert values[SimpleCPUMetricName.PENDING_LOADS][()] == 0
    assert values[SimpleCPUMetricName.PENDING_STORES][()] == 0
    # No transfer counters absent any worker-reported activity.
    assert _TransferMetricName.LOAD_BYTES not in values


def test_scheduler_gauges_reflect_pending_store_and_load():
    fix = make_scheduler(num_cpu_blocks=8, num_gpu_blocks=16, lazy=False)
    sched = fix.scheduler

    num_blocks = 2
    req = make_request(num_blocks=num_blocks)
    kv_blocks = _alloc_and_register(fix, req, num_blocks)
    sched.update_state_after_alloc(req, kv_blocks, num_external_tokens=0)
    sched_out = make_scheduler_output(
        {req.request_id: num_blocks * BLOCK_SIZE},
        new_reqs={req.request_id: kv_blocks.get_block_ids()},
    )
    meta = sched.build_connector_meta(sched_out)
    assert meta.store_event >= 0

    values = sched.get_stats().data[_StatsKey.DATA]
    assert values[SimpleCPUMetricName.PENDING_STORES][()] == 1

    simulate_store_completion(sched, meta.store_event)

    values_after = sched.get_stats().data[_StatsKey.DATA]
    assert values_after[SimpleCPUMetricName.PENDING_STORES][()] == 0

    # --- Pending load: new request hits the now-cached CPU blocks. ---
    req2 = Request(
        request_id="req-metrics-load",
        prompt_token_ids=req.prompt_token_ids,
        sampling_params=req.sampling_params,
        pooling_params=None,
        mm_features=None,
        block_hasher=req._block_hasher,
    )
    hit_tokens, is_async = sched.get_num_new_matched_tokens(req2, num_computed_tokens=0)
    assert hit_tokens > 0
    assert is_async is True

    gpu_blocks2 = fix.gpu_block_pool.get_new_blocks(num_blocks)
    kv_blocks2 = KVCacheBlocks(blocks=(gpu_blocks2,))
    sched.update_state_after_alloc(req2, kv_blocks2, num_external_tokens=hit_tokens)

    values_load = sched.get_stats().data[_StatsKey.DATA]
    assert values_load[SimpleCPUMetricName.PENDING_LOADS][()] == 1


# ---------------------------------------------------------------------------
# 11. Worker+scheduler aggregation: counters popped after get_stats(),
#     gauges always re-emitted; TP/PP metadata.aggregate() sums stats.
# ---------------------------------------------------------------------------


def test_worker_scheduler_aggregation_counters_popped_gauges_persist():
    fix = make_scheduler(num_cpu_blocks=8, num_gpu_blocks=16, lazy=False)
    sched = fix.scheduler

    transfer_stats = TransferStats()
    transfer_stats.load.record(1024, 0.01)
    transfer_stats.store.record(2048, 0.02)

    output = KVConnectorOutput(
        finished_recving=set(),
        kv_connector_worker_meta=SimpleCPUOffloadWorkerMetadata(
            completed_store_events={},
            transfer_stats=transfer_stats,
        ),
    )
    sched.update_connector_output(output)

    values = sched.get_stats().data[_StatsKey.DATA]
    assert values[_TransferMetricName.LOAD_BYTES][()] == 1024
    assert values[_TransferMetricName.LOAD_TIME][()] == pytest.approx(0.01)
    assert values[_TransferMetricName.LOAD_SIZE][()] == [1024]
    assert values[_TransferMetricName.STORE_BYTES][()] == 2048
    assert values[_TransferMetricName.STORE_TIME][()] == pytest.approx(0.02)
    assert values[_TransferMetricName.STORE_SIZE][()] == [2048]
    assert SimpleCPUMetricName.TOTAL_BLOCKS in values  # gauges present too

    values_second = sched.get_stats().data[_StatsKey.DATA]
    assert _TransferMetricName.LOAD_BYTES not in values_second  # counters popped
    assert SimpleCPUMetricName.TOTAL_BLOCKS in values_second  # gauges persist


def test_worker_metadata_aggregate_sums_across_workers():
    stats_a = TransferStats()
    stats_a.load.record(100, 0.001)
    stats_b = TransferStats()
    stats_b.load.record(200, 0.002)

    meta_a = SimpleCPUOffloadWorkerMetadata(
        completed_store_events={1: 1}, transfer_stats=stats_a
    )
    meta_b = SimpleCPUOffloadWorkerMetadata(
        completed_store_events={1: 1, 2: 1}, transfer_stats=stats_b
    )

    merged = meta_a.aggregate(meta_b)

    assert merged.completed_store_events == {1: 2, 2: 1}
    assert merged.transfer_stats.load.bytes == 300
    assert merged.transfer_stats.load.time == pytest.approx(0.003)
    assert merged.transfer_stats.load.sizes == [100, 200]


# ---------------------------------------------------------------------------
# 12. Prometheus observe(): SimpleCPUOffloadPromMetrics wiring.
# ---------------------------------------------------------------------------


class _FakeMetric:
    """Stand-in for a prometheus_client Counter/Gauge/Histogram, mirroring
    tests/v1/kv_connector/unit/offloading_connector/test_metrics.py."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.children: list[_FakeMetric] = []
        self.observed: list[float] = []
        self.increments: list[float] = []
        self.set_values: list[float] = []
        self.labelvalues: tuple[object, ...] = ()

    def labels(self, *labelvalues):
        child = _FakeMetric(**self.kwargs)
        child.labelvalues = labelvalues
        self.children.append(child)
        return child

    def observe(self, value):
        self.observed.append(value)

    def inc(self, value):
        self.increments.append(value)

    def set(self, value):
        self.set_values.append(value)


class _FakeVllmConfig:
    def __init__(self):
        self.kv_transfer_config = SimpleNamespace(kv_connector_extra_config={})


def _make_prom_metrics(per_engine_labelvalues=None) -> SimpleCPUOffloadPromMetrics:
    return SimpleCPUOffloadPromMetrics(
        vllm_config=_FakeVllmConfig(),  # type: ignore[arg-type]
        metric_types={
            Gauge: _FakeMetric,
            Counter: _FakeMetric,
            Histogram: _FakeMetric,
        },
        labelnames=["model_name", "engine"],
        per_engine_labelvalues=per_engine_labelvalues or {0: ["model", "0"]},
    )


_ALL_SIMPLE_CPU_GAUGES = (
    (SimpleCPUMetricName.TOTAL_BLOCKS, 100),
    (SimpleCPUMetricName.FREE_BLOCKS, 40),
    (SimpleCPUMetricName.USED_BLOCKS, 60),
    (SimpleCPUMetricName.USAGE_PERC, 0.6),
    (SimpleCPUMetricName.PENDING_LOADS, 2),
    (SimpleCPUMetricName.PENDING_STORES, 1),
)


def test_simple_cpu_prom_metrics_observes_transfer_and_gauges():
    prom = _make_prom_metrics()

    types = {
        _TransferMetricName.LOAD_BYTES: _MetricType.COUNTER,
        _TransferMetricName.LOAD_TIME: _MetricType.COUNTER,
        _TransferMetricName.LOAD_SIZE: _MetricType.HISTOGRAM,
        _TransferMetricName.STORE_BYTES: _MetricType.COUNTER,
        _TransferMetricName.STORE_TIME: _MetricType.COUNTER,
        _TransferMetricName.STORE_SIZE: _MetricType.HISTOGRAM,
        **{name: _MetricType.GAUGE for name, _ in _ALL_SIMPLE_CPU_GAUGES},
    }
    data = {
        _TransferMetricName.LOAD_BYTES: {(): 4096},
        _TransferMetricName.LOAD_TIME: {(): 0.01},
        _TransferMetricName.LOAD_SIZE: {(): [4096]},
        _TransferMetricName.STORE_BYTES: {(): 8192},
        _TransferMetricName.STORE_TIME: {(): 0.02},
        _TransferMetricName.STORE_SIZE: {(): [8192]},
        **{name: {(): value} for name, value in _ALL_SIMPLE_CPU_GAUGES},
    }

    prom.observe({_StatsKey.TYPES: types, _StatsKey.DATA: data})

    def flat(name):
        return prom.offloading_metrics[(0, name, ())]

    assert flat(_TransferMetricName.LOAD_BYTES).increments == [4096]
    assert flat(_TransferMetricName.LOAD_TIME).increments == [0.01]
    assert flat(_TransferMetricName.LOAD_SIZE).observed == [4096]
    assert flat(_TransferMetricName.STORE_BYTES).increments == [8192]
    assert flat(_TransferMetricName.STORE_TIME).increments == [0.02]
    assert flat(_TransferMetricName.STORE_SIZE).observed == [8192]

    for name, expected in _ALL_SIMPLE_CPU_GAUGES:
        gauge = flat(name)
        assert gauge.set_values == [expected]
        assert gauge.labelvalues == ("model", "0")  # engine labels applied

    # Deprecated transfer_type-labeled mirrors (CPU_to_GPU=load,
    # GPU_to_CPU=store) must also fire, since SimpleCPU always sets
    # _should_observe_deprecated_metrics=True.
    assert prom.counter_kv_bytes[(0, "CPU_to_GPU")].increments == [4096]
    assert prom.counter_kv_transfer_time[(0, "CPU_to_GPU")].increments == [0.01]
    assert prom.histogram_transfer_size[(0, "CPU_to_GPU")].observed == [4096]
    assert prom.counter_kv_bytes[(0, "GPU_to_CPU")].increments == [8192]
    assert prom.counter_kv_transfer_time[(0, "GPU_to_CPU")].increments == [0.02]
    assert prom.histogram_transfer_size[(0, "GPU_to_CPU")].observed == [8192]

    # No double observation of the flat (non-deprecated) metrics: exactly
    # one call recorded per metric despite the deprecated mirror also firing.
    assert len(flat(_TransferMetricName.LOAD_BYTES).increments) == 1
    assert len(flat(_TransferMetricName.LOAD_SIZE).observed) == 1
    assert len(flat(_TransferMetricName.STORE_BYTES).increments) == 1


def test_simple_cpu_prom_metrics_engine_labels_applied_per_engine():
    prom = _make_prom_metrics(
        per_engine_labelvalues={0: ["model", "0"], 1: ["model", "1"]}
    )

    prom.observe(
        {
            _StatsKey.TYPES: {SimpleCPUMetricName.TOTAL_BLOCKS: _MetricType.GAUGE},
            _StatsKey.DATA: {SimpleCPUMetricName.TOTAL_BLOCKS: {(): 42}},
        },
        engine_idx=1,
    )

    assert (0, SimpleCPUMetricName.TOTAL_BLOCKS, ()) not in prom.offloading_metrics
    engine1 = prom.offloading_metrics[(1, SimpleCPUMetricName.TOTAL_BLOCKS, ())]
    assert engine1.set_values == [42]
    assert engine1.labelvalues == ("model", "1")


def test_simple_cpu_prom_metrics_does_not_depend_on_spec_factory():
    """SimpleCPU has no OffloadingSpec; its metric metadata must come from
    the flat transfer defs + SimpleCPU gauge defs only, never the spec
    factory (unlike the generic OffloadPromMetrics default)."""
    with patch.object(
        OffloadingSpecFactory,
        "get_spec_cls",
        side_effect=AssertionError("spec factory should not be consulted"),
    ):
        prom = _make_prom_metrics()

    assert _TransferMetricName.LOAD_BYTES in prom._offloading_metric_metadata
    assert SimpleCPUMetricName.TOTAL_BLOCKS in prom._offloading_metric_metadata
    assert prom._observe_deprecated_metrics is True
