# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from prometheus_client import REGISTRY, Counter, Gauge, Histogram

from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorProm
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    MoRIIOMode,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnector,
    MoRIIOConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.stats import (
    MoRIIOKVConnectorStats,
    MoRIIOPromMetrics,
)
from vllm.v1.metrics.prometheus import unregister_vllm_metrics

pytestmark = pytest.mark.cpu_test

TEST_MODEL = "test-model"


def test_reduce_reports_transfers_and_failures():
    stats = MoRIIOKVConnectorStats()
    assert stats.is_empty()
    stats.record_transfer(duration_s=0.001, total_bytes=1 * 2**20, num_descs=4)
    stats.record_transfer(duration_s=0.002, total_bytes=2 * 2**20, num_descs=6)
    stats.record_failed_transfer(2)
    stats.record_failed_notification()
    stats.record_kv_expired_req(3)

    reduced = stats.reduce()

    assert reduced == {
        "Num successful transfers": 2,
        "Avg xfer time (ms)": 1.5,
        "P90 xfer time (ms)": 1.9,
        "Avg MB per transfer": 1.5,
        "Throughput (MB/s)": 1000.0,
        "Avg number of descriptors": 5.0,
        "Num failed transfers": 2,
        "Num failed notifications": 1,
        "Num KV expired reqs": 3,
    }
    assert all(not isinstance(v, np.generic) for v in reduced.values())


def test_failures_alone_are_reported():
    """An interval with only failures must still reach the logger."""
    stats = MoRIIOKVConnectorStats()
    stats.record_failed_notification()

    assert not stats.is_empty()
    reduced = stats.reduce()
    assert reduced["Num successful transfers"] == 0
    assert reduced["Num failed notifications"] == 1


def test_getstate_drops_lock_and_setstate_recreates_it():
    """Worker stats cross a process boundary; the lock must not."""
    stats = MoRIIOKVConnectorStats()
    stats.record_transfer(duration_s=0.01, total_bytes=2048, num_descs=3)

    state = stats.__getstate__()
    assert "_lock" not in state
    rebuilt = MoRIIOKVConnectorStats.__new__(MoRIIOKVConnectorStats)
    rebuilt.__setstate__(state)

    assert rebuilt.data == stats.data
    rebuilt.record_failed_transfer()
    assert rebuilt.reduce()["Num failed transfers"] == 1


def test_writer_thread_and_snapshots_keep_rows_aligned():
    """The MoRIIOWriter thread records while the main thread snapshots."""
    stats = MoRIIOKVConnectorStats()
    stop = threading.Event()
    snapshots: list[MoRIIOKVConnectorStats] = []

    def writer():
        i = 0
        while not stop.is_set():
            stats.record_transfer(1e-3 + i * 1e-9, 1024 + i, 1 + i % 8)
            i += 1

    def snapper():
        while not stop.is_set():
            snapshots.append(stats.clone_and_reset())

    threads = [threading.Thread(target=writer) for _ in range(4)]
    threads.append(threading.Thread(target=snapper))
    for t in threads:
        t.start()
    threading.Event().wait(0.2)
    stop.set()
    for t in threads:
        t.join()
    snapshots.append(stats.clone_and_reset())

    total = 0
    for snap in snapshots:
        n = snap.num_successful_transfers
        assert len(snap.data["bytes_transferred"]) == n
        assert len(snap.data["num_descriptors"]) == n
        total += n
    assert total > 0


def test_build_kv_connector_stats_round_trips_data():
    original = MoRIIOKVConnectorStats()
    original.record_transfer(duration_s=0.01, total_bytes=1024, num_descs=3)
    original.record_kv_expired_req()

    rebuilt = MoRIIOConnector.build_kv_connector_stats(data=original.data)
    empty = MoRIIOConnector.build_kv_connector_stats()

    assert isinstance(rebuilt, MoRIIOKVConnectorStats)
    assert rebuilt.reduce() == original.reduce()
    assert isinstance(empty, MoRIIOKVConnectorStats) and empty.is_empty()


def _read_producer_scheduler(deferred: dict[str, float]) -> Any:
    scheduler = SimpleNamespace(
        is_producer=True,
        mode=MoRIIOMode.READ,
        _defer_timeout=60.0,
        _deferred_send_deadlines={
            req_id: (deadline, f"tx-{req_id}") for req_id, deadline in deferred.items()
        },
        _pending_sent_acks={},
        unmap_request_id=lambda *args, **kwargs: None,
        xfer_stats=MoRIIOKVConnectorStats(),
    )
    scheduler.get_kv_connector_stats = (
        lambda: MoRIIOConnectorScheduler.get_kv_connector_stats(scheduler)
    )
    return scheduler


def test_read_scheduler_reports_expired_sends_through_the_connector():
    """READ-mode reaping frees P blocks D never read; only the scheduler-side
    connector sees it, so it must report the count itself."""
    past = time.monotonic() - 1.0
    scheduler = _read_producer_scheduler({"lost": past, "acked-late": past})
    connector = MoRIIOConnector.__new__(MoRIIOConnector)
    connector.connector_worker = None
    connector.connector_scheduler = scheduler

    output = SimpleNamespace(finished_sending={"acked-late"})
    MoRIIOConnectorScheduler.update_connector_output(scheduler, output)

    assert output.finished_sending == {"lost", "acked-late"}
    snapshot = connector.get_kv_connector_stats()
    assert isinstance(snapshot, MoRIIOKVConnectorStats)
    assert snapshot.reduce()["Num KV expired reqs"] == 1
    assert connector.get_kv_connector_stats() is None


@pytest.fixture(autouse=True)
def _clean_registry():
    unregister_vllm_metrics()
    yield
    unregister_vllm_metrics()


def _sample(series: str, engine_idx: int = 0, **labels: str) -> float | None:
    return REGISTRY.get_sample_value(
        series, {"model_name": TEST_MODEL, "engine": str(engine_idx), **labels}
    )


def _build_prom_metrics(
    per_engine_labelvalues: dict[int, list[object]],
) -> MoRIIOPromMetrics:
    prom_metrics = MoRIIOConnector.build_prom_metrics(
        vllm_config=SimpleNamespace(kv_transfer_config=SimpleNamespace()),  # type: ignore[arg-type]
        metric_types={Gauge: Gauge, Counter: Counter, Histogram: Histogram},
        labelnames=["model_name", "engine"],
        per_engine_labelvalues=per_engine_labelvalues,
    )
    assert isinstance(prom_metrics, MoRIIOPromMetrics)
    return prom_metrics


def _recorded_stats() -> dict[str, list[float | int]]:
    stats = MoRIIOKVConnectorStats()
    stats.record_transfer(duration_s=0.001, total_bytes=1024, num_descs=4)
    stats.record_transfer(duration_s=2.0, total_bytes=2048, num_descs=6)
    stats.record_failed_transfer()
    stats.record_failed_notification()
    stats.record_failed_notification()
    stats.record_kv_expired_req(3)
    return stats.clone_and_reset().to_dict()


def test_kv_connector_prom_wires_the_moriio_metrics():
    prom = KVConnectorProm(
        vllm_config=SimpleNamespace(
            kv_transfer_config=KVTransferConfig(
                kv_connector="MoRIIOConnector", kv_role="kv_producer"
            )
        ),
        labelnames=["model_name", "engine"],
        per_engine_labelvalues={0: [TEST_MODEL, "0"]},
    )

    assert isinstance(prom.prom_metrics, MoRIIOPromMetrics)


def test_observe_exports_every_series_to_the_reported_engine():
    metrics = _build_prom_metrics({0: [TEST_MODEL, "0"], 1: [TEST_MODEL, "1"]})

    metrics.observe(_recorded_stats(), engine_idx=1)

    assert _sample("vllm:moriio_xfer_time_seconds_count", 1) == 2
    assert _sample("vllm:moriio_xfer_time_seconds_sum", 1) == 2.0 + 0.001
    assert _sample("vllm:moriio_xfer_time_seconds_bucket", 1, le="0.005") == 1
    assert _sample("vllm:moriio_bytes_transferred_sum", 1) == 1024 + 2048
    assert _sample("vllm:moriio_num_descriptors_sum", 1) == 4 + 6
    assert _sample("vllm:moriio_num_failed_transfers_total", 1) == 1
    assert _sample("vllm:moriio_num_failed_notifications_total", 1) == 2
    assert _sample("vllm:moriio_num_kv_expired_reqs_total", 1) == 3
    assert _sample("vllm:moriio_xfer_time_seconds_count", 0) == 0
    assert _sample("vllm:moriio_num_kv_expired_reqs_total", 0) == 0


def test_observe_rejects_a_snapshot_missing_a_declared_key():
    metrics = _build_prom_metrics({0: [TEST_MODEL, "0"]})

    with pytest.raises(KeyError):
        metrics.observe({"transfer_duration": []}, engine_idx=0)
