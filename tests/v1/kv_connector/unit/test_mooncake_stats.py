# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from unittest.mock import MagicMock

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    MooncakeConnector,
    MooncakeConnectorWorker,
    SendBlockMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.stats import (
    MooncakeKVConnectorStats,
)


def test_is_empty_on_fresh_stats():
    stats = MooncakeKVConnectorStats()
    assert stats.is_empty()
    assert stats.num_successful_transfers == 0


def test_record_transfer_and_reduce():
    stats = MooncakeKVConnectorStats()
    # 1 MB transfer in 1 ms -> 1000 MB/s throughput
    stats.record_transfer(duration_s=0.001, total_bytes=1 * 2**20, num_descs=4)
    # 2 MB transfer in 2 ms
    stats.record_transfer(duration_s=0.002, total_bytes=2 * 2**20, num_descs=6)
    assert not stats.is_empty()
    assert stats.num_successful_transfers == 2

    reduced = stats.reduce()
    assert reduced["Num successful transfers"] == 2
    # avg = (1 + 2) / 2 = 1.5 ms
    assert reduced["Avg xfer time (ms)"] == 1.5
    assert reduced["Avg MB per transfer"] == 1.5
    # 3 MB total / 3 ms total = 1000 MB/s
    assert reduced["Throughput (MB/s)"] == 1000.0
    assert reduced["Avg number of descriptors"] == 5.0
    assert reduced["Num failed transfers"] == 0
    assert reduced["Num failed recvs"] == 0
    assert reduced["Num KV expired reqs"] == 0


def test_record_failures_keeps_stats_non_empty():
    stats = MooncakeKVConnectorStats()
    stats.record_failed_transfer()
    stats.record_failed_recv()
    stats.record_kv_expired_req()
    assert not stats.is_empty()

    reduced = stats.reduce()
    # No successful transfers -> latency/throughput all zero, but failure
    # counters still surface.
    assert reduced["Num successful transfers"] == 0
    assert reduced["Num failed transfers"] == 1
    assert reduced["Num failed recvs"] == 1
    assert reduced["Num KV expired reqs"] == 1


def test_aggregate_sums_observations():
    a = MooncakeKVConnectorStats()
    b = MooncakeKVConnectorStats()
    a.record_transfer(duration_s=0.001, total_bytes=1 * 2**20, num_descs=1)
    b.record_transfer(duration_s=0.002, total_bytes=2 * 2**20, num_descs=2)
    b.record_failed_transfer()

    a.aggregate(b)

    assert a.num_successful_transfers == 2
    reduced = a.reduce()
    assert reduced["Num successful transfers"] == 2
    assert reduced["Num failed transfers"] == 1


def test_aggregate_with_empty_other_is_noop():
    a = MooncakeKVConnectorStats()
    a.record_transfer(duration_s=0.001, total_bytes=1, num_descs=1)
    b = MooncakeKVConnectorStats()

    a.aggregate(b)

    assert a.num_successful_transfers == 1


def test_getstate_drops_lock_and_setstate_recreates_it():
    # KVConnectorStats subclasses must be picklable (worker→scheduler IPC),
    # but threading.Lock isn't — so __getstate__ strips it and __setstate__
    # rebuilds a fresh per-process lock.
    original = MooncakeKVConnectorStats()
    original.record_transfer(duration_s=0.01, total_bytes=2048, num_descs=3)

    state = original.__getstate__()
    assert "_lock" not in state

    rebuilt = MooncakeKVConnectorStats.__new__(MooncakeKVConnectorStats)
    rebuilt.__setstate__(state)
    assert rebuilt.data == original.data
    # Lock works on the receiver side.
    rebuilt.record_transfer(duration_s=0.02, total_bytes=4096, num_descs=5)
    assert rebuilt.num_successful_transfers == 2


def test_concurrent_writers_keep_row_lengths_aligned():
    # Multiple writers + a snapshot reader must never produce a snapshot
    # with mismatched column lengths — reduce()'s
    # len(descs) == num_successful_transfers assertion would fire.
    stats = MooncakeKVConnectorStats()
    stop = threading.Event()
    writer_count = 4
    snapshots: list[MooncakeKVConnectorStats] = []

    def writer():
        i = 0
        while not stop.is_set():
            stats.record_transfer(
                duration_s=0.001 + i * 1e-9,
                total_bytes=1024 + i,
                num_descs=1 + (i % 8),
            )
            i += 1

    def snapper():
        while not stop.is_set():
            snap = stats.clone_and_reset()
            if not snap.is_empty():
                # Force the same path the logger walks; reduce() will
                # blow up on torn rows via its internal assert.
                snap.reduce()
                snapshots.append(snap)

    threads = [threading.Thread(target=writer) for _ in range(writer_count)]
    snapshotter = threading.Thread(target=snapper)
    for t in threads:
        t.start()
    snapshotter.start()
    # Short fixed window — long enough to interleave thousands of ops.
    threading.Event().wait(0.2)
    stop.set()
    for t in threads:
        t.join()
    snapshotter.join()

    # Final drain so we don't lose the in-flight tail.
    final = stats.clone_and_reset()
    if not final.is_empty():
        final.reduce()
        snapshots.append(final)

    # Every snapshot's columns must have identical lengths (the invariant
    # the lock protects), and the union must contain at least one row.
    total_rows = 0
    for snap in snapshots:
        n = len(snap.data["transfer_duration"])
        assert len(snap.data["bytes_transferred"]) == n
        assert len(snap.data["num_descriptors"]) == n
        total_rows += n
    assert total_rows > 0


def test_clone_and_reset_hands_off_old_data():
    stats = MooncakeKVConnectorStats()
    stats.record_transfer(duration_s=0.001, total_bytes=1, num_descs=1)
    stats.record_failed_recv()

    snapshot = stats.clone_and_reset()

    assert snapshot.num_successful_transfers == 1
    assert not snapshot.is_empty()
    # Original is now empty.
    assert stats.is_empty()
    assert stats.num_successful_transfers == 0
    # Recording on the original does not mutate the snapshot.
    stats.record_transfer(duration_s=0.005, total_bytes=2, num_descs=2)
    assert snapshot.num_successful_transfers == 1


def test_build_kv_connector_stats_none_returns_empty_instance():
    out = MooncakeConnector.build_kv_connector_stats()
    assert isinstance(out, MooncakeKVConnectorStats)
    assert out.is_empty()


def test_build_kv_connector_stats_with_data_round_trips():
    original = MooncakeKVConnectorStats()
    original.record_transfer(duration_s=0.01, total_bytes=1024, num_descs=3)
    original.record_failed_transfer()

    # Serialized form is the .data dict; build should reconstruct an instance
    # that behaves the same.
    rebuilt = MooncakeConnector.build_kv_connector_stats(data=original.data)

    assert isinstance(rebuilt, MooncakeKVConnectorStats)
    assert rebuilt.num_successful_transfers == 1
    assert rebuilt.reduce()["Num failed transfers"] == 1


def _bare_worker() -> MooncakeConnectorWorker:
    """Construct a MooncakeConnectorWorker skipping __init__ (full init requires
    a live TransferEngine). Only the attributes touched by the methods under
    test are populated; role flags and async_zmq_ctx keep __del__'s shutdown
    path a no-op."""
    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker.xfer_stats = MooncakeKVConnectorStats()
    worker.engine = MagicMock()
    worker.async_zmq_ctx = MagicMock()
    worker.is_kv_consumer = True
    worker.is_kv_producer = True
    return worker


def test_send_blocks_records_success():
    worker = _bare_worker()
    worker.engine.batch_transfer_sync_write.return_value = 0

    ret = worker._send_blocks(
        "host:1234",
        src_ptrs=[0x1000, 0x2000],
        dst_ptrs=[0x3000, 0x4000],
        lengths=[1024, 2048],
    )

    assert ret == 0
    assert worker.xfer_stats.num_successful_transfers == 1
    data = worker.xfer_stats.data
    assert data["bytes_transferred"] == [1024 + 2048]
    assert data["num_descriptors"] == [2]
    assert data["num_failed_transfers"] == []


def test_send_blocks_records_failure():
    worker = _bare_worker()
    worker.engine.batch_transfer_sync_write.return_value = 1  # non-zero = fail

    ret = worker._send_blocks("host:1234", [0x1000], [0x2000], [4096])

    assert ret == 1
    assert worker.xfer_stats.num_successful_transfers == 0
    assert worker.xfer_stats.data["num_failed_transfers"] == [1]


def test_get_kv_connector_stats_returns_none_when_empty():
    worker = _bare_worker()

    assert worker.get_kv_connector_stats() is None


def test_get_kv_connector_stats_returns_and_resets():
    worker = _bare_worker()
    worker.engine.batch_transfer_sync_write.return_value = 0
    worker._send_blocks("host:1234", [0x1000], [0x2000], [4096])

    snapshot = worker.get_kv_connector_stats()
    assert isinstance(snapshot, MooncakeKVConnectorStats)
    assert snapshot.num_successful_transfers == 1

    # Second call returns None because the worker's stats were reset.
    assert worker.get_kv_connector_stats() is None


def test_expired_request_bumps_counter():
    import asyncio

    worker = _bare_worker()
    worker.reqs_need_send = {
        "tid1": SendBlockMeta(
            p_req_id="req1",
            transfer_id="tid1",
            local_block_ids=[0, 1],
            ready=asyncio.Event(),
            expire_time=-1.0,  # Already expired.
            sending=0,
        ),
    }
    worker.finished_sending_reqs = set()

    asyncio.run(worker.fetch_finished_sending_reqs())

    assert worker.xfer_stats.data["num_kv_expired_reqs"] == [1]
    # Expired transfer also cleaned out of reqs_need_send.
    assert "tid1" not in worker.reqs_need_send
