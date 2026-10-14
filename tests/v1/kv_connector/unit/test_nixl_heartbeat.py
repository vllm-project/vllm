# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the scheduler-driven heartbeat / lease-renewal system."""

import time
from unittest.mock import MagicMock, patch

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    HeartbeatInfo,
    NixlConnectorMetadata,
    RemoteMeta,
    ReqMeta,
)
from vllm.v1.outputs import KVConnectorOutput

from .utils import (
    create_request,
    create_vllm_config,
    make_kv_cache_config,
    make_nixl_scheduler,
    maybe_update_block_size,
)

_ENGINE_A = "my-engine-id"


def _sched(kv_lease_duration: int = 30):
    return make_nixl_scheduler(heartbeat=True, kv_lease_duration=kv_lease_duration)


def _req(request_id: int = 1):
    return create_request(request_id=request_id, do_remote_prefill=True)


def _worker_stub():
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
        NixlConnectorWorker,
    )

    w = object.__new__(NixlConnectorWorker)
    w._reqs_to_send = {}
    w._lease_extension = 20
    w._reqs_to_process = set()
    w.consumer_notification_counts_by_req = {}
    w.expected_consumer_notifications_by_req = {}
    w.xfer_stats = MagicMock()
    return w


# ===================================================================
# Scheduler: on_new_request
# ===================================================================


def test_on_new_request_tracks_and_groups():
    """Add two reqs to same engine, one to another; verify grouping."""
    s = _sched()
    s.on_new_request(_req(1))
    s.on_new_request(_req(2))

    assert s._heartbeat_by_engine[_ENGINE_A].req_ids == {"prefill-1", "prefill-2"}
    info = s._heartbeat_by_engine[_ENGINE_A]
    assert (info.host, info.port, info.tp_size) == ("my-host", 1234, 1)
    assert s._heartbeat_req_engine["id-1"] == (_ENGINE_A, "prefill-1")

    # Different engine.
    r3 = _req(3)
    r3.kv_transfer_params["remote_engine_id"] = "engine-b"
    s.on_new_request(r3)
    assert len(s._heartbeat_by_engine) == 2


@pytest.mark.parametrize(
    "make_req",
    [
        lambda: create_request(request_id=2, do_remote_decode=True),
        lambda: create_request(request_id=3),  # no kv_transfer_params
    ],
    ids=["decode", "plain"],
)
def test_on_new_request_ignores_non_prefill(make_req):
    s = _sched()
    s.on_new_request(make_req())
    assert len(s._heartbeat_by_engine) == 0


# ===================================================================
# Scheduler: _stop_heartbeat
# ===================================================================


def test_stop_heartbeat_partial_and_full():
    """Stop one of two reqs on same engine, then stop the other."""
    s = _sched()
    s.on_new_request(_req(1))
    s.on_new_request(_req(2))

    s._stop_heartbeat("id-1")
    assert s._heartbeat_by_engine[_ENGINE_A].req_ids == {"prefill-2"}
    assert "id-1" not in s._heartbeat_req_engine

    s._stop_heartbeat("id-2")
    assert len(s._heartbeat_by_engine) == 0
    assert len(s._heartbeat_req_engine) == 0


# ===================================================================
# Scheduler: build_connector_meta throttling
# ===================================================================


def test_build_connector_meta_heartbeat_throttling():
    # kv_lease_duration=30 => _heartbeat_interval = 30 // 6 = 5
    s = _sched(kv_lease_duration=30)
    s.on_new_request(_req(1))

    # Ensure the first call triggers by placing last_heartbeat far in the past.
    s._last_heartbeat_time = time.perf_counter() - 10
    meta1 = s.build_connector_meta(MagicMock())
    assert _ENGINE_A in meta1.heartbeat_by_engine

    # Immediate second call is throttled (< 5s since last).
    meta2 = s.build_connector_meta(MagicMock())
    assert len(meta2.heartbeat_by_engine) == 0


# ===================================================================
# Scheduler: cleanup paths (update_connector_output / request_finished)
# ===================================================================


def test_update_connector_output_stops_heartbeat():
    s = _sched()
    s.on_new_request(_req(1))

    s.update_connector_output(
        KVConnectorOutput(
            finished_sending=None,
            finished_recving={"id-1"},
            invalid_block_ids=set(),
        )
    )

    assert len(s._heartbeat_by_engine) == 0
    assert len(s._heartbeat_req_engine) == 0


def test_request_finished_stops_heartbeat():
    s = _sched()
    r = _req(1)
    s.on_new_request(r)

    # Simulate update_state_after_alloc having consumed do_remote_prefill.
    r.kv_transfer_params["do_remote_prefill"] = False
    s.request_finished(r, block_ids=())

    assert len(s._heartbeat_by_engine) == 0
    assert len(s._heartbeat_req_engine) == 0


# ===================================================================
# Worker: _handle_heartbeat
# ===================================================================


def test_handle_heartbeat():
    w = _worker_stub()
    far_future = time.perf_counter() + 99999
    w._reqs_to_send = {"req-a": 100.0, "req-b": far_future}

    before = time.perf_counter()
    w._handle_heartbeat("req-a,req-b,req-unknown")

    # req-a: pushed forward to ~now+20.
    assert w._reqs_to_send["req-a"] >= before + 20
    # req-b: already far out, max() keeps it.
    assert w._reqs_to_send["req-b"] >= far_future
    # req-unknown: not added.
    assert "req-unknown" not in w._reqs_to_send


def test_reaper_reclaims_expired_leases_behind_heartbeated_head(monkeypatch):
    """A heartbeated head entry must not strand expired leases behind it.

    Mirrors the stdlib repro in https://github.com/vllm-project/vllm/issues/58222:
    A is inserted first and kept alive via heartbeats; B and C expire and
    must still be reaped. The old early-exit scan returned nothing here.
    """
    w = _worker_stub()
    w._reqs_to_process.update(("A", "B", "C"))
    # Insertion order is A, B, C. Heartbeats will keep A at the head.
    w._reqs_to_send = {"A": 30.0, "B": 32.0, "C": 34.0}

    from vllm.distributed.kv_transfer.kv_connector.v1.nixl import base_worker

    now = {"t": 0.0}
    monkeypatch.setattr(base_worker.time, "perf_counter", lambda: now["t"])
    for t in (20.0, 40.0, 60.0, 80.0):
        now["t"] = t
        w._handle_heartbeat("A")

    # A was renewed in place: still the dict head, expiry 100.
    assert next(iter(w._reqs_to_send)) == "A"
    assert w._reqs_to_send["A"] == pytest.approx(100.0)

    done: set[str] = set()
    now["t"] = 50.0
    w._reap_expired_send_leases(done)

    assert done == {"B", "C"}
    assert list(w._reqs_to_send) == ["A"]
    assert w._reqs_to_process == {"A"}
    assert w.xfer_stats.record_kv_expired_req.call_count == 2

    # now == expires is expired (now >= expires): A is due at 100.
    done_eq: set[str] = set()
    now["t"] = 100.0
    w._reap_expired_send_leases(done_eq)
    assert done_eq == {"A"}
    assert not w._reqs_to_send
    assert not w._reqs_to_process
    assert w.xfer_stats.record_kv_expired_req.call_count == 3


def test_reaper_reclaims_when_now_equals_expiry(monkeypatch):
    """The reaper must reclaim on the now == expires boundary."""
    w = _worker_stub()
    w._reqs_to_process.add("eq")
    w._reqs_to_send = {"eq": 50.0}

    from vllm.distributed.kv_transfer.kv_connector.v1.nixl import base_worker

    monkeypatch.setattr(base_worker.time, "perf_counter", lambda: 50.0)

    done: set[str] = set()
    w._reap_expired_send_leases(done)

    assert done == {"eq"}
    assert not w._reqs_to_send
    assert not w._reqs_to_process
    w.xfer_stats.record_kv_expired_req.assert_called_once()


def test_reaper_reclaims_shorter_lease_behind_later_deadline(monkeypatch):
    """A later-inserted longer TTL must not block an earlier-expiring entry.

    Delete+reinsert on heartbeat cannot restore expiry order in this case
    (issue #58222), so the reaper must scan the whole map.
    """
    w = _worker_stub()
    w._reqs_to_process.update(("X", "Y"))
    w._reqs_to_send = {"X": 55.0, "Y": 46.0}

    from vllm.distributed.kv_transfer.kv_connector.v1.nixl import base_worker

    monkeypatch.setattr(base_worker.time, "perf_counter", lambda: 50.0)

    done: set[str] = set()
    w._reap_expired_send_leases(done)

    assert done == {"Y"}
    assert list(w._reqs_to_send) == ["X"]
    assert w._reqs_to_process == {"X"}


# ===================================================================
# Worker: pull reads check the producer lease, crediting heartbeats
# ===================================================================

_REMOTE = "remote_engine"
_LEASE = 30.0


@pytest.fixture
def pull_worker(dist_init, monkeypatch):
    """Non-bidirectional D worker paired with P, on a fake clock that reads 0
    when P stamps a 30 s lease."""
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl import base_worker

    from .test_nixl_connector import FakeNixlConnectorWorker, FakeNixlWrapper

    monkeypatch.setattr(base_worker, "NixlWrapper", FakeNixlWrapper)
    block_size = maybe_update_block_size(32)
    vllm_config = create_vllm_config(block_size=block_size)
    worker = FakeNixlConnectorWorker(
        vllm_config,
        vllm_config.kv_transfer_config.engine_id,
        hand_shake_latency=0,
        kv_cache_config=make_kv_cache_config(block_size=block_size, num_blocks=2),
    )
    assert worker.REMOTE_ENGINE_ID == _REMOTE
    worker._remote_agents[_REMOTE], _ = worker._nixl_handshake(
        "localhost", 1234, 1, _REMOTE
    )
    worker._engine_clock_offset[_REMOTE] = 0.0
    clock = {"t": 0.0}
    monkeypatch.setattr(base_worker.time, "perf_counter", lambda: clock["t"])
    yield worker, clock
    worker.shutdown()


def _heartbeat_at(worker, clock, t: float) -> None:
    clock["t"] = t
    meta = NixlConnectorMetadata()
    meta.heartbeat_by_engine = {
        _REMOTE: HeartbeatInfo(
            req_ids={"prefill-req"}, host="localhost", port=1234, tp_size=1
        )
    }
    worker.start_load_kv(meta)


def _read_at(worker, clock, t: float) -> None:
    clock["t"] = t
    meta = NixlConnectorMetadata()
    meta.add_new_req_to_recv(
        request_id="req",
        local_block_ids=([0, 1],),
        kv_transfer_params={
            "remote_block_ids": ([0, 1],),
            "remote_engine_id": _REMOTE,
            "remote_request_id": "prefill-req",
            "remote_host": "localhost",
            "remote_port": 1234,
            "remote_blocks_expiry_time": _LEASE,
        },
        awaiting_kvs=True,
    )
    worker.start_load_kv(meta)


_EVERY_5S = [5.0 * i for i in range(1, 9)]


@pytest.mark.parametrize(
    ("heartbeat_times", "send_fails", "read_time", "expect_expired"),
    [
        # Queued past the exported expiry, kept alive by a heartbeat every 5 s.
        pytest.param(_EVERY_5S, False, 45.0, False, id="renewed"),
        # Heartbeats that failed to send extended nothing.
        pytest.param(_EVERY_5S, True, 45.0, True, id="send_failed"),
        # A 30 s heartbeat gap exceeds P's 20 s extension: the lease lapsed.
        pytest.param([5.0, 10.0, 40.0], False, 41.0, True, id="heartbeat_gap"),
        pytest.param([], False, 31.0, True, id="no_heartbeat"),
        # Heartbeats that start this close to the expiry are not credited, so
        # the read is declined past expiry - 5 s.
        pytest.param([20.0, 25.0], False, 26.0, True, id="late_first_heartbeat"),
    ],
)
def test_pull_read_checks_producer_lease(
    pull_worker, heartbeat_times, send_fails, read_time, expect_expired
):
    """D must not read P blocks whose lease P may have reaped and reused."""
    worker, clock = pull_worker
    with patch.object(
        worker.nixl_wrapper,
        "send_notif",
        side_effect=RuntimeError("unreachable") if send_fails else None,
    ):
        for t in heartbeat_times:
            _heartbeat_at(worker, clock, t)
    _read_at(worker, clock, read_time)

    result = worker.get_transfer_results()
    assert result.finished_recving == {"req"}
    assert result.failed_recving == ({"req"} if expect_expired else set())
    assert worker.xfer_stats.data["num_kv_expired_reqs"] == (
        [1] if expect_expired else []
    )
    assert worker.xfer_stats.data["num_failed_transfers"] == []


@pytest.mark.parametrize("read_time", [15.0, 50.0], ids=["loaded", "expired"])
def test_finished_load_drops_heartbeat_record(pull_worker, read_time):
    """Heartbeat state does not outlive the request's load on this worker."""
    worker, clock = pull_worker
    _heartbeat_at(worker, clock, 5.0)
    _read_at(worker, clock, read_time)
    assert worker.get_transfer_results().finished_recving == {"req"}
    assert not worker._heartbeat_sends


def test_pull_read_completing_after_lease_expiry_fails(pull_worker):
    """A read issued in time but finishing after the lease lapsed may hold
    another request's KV, so its blocks are reported invalid."""
    worker, clock = pull_worker
    worker.nixl_wrapper._cycles_before_xfer_done = 1
    _read_at(worker, clock, 10.0)
    assert not worker.get_transfer_results().finished_recving

    clock["t"] = _LEASE
    result = worker.get_transfer_results()
    assert result.finished_recving == result.failed_recving == {"req"}
    assert worker.get_block_ids_with_load_errors() == {0, 1}
    assert worker.xfer_stats.data["num_kv_expired_reqs"] == [1]


@pytest.mark.parametrize("lease", [1, 4])
def test_heartbeat_credit_never_outlives_producer_lease(pull_worker, lease):
    """P extends by lease * 2 // 3 whole seconds; D's credited deadline must
    still pass no later than P reaps the blocks."""
    worker, clock = pull_worker
    producer = _worker_stub()
    producer._lease_extension = lease * 2 // 3
    producer._reqs_to_send = {"prefill-req": float(lease)}
    meta = ReqMeta(
        local_block_ids=([0],),
        local_physical_block_ids=([0],),
        tp_size=1,
        remote=RemoteMeta(
            ([0],), "localhost", 1234, _REMOTE, "prefill-req", float(lease)
        ),
    )
    for step in range(1, 80 * lease):
        t = step / 20
        if t <= 2 * lease and step % 2 == 0:
            _heartbeat_at(worker, clock, t)
            producer._handle_heartbeat("prefill-req")
        clock["t"] = t
        if t >= producer._reqs_to_send["prefill-req"]:
            assert worker._is_lease_expired(meta), t
