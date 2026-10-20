# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoRIIO sleep mode: the KV cache is deregistered only after every peer
that fetched its region metadata dropped it and finished its transfers, and
is registered again after wake-up.

A transfer against a deregistered region fails with a remote access error that
leaves the peer's queue pair unusable, so peers must forget the metadata first.
"""

import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import msgpack
import msgspec
import pytest
import zmq

from vllm.distributed.kv_transfer.kv_connector.v1.moriio import moriio_engine
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    ROLE,
    HandshakeError,
    HandshakeTimeoutError,
    MoRIIOAgentMetadata,
    MoRIIOConstants,
    MoRIIOError,
    PeerUnavailableError,
    RemoteAllocInfo,
    TransferError,
    get_port_offset,
    zmq_ctx,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnectorScheduler,
    MoRIIOConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine import (
    MoRIIOWrapper,
    MoRIIOWriter,
)
from vllm.utils.network_utils import get_open_port, make_zmq_path

W = MoRIIOConnectorWorker


def _worker(
    region_metas: dict[str, list[bytes]], region_tensors: dict
) -> tuple[SimpleNamespace, list[tuple[str, object]]]:
    registered = iter(f"new{i}".encode() for i in range(100))
    calls: list[tuple[str, object]] = []
    wrapper = MagicMock()
    wrapper.get_unpack_memory_metadata.side_effect = lambda meta: meta
    wrapper.moriio_engine.deregister_memory.side_effect = lambda desc: calls.append(
        ("deregister", desc)
    )
    wrapper.register_local_tensor.side_effect = lambda tensor: next(registered)

    def invalidate_peers(peers) -> list[str]:
        calls.append(("invalidate", peers))
        return []  # Every peer confirmed.

    worker = SimpleNamespace(
        moriio_wrapper=wrapper,
        layer_name_to_local_kv_cache_metadata=region_metas,
        _region_tensors=region_tensors,
        _kv_released=False,
        _peer_lock=threading.Lock(),
        _handshake_lock=threading.RLock(),
        _peer_listeners={},
        built_write_session=defaultdict(list, {"peer_dp0": ["session"]}),
        _invalidate_peers=invalidate_peers,
        _drain_transfers=lambda: calls.append(("drain", None)),
        _all_tp_ranks_ok=lambda ok: ok,
    )
    worker._deregister_regions = partial(W._deregister_regions, worker)
    worker._register_regions = partial(W._register_regions, worker)
    return worker, calls


def test_each_peer_worker_is_recorded_and_metadata_withheld_while_unregistered():
    worker, _ = _worker({"l0": [b"m"]}, {"l0": [object()]})
    # The TP/DP workers of one peer engine share its engine ID.
    for port in (1000, 1001):
        peer = {"engine_id": "peer", "host": "h", "port": port}
        assert msgpack.loads(W._serve_kv_metadata(worker, peer)) == {"l0": [b"m"]}
    assert worker._peer_listeners == {("h", 1000): "peer", ("h", 1001): "peer"}

    worker._kv_released = True
    late = {"engine_id": "late", "host": "h", "port": 9}
    assert W._serve_kv_metadata(worker, late) is None
    assert ("h", 9) not in worker._peer_listeners


def test_unregister_invalidates_peers_drains_then_deregisters_each_region_once():
    shared = object()
    worker, calls = _worker(
        {"l0": [b"m"], "l1": [b"m"]}, {"l0": [shared], "l1": [shared]}
    )
    peers = {("h", 1000): "peer", ("h", 1001): "peer"}
    worker._peer_listeners = dict(peers)

    W.unregister_kv_caches(worker)
    W.unregister_kv_caches(worker)

    assert calls == [("invalidate", peers), ("drain", None), ("deregister", b"m")]
    assert worker._kv_released
    assert not worker._peer_listeners
    assert not worker.built_write_session


def test_restore_registers_each_region_once_and_serves_it_again():
    shared = MagicMock(data_ptr=lambda: 0x1000, nbytes=64)
    metas = {"l0": [b"m"], "l1": [b"m"]}
    worker, _ = _worker(metas, {"l0": [shared], "l1": [shared]})

    W.unregister_kv_caches(worker)
    W.restore_kv_caches(worker)

    # The listener serves this dict, so it must be updated, not replaced.
    assert worker.layer_name_to_local_kv_cache_metadata is metas
    assert metas == {"l0": [b"new0"], "l1": [b"new0"]}
    assert worker.moriio_wrapper.register_local_tensor.call_count == 1
    assert not worker._kv_released
    peer = {"engine_id": "p", "host": "h", "port": 1}
    assert msgpack.loads(W._serve_kv_metadata(worker, peer)) == metas


def test_unregister_deregisters_every_region_even_if_one_fails():
    worker, calls = _worker({"l0": [b"a"], "l1": [b"b"]}, {"l0": [1], "l1": [2]})

    def deregister(desc):
        calls.append(("deregister", desc))
        if desc == b"a":
            raise RuntimeError("busy")

    worker.moriio_wrapper.moriio_engine.deregister_memory.side_effect = deregister

    W.unregister_kv_caches(worker)

    assert calls[-2:] == [("deregister", b"a"), ("deregister", b"b")]
    assert worker._kv_released


def _expect_sleep_refused(worker, calls, match: str) -> None:
    with pytest.raises(MoRIIOError, match=match):
        W.unregister_kv_caches(worker)
    # Still registered and served: a transfer may still use the memory.
    assert not [c for c in calls if c[0] == "deregister"]
    assert not worker._kv_released
    assert worker._peer_listeners == {("h", 1000): "peer"}
    peer = {"engine_id": "p", "host": "h", "port": 1}
    assert W._serve_kv_metadata(worker, peer) is not None


def test_sleep_is_refused_while_a_peer_may_still_transfer_to_us():
    worker, calls = _worker({"l0": [b"m"]}, {"l0": [object()]})
    worker._peer_listeners = {("h", 1000): "peer"}
    # The peer could not drain (negative ack) or did not answer.
    worker._invalidate_peers = lambda peers: ["peer at tcp://h:1000 (no answer)"]
    _expect_sleep_refused(worker, calls, "no answer")

    # The sleep can be retried once the peer confirms.
    worker._invalidate_peers = lambda peers: []
    W.unregister_kv_caches(worker)
    assert calls[-1] == ("deregister", b"m")
    assert worker._kv_released


def test_sleep_is_refused_while_our_own_transfers_are_pending():
    worker, calls = _worker({"l0": [b"m"]}, {"l0": [object()]})
    worker._peer_listeners = {("h", 1000): "peer"}
    worker._drain_transfers = MagicMock(side_effect=TransferError("1 pending"))
    _expect_sleep_refused(worker, calls, "1 pending")


def test_sleep_is_refused_on_every_tp_rank_if_it_fails_on_one(monkeypatch):
    import torch.distributed as dist

    worker, calls = _worker({"l0": [b"m"]}, {"l0": [object()]})
    worker._peer_listeners = {("h", 1000): "peer"}
    worker._all_tp_ranks_ok = partial(W._all_tp_ranks_ok, worker)
    worker.world_size = 2
    worker.tp_group = SimpleNamespace(cpu_group=None)
    # Fine on this rank; another rank voted 0.
    monkeypatch.setattr(dist, "all_reduce", lambda vote, group, op: vote.fill_(0))
    _expect_sleep_refused(worker, calls, "another TP rank")


def test_failed_restore_releases_what_it_registered_and_can_be_retried():
    tensors = [MagicMock(data_ptr=lambda i=i: i, nbytes=64) for i in range(2)]
    worker, calls = _worker(
        {"l0": [b"a"], "l1": [b"b"]}, {"l0": [tensors[0]], "l1": [tensors[1]]}
    )
    W.unregister_kv_caches(worker)
    calls.clear()
    worker.moriio_wrapper.register_local_tensor.side_effect = [b"new0", OSError("x")]

    with pytest.raises(OSError):
        W.restore_kv_caches(worker)

    assert calls == [("deregister", b"new0")]
    assert worker._kv_released
    worker.moriio_wrapper.register_local_tensor.side_effect = [b"r0", b"r1"]
    W.restore_kv_caches(worker)
    assert worker.layer_name_to_local_kv_cache_metadata == {
        "l0": [b"r0"],
        "l1": [b"r1"],
    }


def test_sleep_mode_refuses_regions_sharing_a_start_address():
    shared = MagicMock(data_ptr=lambda: 0x1000)
    other = MagicMock(data_ptr=lambda: 0x2000)
    # One registration shared by several layers is fine.
    worker = SimpleNamespace(
        _region_tensors={"a": [shared], "b": [shared], "c": [other]},
        layer_name_to_local_kv_cache_metadata={"a": [b"m"], "b": [b"m"], "c": [b"o"]},
    )
    W._check_regions_releasable(worker)
    # Two registrations at one address: MoRI would deregister only one.
    worker.layer_name_to_local_kv_cache_metadata["b"] = [b"m2"]
    with pytest.raises(MoRIIOError, match="does not support this KV cache layout"):
        W._check_regions_releasable(worker)


def test_drop_remote_engine_forgets_only_that_peers_regions_then_drains():
    drained = []
    worker = SimpleNamespace(
        _handshake_lock=threading.RLock(),
        _remote_epoch=0,
        _invalidated_at={},
        _is_peer_key=W._is_peer_key,
        layer_name_to_remote_kv_cache_metadata={"A_dp0": {}, "B_dp0": {}},
        remote_moriio_metadata={"A_dp0": 1, "B_dp0": 2},
        _remote_agents={"A": {"x"}, "A_dp0": {"x"}, "AB_dp0": {"y"}},
        _handshake_futures={"A_dp0": None},
        built_write_session=defaultdict(list, {"A_dp0": [1], "B_dp0": [2]}),
        _eager_handshaked_engines={"A_dp0", "B_dp0"},
        _drain_transfers=lambda: drained.append(True),
    )

    W._drop_remote_engine(worker, "A")

    assert set(worker.layer_name_to_remote_kv_cache_metadata) == {"B_dp0"}
    assert set(worker.remote_moriio_metadata) == {"B_dp0"}
    # The connection is kept: only the region metadata is fetched again.
    assert set(worker._remote_agents) == {"A", "A_dp0", "AB_dp0"}
    assert worker._handshake_futures == {"A_dp0": None}
    assert set(worker.built_write_session) == {"B_dp0"}
    # Untouched: TP ranks handle the invalidation at different times, and
    # this set decides whether a rank joins the eager handshake's all-reduce.
    assert worker._eager_handshaked_engines == {"A_dp0", "B_dp0"}
    assert drained == [True]
    # Handshakes that started before the drop must not store what they received
    # from that peer; handshakes with other peers are unaffected.
    assert W._dropped_since(worker, 0, "A_dp1")
    assert not W._dropped_since(worker, 0, "B_dp0")
    assert not W._dropped_since(worker, 1, "A_dp1")

    # However many other peers are invalidated meanwhile.
    for i in range(100):
        W._drop_remote_engine(worker, f"C{i}")
    assert W._dropped_since(worker, 0, "A_dp1")


def test_peer_is_not_acknowledged_while_our_transfers_to_it_are_pending():
    worker = SimpleNamespace(
        _transfer_post_lock=threading.RLock(),
        moriio_config=SimpleNamespace(transfer_timeout=0.1),
        moriio_wrapper=SimpleNamespace(drain_transfers=lambda timeout: 1),
    )
    # The handshake listener then sends INVALIDATE_NACK.
    with pytest.raises(TransferError, match="1 MoRIIO transfer"):
        W._drain_transfers(worker)


def test_drop_matches_the_engine_id_the_peer_named_itself():
    # Stored under the address this engine dialed (a hostname), not the
    # peer's own engine ID (its IP).
    peer = SimpleNamespace(engine_id="10.0.0.1:5000")
    worker = SimpleNamespace(
        _handshake_lock=threading.RLock(),
        _remote_epoch=0,
        _invalidated_at={},
        _is_peer_key=W._is_peer_key,
        layer_name_to_remote_kv_cache_metadata={"pod-a:5000_dp0": {}},
        remote_moriio_metadata={"pod-a:5000_dp0": peer},
        built_write_session=defaultdict(list, {"pod-a:5000_dp0": [1]}),
        _drain_transfers=lambda: None,
    )

    W._drop_remote_engine(worker, "10.0.0.1:5000")

    assert not worker.layer_name_to_remote_kv_cache_metadata
    assert not worker.remote_moriio_metadata
    assert not worker.built_write_session
    # A handshake in flight to it then discards what it receives.
    assert W._dropped_since(worker, 0, "pod-a:5000_dp1", "10.0.0.1:5000")


def _session_worker(handshake):
    worker = SimpleNamespace(
        _kv_released=False,
        _handshake_lock=threading.RLock(),
        layer_name_to_remote_kv_cache_metadata={},
        layer_name_to_local_kv_cache_metadata={"l0": [b"local"]},
        remote_moriio_metadata={},
        built_write_session=defaultdict(list),
        _handshake_args={"A_dp0": ("h", 1, 1, "A_dp0", 0, None)},
        _moriio_handshake=handshake,
        _handshake_initiation_executor=ThreadPoolExecutor(max_workers=1),
        _refresh_timed_out={},
        moriio_config=SimpleNamespace(transfer_timeout=30.0),
        moriio_wrapper=MagicMock(
            get_unpack_memory_metadata=lambda m: m,
            build_session=lambda local, remote: (local, remote),
        ),
        _remote_epoch=0,
        _invalidated_at={},
        _is_peer_key=W._is_peer_key,
    )
    # The methods these call on self.
    worker._refresh_remote_metadata = partial(W._refresh_remote_metadata, worker)
    worker._build_sessions = partial(W._build_sessions, worker)
    worker._dropped_since = partial(W._dropped_since, worker)
    return worker


def test_session_lookup_refetches_regions_of_a_peer_that_re_registered():
    def handshake(*args):
        worker.layer_name_to_remote_kv_cache_metadata["A_dp0"] = {"l0": [b"new"]}
        worker.remote_moriio_metadata["A_dp0"] = "meta"
        return {"agent"}

    worker = _session_worker(MagicMock(side_effect=handshake))

    W._prepare_transfer_to(worker, "A_dp0")
    sessions, meta = W._get_built_session_locked(worker, "A_dp0")

    worker._moriio_handshake.assert_called_once_with("h", 1, 1, "A_dp0", 0, None)
    assert sessions == [(b"local", b"new")]
    assert meta == "meta"
    # Known metadata: no handshake.
    W._refresh_remote_metadata(worker, "A_dp0")
    worker._moriio_handshake.assert_called_once()


def test_sessions_are_built_without_locks_and_dropped_if_the_peer_left():
    worker = _session_worker(MagicMock())
    worker.layer_name_to_remote_kv_cache_metadata["A_dp0"] = {"l0": [b"r"]}

    def build_session(local, remote):
        # MoRI may ask the peer over the network: no lock is held, and the
        # peer invalidates meanwhile (dropping its metadata, under any name).
        assert worker._handshake_lock.acquire(blocking=False)
        worker._handshake_lock.release()
        del worker.layer_name_to_remote_kv_cache_metadata["A_dp0"]
        return "session"

    worker.moriio_wrapper.build_session = build_session
    W._prepare_transfer_to(worker, "A_dp0")
    assert "A_dp0" not in worker.built_write_session

    # Invalidated and fetched again (new metadata) during the build.
    worker.layer_name_to_remote_kv_cache_metadata["A_dp0"] = {"l0": [b"r"]}

    def build_then_refetch(local, remote):
        worker.layer_name_to_remote_kv_cache_metadata["A_dp0"] = {"l0": [b"r2"]}
        return "session"

    worker.moriio_wrapper.build_session = build_then_refetch
    W._prepare_transfer_to(worker, "A_dp0")
    assert "A_dp0" not in worker.built_write_session

    # A repeated handshake returned the same regions: the sessions are valid.
    def build_then_repeat(local, remote):
        worker.layer_name_to_remote_kv_cache_metadata["A_dp0"] = {"l0": [b"r2"]}
        return "session"

    worker.moriio_wrapper.build_session = build_then_repeat
    W._prepare_transfer_to(worker, "A_dp0")
    assert worker.built_write_session["A_dp0"] == ["session"]


def test_session_build_failing_because_the_peer_left_is_unavailability():
    worker = _session_worker(MagicMock())
    worker.layer_name_to_remote_kv_cache_metadata["A_dp0"] = {"l0": [b"r"]}

    def build_session(local, remote):
        # The peer deregistered while it went to sleep: MoRI cannot find
        # the region on it.
        del worker.layer_name_to_remote_kv_cache_metadata["A_dp0"]
        raise RuntimeError("remote memory region not found")

    worker.moriio_wrapper.build_session = build_session
    with pytest.raises(PeerUnavailableError):
        W._prepare_transfer_to(worker, "A_dp0")

    # Any other failure is not hidden.
    worker.layer_name_to_remote_kv_cache_metadata["A_dp0"] = {"l0": [b"r"]}
    worker.moriio_wrapper.build_session = MagicMock(side_effect=RuntimeError("x"))
    with pytest.raises(RuntimeError):
        W._prepare_transfer_to(worker, "A_dp0")


def test_metadata_refresh_holds_no_lock_while_waiting_on_the_peer():
    # Other transfers, unregister_kv_caches and invalidation handling take
    # _handshake_lock; a slow or dead peer must not stall them.
    acquired_elsewhere: list[bool] = []

    def handshake(*args):
        def try_lock():
            ok = worker._handshake_lock.acquire(timeout=1)
            acquired_elsewhere.append(ok)
            if ok:
                worker._handshake_lock.release()

        t = threading.Thread(target=try_lock)
        t.start()
        t.join()
        raise HandshakeError("timed out")

    worker = _session_worker(handshake)
    with pytest.raises(PeerUnavailableError):
        W._refresh_remote_metadata(worker, "A_dp0")
    assert acquired_elsewhere == [True]


def test_session_lookup_fails_cleanly_while_the_peer_is_asleep():
    worker = _session_worker(MagicMock(side_effect=HandshakeError("asleep")))
    with pytest.raises(PeerUnavailableError, match="cannot refresh"):
        W._refresh_remote_metadata(worker, "A_dp0")
    # Invalidated after the refresh: fail without a network round trip.
    with pytest.raises(PeerUnavailableError, match="no region metadata"):
        W._get_built_session_locked(worker, "A_dp0")
    worker._moriio_handshake.assert_called_once()


def test_peer_that_timed_out_is_not_dialed_again_for_transfer_timeout():
    worker = _session_worker(MagicMock(side_effect=HandshakeTimeoutError("t/o")))
    for _ in range(3):
        with pytest.raises(PeerUnavailableError):
            W._refresh_remote_metadata(worker, "A_dp0")
    worker._moriio_handshake.assert_called_once()
    # An asleep peer answers at once, so it is asked again every time.
    worker = _session_worker(MagicMock(side_effect=HandshakeError("asleep")))
    for _ in range(2):
        with pytest.raises(PeerUnavailableError):
            W._refresh_remote_metadata(worker, "A_dp0")
    assert worker._moriio_handshake.call_count == 2


def test_handshake_gives_up_on_a_peer_that_does_not_answer():
    port = get_open_port()
    with zmq_ctx(zmq.ROUTER, make_zmq_path("tcp", "127.0.0.1", port)) as silent:
        worker = SimpleNamespace(
            _remote_epoch=0,
            engine_id="me",
            local_ip="127.0.0.1",
            side_channel_port=0,
            moriio_config=SimpleNamespace(transfer_timeout=0.3),
        )
        start = time.monotonic()
        with pytest.raises(HandshakeError, match="did not answer"):
            W._moriio_handshake(worker, "127.0.0.1", port, 1, "peer", 0, 0)
        assert time.monotonic() - start < 5
        assert silent.poll(1000)  # the query did reach the peer


def _read_worker(**attrs):
    worker = SimpleNamespace(
        _has_mamba=False,
        _invalid_block_ids=set(),
        world_size=2,
        moriio_wrapper=MagicMock(),
        _resolve_read_source=lambda meta: (1, False),
    )
    worker.__dict__.update(attrs)
    return worker


_READ_META = SimpleNamespace(
    remote_engine_id="P",
    local_block_ids=[[3, 4]],
    remote_host="10.0.0.1",
    remote_notify_port=7000,
    remote_dp_rank=0,
    tp_size=2,
    transfer_id="tx",
)


def test_read_from_an_unavailable_peer_fails_the_load_not_the_step():
    worker = _read_worker()
    worker._fail_read = lambda *args: W._fail_read(worker, *args)

    def read(*args):
        raise PeerUnavailableError("asleep")

    worker._read_session_blocks = read

    W._read_blocks_for_req(worker, "req", _READ_META)

    # The scheduler recomputes or fails the request per kv_load_failure_policy.
    assert worker._invalid_block_ids == {3, 4}
    worker.moriio_wrapper.send_notify.assert_called_once_with(
        "tx",
        "10.0.0.1",
        str(7000 + get_port_offset(0, 1, 2)),
        message_type="release",
        message_fields={"consumer_tp_size": 2},
    )


def test_hybrid_read_errors_fail_the_step_on_every_tp_rank(monkeypatch):
    import torch.distributed as dist

    worker = _read_worker(
        _has_mamba=True,
        _failed_hybrid_reads=[],
        tp_group=SimpleNamespace(cpu_group=None),
    )
    W._fail_read(worker, "req", _READ_META, "asleep")
    # No load-error recovery for hybrid models: the step fails closed.
    assert worker._invalid_block_ids == set()

    monkeypatch.setattr(dist, "all_reduce", lambda vote, group, op: None)
    with pytest.raises(PeerUnavailableError, match="req: asleep"):
        W._fail_step_on_hybrid_read_errors(worker)
    # Fine here, failed on another rank: fail together.
    monkeypatch.setattr(dist, "all_reduce", lambda vote, group, op: vote.fill_(1))
    with pytest.raises(PeerUnavailableError, match="other rank"):
        W._fail_step_on_hybrid_read_errors(worker)
    monkeypatch.setattr(dist, "all_reduce", lambda vote, group, op: None)
    W._fail_step_on_hybrid_read_errors(worker)


def _write_consumer_scheduler(stop_after: float | None):
    S = MoRIIOConnectorScheduler
    scheduler = S.__new__(S)
    scheduler.is_producer = False
    scheduler._global_dp_rank = 0
    scheduler._stop_write_after = stop_after
    scheduler._write_stop_due = {}
    scheduler._awaiting_write = {}
    scheduler._unmap_after_recv = {}
    scheduler.paths = {f"tcp://h:{port}": MagicMock() for port in (1, 2)}
    for port in (1, 2):  # One producer TP rank each.
        S.send_notify_block(scheduler, "req", "tx", [5], host="h", port=port)
    scheduler._awaiting_write["tx"] = "req"
    return scheduler


def test_overdue_write_load_asks_the_producer_to_stop_and_keeps_its_blocks():
    S = MoRIIOConnectorScheduler
    scheduler = _write_consumer_scheduler(stop_after=10.0)
    stop = call(msgpack.dumps({"type": "stop_write", "transfer_id": "tx"}))

    S._stop_overdue_writes(scheduler)
    for sock in scheduler.paths.values():
        assert stop not in sock.send.mock_calls

    due, paths = scheduler._write_stop_due["tx"]
    scheduler._write_stop_due["tx"] = (time.monotonic() - 1, paths)
    S._stop_overdue_writes(scheduler)
    for sock in scheduler.paths.values():
        assert sock.send.mock_calls[-1] == stop
    # No timer frees the blocks: only the producer's answer does, and the
    # stop is sent again if none comes.
    assert scheduler._awaiting_write == {"tx": "req"}
    assert scheduler._write_stop_due["tx"][0] > time.monotonic() + 5

    # The producer's write_failed (or write_done) completes the load.
    S.update_connector_output(scheduler, SimpleNamespace(finished_recving={"req"}))
    assert scheduler._write_stop_due == {}


def test_write_loads_are_stopped_only_with_sleep_mode():
    # Without sleep mode nothing waits for the engine to drain, and a busy
    # prefill must not turn loads into failures.
    scheduler = _write_consumer_scheduler(stop_after=None)
    assert scheduler._write_stop_due == {}


def test_aborted_write_load_stays_mapped_until_it_completes_or_fails():
    S = MoRIIOConnectorScheduler
    scheduler = S.__new__(S)
    scheduler.is_producer = False
    scheduler._awaiting_write = {"tx": "req"}
    scheduler._write_stop_due = {}
    scheduler._unmap_after_recv = {}
    scheduler.unmap_request_id = MagicMock()
    request = SimpleNamespace(
        request_id="req",
        status=None,
        kv_transfer_params={"transfer_id": "tx", "do_remote_prefill": False},
    )

    assert S.request_finished(scheduler, request, block_ids=[]) == (False, None)
    # The worker must still map the completion so the scheduler frees the
    # blocks, and the producer can still be asked to stop the load.
    scheduler.unmap_request_id.assert_not_called()
    assert scheduler._awaiting_write == {"tx": "req"}

    S.update_connector_output(scheduler, SimpleNamespace(finished_recving={"req"}))
    scheduler.unmap_request_id.assert_called_once_with("req", transfer_id="tx")
    assert scheduler._awaiting_write == {}
    assert scheduler._unmap_after_recv == {}


class _Status:
    def __init__(self, state: str = "pending"):
        self.state = state

    def Succeeded(self) -> bool:
        return self.state == "ok"

    def Failed(self) -> bool:
        return self.state == "failed"


def test_tracking_prunes_finished_transfers_in_amortized_time():
    wrapper = SimpleNamespace(lock=threading.Lock(), transfer_status=[])
    wrapper._track_prune_at = 4
    live = [_Status() for _ in range(3)]
    for status in [*live, _Status("ok")]:
        MoRIIOWrapper._track(wrapper, status)
    # Full: the finished one is dropped, and the next prune waits until the
    # list is twice the live transfers (at least _MIN_TRACK_PRUNE_AT).
    MoRIIOWrapper._track(wrapper, live[0])
    assert wrapper.transfer_status == [*live, live[0]]
    assert wrapper._track_prune_at == 1024


def test_drain_waits_past_failures_and_keeps_tracking_unfinished_transfers():
    failed, finishing, stuck = _Status("failed"), _Status(), _Status()
    wrapper = SimpleNamespace(
        lock=threading.Lock(), transfer_status=[], _track_prune_at=1024
    )
    for status in (failed, finishing, stuck):
        MoRIIOWrapper._track(wrapper, status)
    threading.Timer(0.2, lambda: setattr(finishing, "state", "ok")).start()

    start = time.monotonic()
    assert MoRIIOWrapper.drain_transfers(wrapper, timeout=0.5) == 1
    assert time.monotonic() - start >= 0.5
    assert finishing.Succeeded()
    # A timeout does not cancel RDMA: the stuck transfer stays tracked, so no
    # later drain reports success while it may still touch the memory.
    assert wrapper.transfer_status == [stuck]
    assert MoRIIOWrapper.drain_transfers(wrapper, timeout=0.1) == 1
    stuck.state = "ok"
    assert MoRIIOWrapper.drain_transfers(wrapper, timeout=5) == 0
    assert wrapper.transfer_status == []


class _WriterWorker:
    """What MoRIIOWriter uses of its worker (which it references weakly)."""

    def __init__(self, wrapper: MoRIIOWrapper):
        self.moriio_wrapper = wrapper
        self.moriio_config = SimpleNamespace(defer_timeout=60.0, transfer_timeout=0.2)


def _writer(**allocations: RemoteAllocInfo):
    wrapper = MoRIIOWrapper()
    wrapper.done_remote_allocate_req_dict.update(allocations)
    worker = _WriterWorker(wrapper)
    writer = MoRIIOWriter(worker)
    writer.worker_ref = worker  # keep it alive
    writer.ensure_worker_started = lambda: None
    events: list[str] = []
    writer._send_write_failed = lambda task: events.append(task.transfer_id)
    return writer, wrapper, events


def _task(layer_name: str = "l0") -> SimpleNamespace:
    return SimpleNamespace(transfer_id="tx", request_id="req", layer_name=layer_name)


def _released(wrapper: MoRIIOWrapper) -> list[str]:
    """Transfers whose source blocks the producer released."""
    return [ack.transfer_id for ack in wrapper.done_req_ids]


def test_failed_write_keeps_its_blocks_until_posted_writes_settle():
    posted = _Status()
    writer, wrapper, events = _writer(
        tx=RemoteAllocInfo(block_ids=[1], transfer_statuses=[posted])
    )

    writer._fail_write(_task())
    # The NIC may still read our source blocks and write the consumer's.
    assert events == [] and _released(wrapper) == []
    assert not writer._is_transfer_terminal("tx")

    posted.state = "ok"
    writer._process_failing_writes()
    writer._process_failing_writes()
    assert events == ["tx"] and _released(wrapper) == ["tx"]
    assert writer._is_transfer_terminal("tx")


def test_write_that_never_settles_never_releases_its_blocks():
    writer, wrapper, events = _writer(
        tx=RemoteAllocInfo(block_ids=[1], transfer_statuses=[_Status()])
    )

    writer._fail_write(_task())
    time.sleep(0.3)  # past transfer_timeout
    writer._process_failing_writes()

    assert events == [] and _released(wrapper) == []
    assert writer._failing_writes["tx"].warned


def test_write_failing_at_completion_is_reported_only_after_its_writes_settle():
    # Waiting on a transfer's writes stops at the first failure.
    pending = _Status()
    info = RemoteAllocInfo(
        block_ids=[1],
        writes_done=1,
        writes_expected=1,
        completion_request_id="req",
        completion_notify_port=1,
        completion_remote_ip="h",
        transfer_statuses=[_Status("failed"), pending],
    )
    writer, wrapper, events = _writer(tx=info)

    def wait(statuses):
        raise TransferError("one write failed")

    wrapper.waiting_for_transfer_complete = wait
    with pytest.raises(TransferError):
        writer._finalize_if_complete("tx", info)
    writer._fail_write(_task())
    assert events == [] and _released(wrapper) == []

    pending.state = "failed"
    writer._process_failing_writes()
    assert events == ["tx"] and _released(wrapper) == ["tx"]


def test_consumer_stop_closes_the_transfer_and_fails_it_once_writes_settle():
    posted = _Status()
    info = RemoteAllocInfo(block_ids=[1], transfer_statuses=[posted])
    writer, wrapper, events = _writer(tx=info)
    writer.schedule_write(_task("l0"))
    with patch.object(moriio_engine, "get_role", return_value=ROLE.PRODUCER):
        wrapper._handle_structured_message({"type": "stop_write", "transfer_id": "tx"})

    writer._process_stop_requests()
    assert events == [] and _released(wrapper) == []
    # No later write of the transfer is posted, and it cannot complete normally.
    writer._execute_write_task = MagicMock()
    writer._deferred_tasks = [_task("l1")]
    writer._process_deferred_tasks()
    writer._execute_write_task.assert_not_called()
    info.writes_expected, info.completion_request_id = 0, "req"
    info.completion_notify_port, info.completion_remote_ip = 1, "h"
    writer._finalize_if_complete("tx", info)
    assert not info.completion_notified

    posted.state = "ok"
    writer._process_failing_writes()
    assert events == ["tx"] and _released(wrapper) == ["tx"]


def test_stop_before_the_first_write_fails_the_transfer_when_its_task_arrives():
    writer, wrapper, events = _writer(tx=RemoteAllocInfo(block_ids=[1]))
    wrapper.stop_write_req_ids.add("tx")

    writer._process_stop_requests()
    assert events == []

    writer._execute_write_task = MagicMock()
    writer._deferred_tasks = [_task()]
    writer._process_deferred_tasks()
    writer._execute_write_task.assert_not_called()
    assert events == ["tx"] and _released(wrapper) == ["tx"]


def test_stop_for_a_finished_transfer_is_ignored():
    writer, wrapper, events = _writer()
    with wrapper.lock:
        wrapper._mark_transfer_terminal_locked("tx")
    with patch.object(moriio_engine, "get_role", return_value=ROLE.PRODUCER):
        wrapper._handle_structured_message({"type": "stop_write", "transfer_id": "tx"})

    writer._process_stop_requests()
    assert events == [] and writer._closed_writes == set()


def _start_listener(serve, on_invalidate) -> int:
    port = get_open_port()
    metadata = MoRIIOAgentMetadata(
        engine_id="owner",
        agent_metadata=b"agent",
        kv_caches_base_addr=[0],
        num_blocks=1,
        block_len=1,
        attn_backend_name="test",
    )
    ready = threading.Event()
    threading.Thread(
        target=W._moriio_handshake_listener,
        args=(metadata, ready, port, 0, 0, serve, on_invalidate),
        daemon=True,
    ).start()
    assert ready.wait(10)
    return port


def test_listener_serves_metadata_until_unregistered_and_acks_invalidation():
    peers: list[dict] = []
    invalidated: list[str] = []
    unregistered = threading.Event()

    def serve(peer):
        peers.append(peer)
        return None if unregistered.is_set() else msgpack.dumps({"l0": [b"m"]})

    path = make_zmq_path("tcp", "127.0.0.1", _start_listener(serve, invalidated.append))
    info = {"engine_id": "peer", "host": "127.0.0.1", "port": 1234}
    with zmq_ctx(zmq.DEALER, path) as sock:
        sock.send_multipart((MoRIIOConstants.GET_META_MSG, msgpack.dumps(info)))
        agent = sock.recv_multipart()[1]
        regions = sock.recv_multipart()[1]
    assert msgspec.msgpack.decode(agent, type=MoRIIOAgentMetadata).engine_id == "owner"
    assert msgpack.loads(regions) == {"l0": [b"m"]}
    assert peers == [info]

    unregistered.set()
    with zmq_ctx(zmq.DEALER, path) as sock:
        sock.send_multipart((MoRIIOConstants.GET_META_MSG, msgpack.dumps(info)))
        assert sock.recv_multipart()[1] == MoRIIOConstants.KV_UNAVAILABLE

    with zmq_ctx(zmq.DEALER, path) as sock:
        sock.send_multipart((MoRIIOConstants.INVALIDATE_MSG, b"owner"))
        assert sock.recv_multipart()[1] == MoRIIOConstants.INVALIDATE_ACK
    assert invalidated == ["owner"]


def test_repeated_handshake_keeps_sessions_unless_the_regions_changed():
    regions = {"l0": [b"m"]}
    port = _start_listener(lambda peer: msgpack.dumps(regions), lambda e: None)
    worker = SimpleNamespace(
        _remote_epoch=0,
        _invalidated_at={},
        _is_peer_key=W._is_peer_key,
        _handshake_lock=threading.RLock(),
        engine_id="me",
        local_ip="127.0.0.1",
        side_channel_port=0,
        moriio_config=SimpleNamespace(transfer_timeout=5.0),
        moriio_wrapper=MagicMock(register_remote_engine=lambda agent: "agent"),
        local_kv_cache_metadata=[],
        remote_kv_cache_metadata=[],
        layer_name_to_remote_kv_cache_metadata={},
        remote_moriio_metadata={},
        built_write_session={},
        _handshake_args={},
    )
    worker._dropped_since = partial(W._dropped_since, worker)

    def handshake():
        W._moriio_handshake(worker, "127.0.0.1", port, 1, "P_dp0", 0, 0)

    handshake()
    worker.built_write_session["P_dp0"] = ["session"]
    # The producer starts one per request until the first one completes.
    handshake()
    assert worker.built_write_session == {"P_dp0": ["session"]}

    regions = {"l0": [b"re-registered"]}
    handshake()
    assert worker.built_write_session == {}
    assert worker.layer_name_to_remote_kv_cache_metadata["P_dp0"] == regions


def _refuse_invalidation(engine_id):
    raise TransferError("1 transfer still pending")


def test_listener_nacks_an_invalidation_it_failed_to_apply():
    path = make_zmq_path(
        "tcp", "127.0.0.1", _start_listener(None, _refuse_invalidation)
    )
    with zmq_ctx(zmq.DEALER, path) as sock:
        for _ in range(2):  # The listener keeps serving.
            sock.send_multipart((MoRIIOConstants.INVALIDATE_MSG, b"owner"))
            assert sock.poll(5000)
            assert sock.recv_multipart()[1] == MoRIIOConstants.INVALIDATE_NACK


def test_peers_that_do_not_confirm_fail_and_cost_one_timeout_in_total():
    invalidated: list[str] = []
    live_port = _start_listener(lambda peer: None, invalidated.append)
    refusing_port = _start_listener(lambda peer: None, _refuse_invalidation)
    timeout = 1.0
    worker = SimpleNamespace(
        engine_id="owner",
        zmq_context=zmq.Context(),
        moriio_config=SimpleNamespace(transfer_timeout=timeout),
    )
    peers = {("127.0.0.1", get_open_port()): f"dead{i}" for i in range(3)}
    peers[("127.0.0.1", live_port)] = "live"
    peers[("127.0.0.1", refusing_port)] = "refusing"

    start = time.monotonic()
    failed = W._invalidate_peers(worker, peers)
    elapsed = time.monotonic() - start

    assert invalidated == ["owner"]
    assert sorted(f.split(" at ")[0] for f in failed) == [
        "dead0",
        "dead1",
        "dead2",
        "refusing",
    ]
    assert sum("(no answer)" in f for f in failed) == 3
    # Peers get twice transfer_timeout: they drain their transfers first.
    assert 2 * timeout <= elapsed < 3 * timeout
    worker.zmq_context.destroy(linger=0)
