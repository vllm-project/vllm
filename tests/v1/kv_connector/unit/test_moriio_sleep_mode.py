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
from collections import OrderedDict, defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import msgpack
import msgspec
import pytest
import zmq

from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    HandshakeError,
    MoRIIOAgentMetadata,
    MoRIIOConstants,
    MoRIIOError,
    zmq_ctx,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine import (
    MoRIIOWrapper,
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
    worker = SimpleNamespace(
        moriio_wrapper=wrapper,
        layer_name_to_local_kv_cache_metadata=region_metas,
        _region_tensors=region_tensors,
        _kv_released=False,
        _peer_lock=threading.Lock(),
        _handshake_lock=threading.RLock(),
        _peer_listeners={},
        built_write_session=defaultdict(list, {"peer_dp0": ["session"]}),
        _invalidate_peers=lambda peers: calls.append(("invalidate", peers)),
        _drain_transfers=lambda: calls.append(("drain", None)),
    )
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


def test_drop_remote_engine_forgets_only_that_peers_regions_then_drains():
    drained = []
    worker = SimpleNamespace(
        _handshake_lock=threading.RLock(),
        _remote_epoch=0,
        _dropped_peers=[],
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
    assert worker._eager_handshaked_engines == {"B_dp0"}
    assert drained == [True]
    # Handshakes that started before the drop must not store what they received
    # from that peer; handshakes with other peers are unaffected.
    assert W._dropped_since(worker, 0, "A_dp1")
    assert not W._dropped_since(worker, 0, "B_dp0")
    assert not W._dropped_since(worker, 1, "A_dp1")


def _session_worker(handshake):
    return SimpleNamespace(
        _kv_released=False,
        layer_name_to_remote_kv_cache_metadata={},
        layer_name_to_local_kv_cache_metadata={"l0": [b"local"]},
        remote_moriio_metadata={},
        built_write_session=defaultdict(list),
        _handshake_args={"A_dp0": ("h", 1, 1, "A_dp0", 0, None)},
        _moriio_handshake=handshake,
        moriio_wrapper=MagicMock(
            get_unpack_memory_metadata=lambda m: m,
            build_session=lambda local, remote: (local, remote),
        ),
    )


def test_session_lookup_refetches_regions_of_a_peer_that_re_registered():
    def handshake(*args):
        worker.layer_name_to_remote_kv_cache_metadata["A_dp0"] = {"l0": [b"new"]}
        worker.remote_moriio_metadata["A_dp0"] = "meta"
        return {"agent"}

    worker = _session_worker(MagicMock(side_effect=handshake))

    sessions, meta = W._get_built_session_locked(worker, "A_dp0")

    worker._moriio_handshake.assert_called_once_with("h", 1, 1, "A_dp0", 0, None)
    assert sessions == [(b"local", b"new")]
    assert meta == "meta"


def test_session_lookup_fails_cleanly_while_the_peer_is_asleep():
    worker = _session_worker(MagicMock(side_effect=HandshakeError("asleep")))
    with pytest.raises(MoRIIOError, match="cannot refresh"):
        W._get_built_session_locked(worker, "A_dp0")


def test_write_load_fails_after_recv_abort_timeout():
    worker = SimpleNamespace(
        moriio_config=SimpleNamespace(recv_abort_timeout=10.0),
        _awaiting_write={"tx-late": "late", "tx-new": "new", "tx-done": "done"},
        _write_wait_since={
            "tx-late": time.monotonic() - 11,
            "tx-new": time.monotonic(),
            "tx-done": time.monotonic() - 11,
        },
        _timed_out_writes=OrderedDict(),
    )

    assert W._expire_awaited_writes(worker, done_recving={"done"}) == {"late"}
    assert list(worker._timed_out_writes) == ["tx-late"]
    assert set(worker._write_wait_since) == {"tx-new", "tx-done"}


class _Status:
    def __init__(self, state: str = "pending"):
        self.state = state

    def Succeeded(self) -> bool:
        return self.state == "ok"

    def Failed(self) -> bool:
        return self.state == "failed"


def test_drain_waits_past_failures_and_forgets_transfers_that_never_finish():
    failed, finishing, stuck = _Status("failed"), _Status(), _Status()
    wrapper = SimpleNamespace(lock=threading.Lock(), transfer_status=[])
    for status in (failed, finishing, stuck):
        MoRIIOWrapper._track(wrapper, status)
    threading.Timer(0.2, lambda: setattr(finishing, "state", "ok")).start()

    start = time.monotonic()
    assert MoRIIOWrapper.drain_transfers(wrapper, timeout=0.5) == 1
    assert time.monotonic() - start >= 0.5
    assert finishing.Succeeded()
    # The stuck transfer does not make every later drain wait out the timeout.
    assert wrapper.transfer_status == []
    assert MoRIIOWrapper.drain_transfers(wrapper, timeout=5) == 0


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


def test_unresponsive_peers_cost_one_timeout_in_total():
    invalidated: list[str] = []
    live_port = _start_listener(lambda peer: None, invalidated.append)
    timeout = 1.0
    worker = SimpleNamespace(
        engine_id="owner",
        zmq_context=zmq.Context(),
        moriio_config=SimpleNamespace(transfer_timeout=timeout),
    )
    peers = {("127.0.0.1", get_open_port()): f"dead{i}" for i in range(3)}
    peers[("127.0.0.1", live_port)] = "live"

    start = time.monotonic()
    W._invalidate_peers(worker, peers)
    elapsed = time.monotonic() - start

    assert invalidated == ["owner"]
    # Peers get twice transfer_timeout: they drain their transfers first.
    assert 2 * timeout <= elapsed < 3 * timeout
    worker.zmq_context.destroy(linger=0)
