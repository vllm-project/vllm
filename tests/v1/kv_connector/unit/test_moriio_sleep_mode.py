# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MoRIIO sleep mode: a released KV cache is deregistered only after every peer
that fetched its region metadata dropped it, and is registered again in place.

A transfer against a deregistered region fails with a remote access error that
leaves the peer's queue pair unusable, so peers must forget the metadata first.
"""

import threading
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import msgpack
import msgspec
import zmq

from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    MoRIIOAgentMetadata,
    MoRIIOConstants,
    zmq_ctx,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnectorWorker,
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
        built_write_session=defaultdict(list, {"peer_dp0": ["session"]}),
        _invalidate_peers=lambda: calls.append(("invalidate", None)),
    )
    return worker, calls


def test_release_invalidates_peers_then_deregisters_each_region_once():
    shared = object()
    worker, calls = _worker(
        {"l0": [b"m"], "l1": [b"m"]}, {"l0": [shared], "l1": [shared]}
    )

    W.release_kv_caches(worker)
    W.release_kv_caches(worker)

    assert calls == [("invalidate", None), ("deregister", b"m")]
    assert worker._kv_released
    assert not worker.built_write_session


def test_restore_republishes_new_metadata_in_place():
    shared = MagicMock(data_ptr=lambda: 0x1000, nbytes=64)
    metas = {"l0": [b"m"], "l1": [b"m"]}
    l0_list = metas["l0"]
    worker, _ = _worker(metas, {"l0": [shared], "l1": [shared]})

    W.release_kv_caches(worker)
    W.restore_kv_caches(worker)

    # The listener serves this dict, so it must be updated, not replaced.
    assert worker.layer_name_to_local_kv_cache_metadata is metas
    assert metas["l0"] is l0_list
    assert metas == {"l0": [b"new0"], "l1": [b"new0"]}
    assert worker.moriio_wrapper.register_local_tensor.call_count == 1
    assert not worker._kv_released


def test_drop_remote_engine_forgets_only_that_peer():
    worker = SimpleNamespace(
        _handshake_lock=threading.RLock(),
        layer_name_to_remote_kv_cache_metadata={"A_dp0": {}, "B_dp0": {}},
        remote_moriio_metadata={"A_dp0": 1, "B_dp0": 2},
        _remote_agents={"A": {"x"}, "A_dp0": {"x"}, "AB_dp0": {"y"}},
        _handshake_futures={"A_dp0": None},
        built_write_session=defaultdict(list, {"A_dp0": [1], "B_dp0": [2]}),
        _eager_handshaked_engines={"A_dp0", "B_dp0"},
    )

    W._drop_remote_engine(worker, "A")

    assert set(worker.layer_name_to_remote_kv_cache_metadata) == {"B_dp0"}
    assert set(worker.remote_moriio_metadata) == {"B_dp0"}
    assert set(worker._remote_agents) == {"AB_dp0"}
    assert not worker._handshake_futures
    assert set(worker.built_write_session) == {"B_dp0"}
    assert worker._eager_handshaked_engines == {"B_dp0"}


def test_listener_records_peers_and_acks_invalidation():
    port = get_open_port()
    peers: list[dict] = []
    invalidated: list[str] = []
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
        args=(
            metadata,
            ready,
            port,
            0,
            0,
            {"l0": [b"m"]},
            peers.append,
            invalidated.append,
        ),
        daemon=True,
    ).start()
    assert ready.wait(10)
    path = make_zmq_path("tcp", "127.0.0.1", port)

    with zmq_ctx(zmq.DEALER, path) as sock:
        info = {"engine_id": "peer", "host": "127.0.0.1", "port": 1234}
        sock.send_multipart((MoRIIOConstants.GET_META_MSG, msgpack.dumps(info)))
        agent = sock.recv_multipart()[1]
        regions = sock.recv_multipart()[1]
    assert msgspec.msgpack.decode(agent, type=MoRIIOAgentMetadata).engine_id == "owner"
    assert msgpack.loads(regions) == {"l0": [b"m"]}
    assert peers == [info]

    with zmq_ctx(zmq.DEALER, path) as sock:
        sock.send_multipart((MoRIIOConstants.INVALIDATE_MSG, b"owner"))
        assert sock.recv_multipart()[1] == MoRIIOConstants.INVALIDATE_ACK
    assert invalidated == ["owner"]
