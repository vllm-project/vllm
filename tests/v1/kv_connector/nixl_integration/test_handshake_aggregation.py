# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import msgspec
import pytest
import zmq

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    GET_META_MSG,
    NixlHandshakePayload,
    RemoteMeta,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.scheduler import (
    NixlConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
    NixlConnectorWorker,
    ReadSpec,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec


class _InlineThread:
    def __init__(
        self,
        *,
        target: Callable[..., Any],
        args: tuple[Any, ...],
        **_: Any,
    ) -> None:
        self._target = target
        self._args = args

    def start(self) -> None:
        self._target(*self._args)


class _FakeZmqContext:
    def __init__(self, sock: "_FakeHandshakeSocket") -> None:
        self._sock = sock

    def __enter__(self) -> "_FakeHandshakeSocket":
        return self._sock

    def __exit__(self, *args: Any) -> None:
        return None


class _FakeHandshakeSocket:
    def __init__(
        self,
        request_msg: bytes,
        *,
        stop_event: threading.Event | None = None,
    ) -> None:
        self._request_msg = request_msg
        self._stop_event = stop_event
        self._recv_count = 0
        self.sent_multipart: list[tuple[bytes, bytes, bytes]] = []

    def setsockopt(self, *_: Any) -> None:
        return None

    def recv_multipart(self) -> tuple[bytes, bytes, bytes]:
        if self._recv_count == 0:
            self._recv_count += 1
            return (b"identity", b"", self._request_msg)
        if self._stop_event is not None:
            self._stop_event.set()
        raise zmq.Again()

    def send_multipart(self, parts: tuple[bytes, bytes, bytes]) -> None:
        self.sent_multipart.append(parts)


def test_engine_merge_preserves_pp_and_tp_keys():
    metadata_a = object()
    metadata_b = object()
    metadata_c = object()
    worker_dicts = [
        {(0, 0): metadata_a},
        {(1, 0): metadata_b},
        {(0, 1): metadata_c},
    ]

    content: dict[tuple[int, int], object] = {}
    for worker_dict in worker_dicts:
        content.update(worker_dict)

    assert content == {
        (0, 0): metadata_a,
        (1, 0): metadata_b,
        (0, 1): metadata_c,
    }


def test_scheduler_listener_serves_three_tuple_key():
    scheduler = NixlConnectorScheduler.__new__(NixlConnectorScheduler)
    scheduler._nixl_handshake_listener_t = None
    scheduler._stop_event = threading.Event()
    scheduler.side_channel_host = "localhost"
    scheduler.side_channel_port = 1234

    payload = NixlHandshakePayload(
        compatibility_hash="hash",
        agent_metadata_bytes=b"agent",
    )
    request = msgspec.msgpack.encode((GET_META_MSG, 1, 0))
    sock = _FakeHandshakeSocket(request, stop_event=scheduler._stop_event)

    with (
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.nixl.scheduler.zmq_ctx",
            return_value=_FakeZmqContext(sock),
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.nixl.scheduler."
            "threading.Thread",
            _InlineThread,
        ),
    ):
        scheduler.set_xfer_handshake_metadata({(1, 0): payload})

    assert len(sock.sent_multipart) == 1
    identity, delimiter, encoded_payload = sock.sent_multipart[0]
    assert identity == b"identity"
    assert delimiter == b""
    decoded_payload = msgspec.msgpack.decode(encoded_payload, type=NixlHandshakePayload)
    assert decoded_payload == payload


def test_ensure_handshake_treats_partial_pp_state_as_inflight():
    worker = NixlConnectorWorker.__new__(NixlConnectorWorker)
    future = MagicMock()
    remote_engine_id = "remote-engine"
    worker._handshake_lock = threading.RLock()
    worker._handshake_futures = {remote_engine_id: future}
    worker._remote_agents = {remote_engine_id: {(0, 0): "agent-0-0"}}
    worker._pp_layer_map = {}

    assert worker._ensure_handshake(remote_engine_id, "localhost", 1234, 1, 2) is future


def test_handshake_complete_requires_pp_layer_map():
    worker = NixlConnectorWorker.__new__(NixlConnectorWorker)
    remote_engine_id = "remote-engine"
    worker._handshake_futures = {}
    worker._remote_agents = {remote_engine_id: {(0, 0): "agent-0-0"}}
    worker._pp_layer_map = {}

    assert not worker._handshake_complete(remote_engine_id, 2)

    worker._pp_layer_map[remote_engine_id] = SimpleNamespace(pp_size=2)

    assert worker._handshake_complete(remote_engine_id, 2)


@pytest.mark.parametrize("pp_size", [1, 4])
def test_background_nixl_handshake_submits_remote_pp_size(pp_size: int):
    worker = NixlConnectorWorker.__new__(NixlConnectorWorker)
    worker._handshake_futures = {}
    worker._handshake_initiation_executor = MagicMock()
    future = MagicMock()
    worker._handshake_initiation_executor.submit.return_value = future
    worker._handshake_lock = threading.Lock()
    worker._remote_agents = {}
    worker._ready_requests = MagicMock()
    worker._log_failure = MagicMock()
    worker._recving_transfers = {}
    worker.src_xfer_handles_by_remote = {}
    worker.src_xfer_handles_by_shard_tp_ratio = {}
    worker.dst_xfer_side_handles = {}
    worker._registered_descs = []

    remote_engine_id = "remote-engine"
    meta = ReqMeta(
        local_block_ids=([0],),
        local_physical_block_ids=([0],),
        tp_size=2,
        pp_size=pp_size,
        remote=RemoteMeta(
            block_ids=([1],),
            host="localhost",
            port=1234,
            engine_id=remote_engine_id,
            request_id="remote-request",
        ),
    )

    worker._background_nixl_handshake("request", remote_engine_id, meta)

    worker._handshake_initiation_executor.submit.assert_called_once_with(
        worker._nixl_handshake,
        "localhost",
        1234,
        2,
        pp_size,
        remote_engine_id,
    )
    assert future.add_done_callback.call_count == 2


def test_hma_pp_read_blocks_maps_each_kv_group() -> None:
    """HMA reads map block-size ratios independently per KV group."""
    import numpy as np

    worker = NixlConnectorWorker.__new__(NixlConnectorWorker)
    worker._is_hma_required = True
    worker.world_size = 1
    worker.block_size = 16
    worker._remote_agents = {"remote-engine": {(0, 0): "agent-0-0"}}
    worker._group_spec_types = (FullAttentionSpec, FullAttentionSpec)
    worker.kv_cache_config = SimpleNamespace(kv_cache_groups=[object(), object()])
    worker._recving_transfers = {"req": []}
    worker._log_failure = MagicMock()
    worker._handle_failed_transfer = MagicMock()
    worker.xfer_stats = MagicMock()
    worker.nixl_wrapper = MagicMock()
    worker.nixl_wrapper.make_prepped_xfer.return_value = 99

    transfer_topo = MagicMock()
    transfer_topo.get_engine_info.return_value = SimpleNamespace(
        remote_block_size=8,
        remote_physical_blocks_per_logical=1,
    )
    transfer_topo.block_size_ratio.return_value = 2
    worker.transfer_topo = transfer_topo
    worker.get_mapped_blocks = NixlConnectorWorker.get_mapped_blocks.__get__(worker)
    worker._apply_prefix_caching = MagicMock(
        side_effect=lambda local, remote, _: (
            local,
            remote,
        )
    )
    worker._get_block_descs_ids_for_shard = MagicMock(
        side_effect=[np.asarray([10, 11, 12, 20, 21]), np.asarray([0, 1, 2, 6, 7])]
    )

    spec = ReadSpec(
        remote_rank=0,
        local_block_ids=[[0, 1], [3]],
        remote_block_ids=[[5, 6, 7], [8, 9]],
    )
    worker._read_blocks(
        read_spec=spec,
        request_id="req",
        dst_engine_id="remote-engine",
        remote_request_id="rreq",
        remote_pp_rank=0,
        local_xfer_side_handle=0,
        remote_xfer_side_handle=0,
    )

    worker._apply_prefix_caching.assert_called_once_with(
        [[0, 1, 2], [6, 7]],
        [[5, 6, 7], [8, 9]],
        1,
    )
    assert worker._recving_transfers["req"] == [99]
