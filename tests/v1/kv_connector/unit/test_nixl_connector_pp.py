# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for pipeline-parallel PD disaggregation in the NIXL connector."""

import threading
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import msgspec
import pytest
import torch
import zmq

from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlAgentMetadata,
    NixlConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    GET_META_MSG,
    NixlHandshakePayload,
    RemoteMeta,
    ReqMeta,
    compute_nixl_compatibility_hash,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.scheduler import (
    NixlConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    ReadSpec,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
    NixlConnectorWorker,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)

from .test_nixl_connector import FakeNixlConnectorWorker, FakeNixlWrapper
from .utils import create_vllm_config

LOCAL_ENGINE_ID = "local-engine"
REMOTE_ENGINE_ID = FakeNixlConnectorWorker.REMOTE_ENGINE_ID
TOTAL_LAYERS = 32
BLOCK_LEN = 4096 * 16  # slot_size_bytes * block_size
NUM_BLOCKS = 4


def _fake_vllm_config(pipeline_parallel_size: int = 1) -> SimpleNamespace:
    model_config = SimpleNamespace(
        model="fake-model",
        dtype="float16",
        get_total_num_kv_heads=lambda: 8,
        get_head_size=lambda: 16,
        get_total_num_hidden_layers=lambda: 32,
    )
    return SimpleNamespace(
        model_config=model_config,
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=16),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pipeline_parallel_size,
            tensor_parallel_size=1,
        ),
    )


def test_compatibility_hash_ignores_pipeline_parallel_size():
    assert compute_nixl_compatibility_hash(
        _fake_vllm_config(pipeline_parallel_size=1), "FLASH_ATTN", False
    ) == compute_nixl_compatibility_hash(
        _fake_vllm_config(pipeline_parallel_size=4), "FLASH_ATTN", False
    )


def test_req_meta_reads_pp_size_and_defaults_to_one():
    metadata = NixlConnectorMetadata()
    params = {
        "remote_block_ids": ([0],),
        "remote_engine_id": "engine",
        "remote_request_id": "remote-req",
        "remote_host": "localhost",
        "remote_port": 1234,
        "tp_size": 2,
        "pp_size": 4,
    }

    metadata.add_new_req_to_recv("req", ([0],), params)
    assert metadata.reqs_to_recv["req"].pp_size == 4

    params.pop("pp_size")
    metadata.add_new_req_to_recv("req-default", ([0],), params)
    assert metadata.reqs_to_recv["req-default"].pp_size == 1


def _pp_kv_cache_config() -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                [f"model.layers.{i}.self_attn" for i in range(TOTAL_LAYERS)],
                FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=4,
                    head_size=16,
                    dtype=torch.float16,
                ),
            )
        ],
    )


def _make_pp_worker() -> NixlConnectorWorker:
    """Real worker via FakeNixlConnectorWorker; seed only the state that
    register_kv_caches() would normally produce."""
    vllm_config = create_vllm_config()
    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker.NixlWrapper",
        FakeNixlWrapper,
    ):
        worker = FakeNixlConnectorWorker(
            vllm_config,
            LOCAL_ENGINE_ID,
            hand_shake_latency=0,
            kv_cache_config=_pp_kv_cache_config(),
        )
    worker.block_len_per_layer = [BLOCK_LEN] * TOTAL_LAYERS
    worker.local_seen_layer_names = [
        f"model.layers.{i}.self_attn" for i in range(TOTAL_LAYERS)
    ]
    worker.local_kv_caches_base_addr = [
        100_000 + i * 10_000 for i in range(TOTAL_LAYERS)
    ]
    return worker


def _meta(
    worker: NixlConnectorWorker,
    pp_rank: int,
    start_layer: int,
    end_layer: int,
    *,
    pp_size: int,
) -> NixlAgentMetadata:
    layer_names = [f"model.layers.{i}.self_attn" for i in range(start_layer, end_layer)]
    return NixlAgentMetadata(
        engine_id=REMOTE_ENGINE_ID,
        agent_metadata=FakeNixlWrapper.AGENT_METADATA,
        kv_caches_base_addr=[
            200_000 + i * 10_000 for i in range(start_layer, end_layer)
        ],
        device_id=pp_rank,
        num_blocks=NUM_BLOCKS,
        block_lens=[BLOCK_LEN] * len(layer_names),
        kv_cache_layout="HND",
        block_size=16,
        ssm_sizes=(0, 0),
        attn_backend_name=worker.backend_name,
        physical_blocks_per_logical_kv_block=1,
        pp_rank=pp_rank,
        pp_size=pp_size,
        registered_layer_names=layer_names,
    )


def _add_two_remote_shards(worker: NixlConnectorWorker) -> list[NixlAgentMetadata]:
    metas = [
        _meta(worker, 0, 0, 16, pp_size=2),
        _meta(worker, 1, 16, 32, pp_size=2),
    ]
    for meta in metas:
        worker.add_remote_agent(
            meta,
            remote_tp_rank=0,
            remote_tp_size=1,
            remote_pp_rank=meta.pp_rank,
            remote_pp_size=meta.pp_size,
        )
    # done_callback marks the handshake complete once every shard is registered.
    worker._remote_pp_size[REMOTE_ENGINE_ID] = 2
    return metas


def test_add_remote_agent_records_both_pp_shard_base_address_keys(
    default_vllm_config, dist_init
):
    worker = _make_pp_worker()

    _add_two_remote_shards(worker)

    assert set(worker.kv_caches_base_addr[REMOTE_ENGINE_ID]) == {(0, 0), (1, 0)}


def test_validate_remote_agent_handshake_accepts_synthetic_pp_shard(
    default_vllm_config, dist_init
):
    worker = _make_pp_worker()
    meta = _meta(worker, 0, 0, 16, pp_size=2)

    worker.add_remote_agent(
        meta,
        remote_tp_rank=0,
        remote_tp_size=1,
        remote_pp_rank=0,
        remote_pp_size=2,
    )
    worker._validate_remote_agent_handshake(meta, 0, 2, 1)


def test_add_remote_agent_prepares_dst_handles_for_each_pp_shard(
    default_vllm_config, dist_init
):
    worker = _make_pp_worker()

    _add_two_remote_shards(worker)

    assert set(worker.dst_xfer_side_handles[REMOTE_ENGINE_ID]) == {
        (0, 0),
        (1, 0),
    }


def test_read_blocks_for_req_appends_one_transfer_per_pp_shard_and_tp_target(
    default_vllm_config, dist_init
):
    worker = _make_pp_worker()
    _add_two_remote_shards(worker)
    req_meta = ReqMeta(
        local_block_ids=([0, 1],),
        local_physical_block_ids=([0, 1],),
        tp_size=1,
        pp_size=2,
        remote=RemoteMeta(
            block_ids=([0, 1],),
            host="localhost",
            port=1234,
            engine_id=REMOTE_ENGINE_ID,
            request_id="prefill-req",
        ),
    )

    worker._read_blocks_for_req("decode-req", req_meta)

    assert len(worker._recving_transfers["decode-req"]) == 2


def test_pp_rank_one_descriptor_ids_are_shard_local(default_vllm_config, dist_init):
    worker = _make_pp_worker()
    _add_two_remote_shards(worker)

    remote_desc_ids = worker._get_block_descs_ids_for_shard(
        REMOTE_ENGINE_ID, 1, worker.dst_num_blocks[REMOTE_ENGINE_ID], ([0],)
    )
    local_desc_ids = worker._get_block_descs_ids_for_shard(
        REMOTE_ENGINE_ID, 1, worker.num_blocks, ([0],)
    )

    assert remote_desc_ids[0] == 0
    assert local_desc_ids[0] == 0


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
            "vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler.zmq_ctx",
            return_value=_FakeZmqContext(sock),
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler."
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
    worker._remote_pp_size = {}
    worker._engine_ttl = 0.0

    assert worker._ensure_handshake(remote_engine_id, "localhost", 1234, 1, 2) is future


def test_handshake_complete_requires_remote_pp_size():
    worker = NixlConnectorWorker.__new__(NixlConnectorWorker)
    remote_engine_id = "remote-engine"
    worker._handshake_futures = {}
    worker._remote_agents = {remote_engine_id: {(0, 0): "agent-0-0"}}
    worker._remote_pp_size = {}

    assert not worker._handshake_complete(remote_engine_id, 2)

    worker._remote_pp_size[remote_engine_id] = 2

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
    worker._engine_ttl = 0.0
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


def test_hma_pp_assertion_guard_in_read_blocks() -> None:
    """HMA × PP combinations must be rejected with AssertionError.

    Per-layer-name HMA × PP routing is not yet supported, so the
    ``assert not self._is_hma_required`` checks inside ``_read_blocks`` and
    friends must fail loud: configuring NIXL with HMA enabled and a
    heterogeneous block-size remote (the path that co-occurs under
    ``pp_size > 1`` with multi-group KV caches) must raise.
    """
    import numpy as np

    worker = NixlConnectorWorker.__new__(NixlConnectorWorker)
    worker._is_hma_required = True
    worker.world_size = 1
    worker.block_size = 16
    worker._remote_agents = {"remote-engine": {(0, 0): "agent-0-0"}}

    transfer_topo = MagicMock()
    transfer_topo.get_engine_info.return_value = SimpleNamespace(
        remote_block_size=8,
        remote_physical_blocks_per_logical=1,
    )
    transfer_topo.block_size_ratio.return_value = 2
    worker.transfer_topo = transfer_topo
    worker.get_mapped_blocks = MagicMock(return_value=np.asarray([0, 1, 2, 3]))

    spec = ReadSpec(remote_rank=0, local_block_ids=[[0]], remote_block_ids=[[1]])
    with pytest.raises(AssertionError):
        worker._read_blocks(
            read_spec=spec,
            request_id="req",
            dst_engine_id="remote-engine",
            remote_request_id="rreq",
            remote_pp_rank=0,
            local_xfer_side_handle=0,
            remote_xfer_side_handle=0,
        )
