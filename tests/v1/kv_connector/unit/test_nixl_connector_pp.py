# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for pipeline-parallel producers in the NIXL pull connector."""

import threading
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import msgspec
import pytest
import zmq

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_scheduler import (
    NixlBaseConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    GET_META_MSG,
    NixlConnectorMetadata,
    NixlHandshakePayload,
    RemoteMeta,
    ReqMeta,
    compute_nixl_compatibility_hash,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_worker import (
    NixlPullConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import TPMapping

from .utils import make_kv_cache_config

REMOTE_ENGINE_ID = "remote-engine"


def _fake_vllm_config(pipeline_parallel_size: int) -> Any:
    """Minimal stand-in carrying only what the compatibility hash reads."""
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
        speculative_config=None,
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pipeline_parallel_size,
            tensor_parallel_size=1,
        ),
    )


@pytest.mark.cpu_test
def test_compatibility_hash_ignores_pipeline_parallel_size():
    """A PP-sharded producer must still handshake with a PP=1 consumer.

    The hash gates whether two engines may transfer at all, so it must not
    change with pipeline_parallel_size.
    """
    assert compute_nixl_compatibility_hash(
        _fake_vllm_config(1), "FLASH_ATTN"
    ) == compute_nixl_compatibility_hash(_fake_vllm_config(4), "FLASH_ATTN")


@pytest.mark.cpu_test
def test_req_meta_reads_pp_size_and_defaults_to_one():
    metadata = NixlConnectorMetadata()
    params = {
        "remote_block_ids": ([0],),
        "remote_engine_id": REMOTE_ENGINE_ID,
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


def _make_read_worker(pp_size: int) -> NixlPullConnectorWorker:
    """Worker with only the state `_read_blocks_for_req` reaches."""
    worker = object.__new__(NixlPullConnectorWorker)
    worker._physical_blocks_per_logical_kv_block = 1
    worker._engine_last_active = {}
    worker._bidirectional_kv_xfer_enabled = False
    worker.kv_cache_config = make_kv_cache_config(block_size=16)
    worker.use_mla = False
    worker.block_size = 16
    worker._recving_metadata = {}

    worker.transfer_topo = MagicMock()
    worker.transfer_topo.tp_ratio.return_value = 1
    remote_info = MagicMock()
    remote_info.remote_physical_blocks_per_logical = 1
    remote_info.remote_block_size = 16
    remote_info.remote_tp_size = 1
    worker.transfer_topo.get_engine_info.return_value = remote_info

    plan = MagicMock(spec=TPMapping)
    plan.all_source_ranks = (0,)
    plan.source_ranks_per_group = ((0,),)
    worker.tp_mappings = {
        (REMOTE_ENGINE_ID, pp_rank): plan for pp_rank in range(pp_size)
    }
    worker.src_xfer_handles_by_block_size = {16: 1}
    worker.src_xfer_handles_by_remote = {
        (REMOTE_ENGINE_ID, pp_rank, 16): 100 + pp_rank for pp_rank in range(pp_size)
    }
    worker.dst_xfer_side_handles = {
        REMOTE_ENGINE_ID: {(pp_rank, 0): 200 + pp_rank for pp_rank in range(pp_size)}
    }
    worker._remote_agents = {
        REMOTE_ENGINE_ID: {
            (pp_rank, 0): f"agent-{pp_rank}" for pp_rank in range(pp_size)
        }
    }
    worker._read_blocks = MagicMock()
    return worker


def _req_meta(pp_size: int) -> ReqMeta:
    return ReqMeta(
        local_block_ids=([0, 1],),
        local_physical_block_ids=([0, 1],),
        tp_size=1,
        pp_size=pp_size,
        remote=RemoteMeta(
            block_ids=([0, 1],),
            host="localhost",
            port=1234,
            engine_id=REMOTE_ENGINE_ID,
            request_id="prefill-req",
        ),
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize("pp_size", [1, 2, 4])
def test_read_blocks_for_req_issues_one_read_per_producer_stage(pp_size):
    """Each producer stage owns a distinct layer slice, so a consumer must
    issue one read per stage to collect a request's whole KV."""
    worker = _make_read_worker(pp_size)

    worker._read_blocks_for_req("decode-req", _req_meta(pp_size))

    assert worker._read_blocks.call_count == pp_size
    seen_stages = [
        call.kwargs["remote_pp_rank"] for call in worker._read_blocks.call_args_list
    ]
    assert seen_stages == list(range(pp_size))


@pytest.mark.cpu_test
def test_read_blocks_for_req_targets_the_handles_of_each_stage():
    """Handles are per stage: reading stage N must use stage N's local and
    remote dlist, otherwise descriptors land at another stage's offsets."""
    worker = _make_read_worker(2)

    worker._read_blocks_for_req("decode-req", _req_meta(2))

    calls = worker._read_blocks.call_args_list
    assert [c.kwargs["local_xfer_side_handle"] for c in calls] == [100, 101]
    assert [c.kwargs["remote_xfer_side_handle"] for c in calls] == [200, 201]


@pytest.mark.cpu_test
def test_read_blocks_for_req_pp1_uses_whole_engine_handles():
    """PP=1 must keep taking the stock whole-region handle, not a shard one."""
    worker = _make_read_worker(1)
    worker.src_xfer_handles_by_remote = {}

    worker._read_blocks_for_req("decode-req", _req_meta(1))

    call = worker._read_blocks.call_args_list[0]
    assert call.kwargs["local_xfer_side_handle"] == 1
    assert call.kwargs["remote_pp_size"] == 1


@pytest.mark.cpu_test
def test_read_blocks_for_req_keeps_remote_block_ids_logical_under_pp():
    """Every stage expands the logical ids with its own geometry, so the
    shared metadata must not be rewritten in place (PP=1 still is, since
    callers downstream rely on the expansion)."""
    worker = _make_read_worker(2)
    meta = _req_meta(2)

    worker._read_blocks_for_req("decode-req", meta)

    assert meta.remote.block_ids == ([0, 1],)


@pytest.mark.cpu_test
def test_full_prefix_hit_under_pp_drops_pending_metadata():
    """No read is posted on a full prefix hit, so nothing will ever report the
    request complete; the PP path must clear it or the entry leaks."""
    worker = _make_read_worker(2)
    meta = _req_meta(2)
    meta.local_physical_block_ids = ()
    worker._recving_metadata["decode-req"] = meta

    worker._read_blocks_for_req("decode-req", meta)

    assert "decode-req" not in worker._recving_metadata


@pytest.mark.cpu_test
def test_get_block_descs_ids_for_shard_is_relative_to_the_shard():
    """A stage registers only its own layers, so descriptor ids run over that
    shard's regions starting at 0 rather than over all local regions."""
    worker = object.__new__(NixlPullConnectorWorker)
    worker._shard_region_group_ids = {(REMOTE_ENGINE_ID, 1): (0, 0)}

    desc_ids = worker._get_block_descs_ids_for_shard(REMOTE_ENGINE_ID, 1, 4, ([0, 1],))

    # 2 regions x blocks [0, 1] over num_blocks=4: region 0 -> 0,1; region 1 -> 4,5.
    assert desc_ids.tolist() == [0, 1, 4, 5]


@pytest.mark.cpu_test
def test_handshake_complete_requires_every_stage_registered():
    """`_remote_agents` alone cannot distinguish a partially-registered
    pipelined peer from a fully-registered PP=1 one."""
    worker = object.__new__(NixlPullConnectorWorker)
    worker._remote_agents = {REMOTE_ENGINE_ID: {(0, 0): "agent-0"}}
    worker._remote_pp_size = {}

    assert worker._handshake_complete(REMOTE_ENGINE_ID, 1)
    assert not worker._handshake_complete(REMOTE_ENGINE_ID, 2)

    worker._remote_pp_size[REMOTE_ENGINE_ID] = 2

    assert worker._handshake_complete(REMOTE_ENGINE_ID, 2)


@pytest.mark.cpu_test
@pytest.mark.parametrize("pp_size", [1, 4])
def test_background_handshake_forwards_the_remote_pp_size(pp_size):
    worker = object.__new__(NixlPullConnectorWorker)
    worker._handshake_lock = threading.RLock()
    worker._handshake_futures = {}
    worker._remote_agents = {}
    worker._remote_pp_size = {}
    worker._engine_last_active = {}
    worker._engine_ttl = 0
    worker._ready_requests = MagicMock()
    future = MagicMock()
    worker._handshake_initiation_executor = MagicMock()
    worker._handshake_initiation_executor.submit.return_value = future

    worker._background_nixl_handshake("req", REMOTE_ENGINE_ID, _req_meta(pp_size))

    # Drop the stub so __del__ takes shutdown()'s partially-initialized path.
    executor = worker._handshake_initiation_executor
    del worker._handshake_initiation_executor

    executor.submit.assert_called_once_with(
        worker._nixl_handshake,
        "localhost",
        1234,
        1,
        REMOTE_ENGINE_ID,
        pp_size,
        False,
    )


class _InlineThread:
    """Runs the listener on the calling thread so the test stays deterministic."""

    def __init__(
        self, *, target: Callable[..., Any], args: tuple[Any, ...], **_: Any
    ) -> None:
        self._target = target
        self._args = args

    def start(self) -> None:
        self._target(*self._args)


class _FakeHandshakeSocket:
    def __init__(self, request_msg: bytes, stop_event: threading.Event) -> None:
        self._request_msg = request_msg
        self._stop_event = stop_event
        self._served = False
        self.sent_multipart: list[tuple[bytes, ...]] = []

    def setsockopt(self, *_: Any) -> None:
        return None

    def recv_multipart(self) -> tuple[bytes, bytes, bytes]:
        if not self._served:
            self._served = True
            return (b"identity", b"", self._request_msg)
        self._stop_event.set()
        raise zmq.Again()

    def send_multipart(self, parts: tuple[bytes, ...]) -> None:
        self.sent_multipart.append(parts)


class _FakeZmqContext:
    def __init__(self, sock: _FakeHandshakeSocket) -> None:
        self._sock = sock

    def __enter__(self) -> _FakeHandshakeSocket:
        return self._sock

    def __exit__(self, *_: Any) -> None:
        return None


@pytest.mark.cpu_test
def test_handshake_listener_serves_a_requested_pp_stage():
    """A consumer asks for one (pp_rank, tp_rank) at a time, so a PP-sharded
    producer must serve the payload keyed by the requested stage."""
    stop_event = threading.Event()
    payloads = {
        (0, 0): NixlHandshakePayload(
            compatibility_hash="h", agent_metadata_bytes=b"s0"
        ),
        (1, 0): NixlHandshakePayload(
            compatibility_hash="h", agent_metadata_bytes=b"s1"
        ),
    }
    request = msgspec.msgpack.encode((GET_META_MSG, 1, 0))
    sock = _FakeHandshakeSocket(request, stop_event)

    scheduler = object.__new__(NixlBaseConnectorScheduler)
    scheduler._nixl_handshake_listener_t = None
    scheduler._stop_event = stop_event
    scheduler.side_channel_host = "localhost"
    scheduler.side_channel_port = 1234

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
        scheduler.set_xfer_handshake_metadata(payloads)

    assert len(sock.sent_multipart) == 1
    identity, delimiter, encoded_payload, _ts = sock.sent_multipart[0]
    assert identity == b"identity"
    assert delimiter == b""
    decoded = msgspec.msgpack.decode(encoded_payload, type=NixlHandshakePayload)
    assert decoded == payloads[(1, 0)]
