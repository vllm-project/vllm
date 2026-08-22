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
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    ShardDescLayout,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    GET_META_MSG,
    NIXL_CONNECTOR_VERSION,
    NixlAgentMetadata,
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
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

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


def _make_desc_shard_worker(
    region_member_groups: tuple[tuple[int, ...], ...],
    group_spec_types: tuple[type, ...],
    ssm_regions_per_layer: int = 0,
) -> NixlPullConnectorWorker:
    """Worker with only the state `_get_block_descs_ids_for_shard` reaches."""
    worker = object.__new__(NixlPullConnectorWorker)
    worker._group_spec_types = group_spec_types
    worker._shard_desc_layouts = {
        (REMOTE_ENGINE_ID, 1): ShardDescLayout(
            region_member_groups=region_member_groups,
            ssm_regions_per_layer=ssm_regions_per_layer,
        )
    }
    return worker


@pytest.mark.cpu_test
def test_get_block_descs_ids_for_shard_is_relative_to_the_shard():
    """A stage registers only its own layers, so descriptor ids run over that
    shard's regions starting at 0 rather than over all local regions."""
    worker = _make_desc_shard_worker(((0,), (0,)), (FullAttentionSpec,))

    desc_ids = worker._get_block_descs_ids_for_shard(REMOTE_ENGINE_ID, 1, 4, ([0, 1],))

    # 2 regions x blocks [0, 1] over num_blocks=4: region 0 -> 0,1; region 1 -> 4,5.
    assert desc_ids.tolist() == [0, 1, 4, 5]


@pytest.mark.cpu_test
def test_get_block_descs_ids_for_shard_covers_every_pooled_member():
    """HMA pools a layer per KV group into one region, and each group indexes
    that region with its own block ids. Emitting only the representative
    group's ids leaves the other members' state never transferred."""
    worker = _make_desc_shard_worker(
        ((0, 1), (0, 2)),
        (FullAttentionSpec, FullAttentionSpec, FullAttentionSpec),
    )

    desc_ids = worker._get_block_descs_ids_for_shard(
        REMOTE_ENGINE_ID, 1, 10, ([1, 2], [7], [4, 5])
    )

    # region 0 holds group 0 blocks [1, 2] and group 1 block [7];
    # region 1 (offset 10) holds group 0 blocks [1, 2] and group 2 blocks [4, 5].
    assert desc_ids.tolist() == [1, 2, 7, 11, 12, 14, 15]


@pytest.mark.cpu_test
def test_get_block_descs_ids_for_shard_emits_the_mamba_section():
    """A hybrid shard's dlist is [fa | ssm], with the SSM half holding one
    sub-region per conv projection plus the temporal state. Without the SSM
    section a Mamba member's state is never read and decode goes stale."""
    worker = _make_desc_shard_worker(
        ((0, 1), (0, 1)),
        (FullAttentionSpec, MambaSpec),
        ssm_regions_per_layer=4,
    )

    desc_ids = worker._get_block_descs_ids_for_shard(
        REMOTE_ENGINE_ID, 1, 10, ([1, 2], [3]), 2
    )

    # FA: 2 regions x num_blocks=10 -> ids 1,2 and 11,12 (num_fa_descs=20).
    # SSM: logical_blocks = 10 // 2 = 5, 4 sub-regions per region, so region 0
    # occupies 20,25,30,35 and region 1 40,45,50,55, each offset by block 3.
    assert desc_ids.tolist() == [1, 2, 11, 12, 23, 28, 33, 38, 43, 48, 53, 58]


@pytest.mark.cpu_test
def test_get_block_descs_ids_for_shard_skips_empty_groups():
    """A group with no blocks for this request contributes no descriptors."""
    worker = _make_desc_shard_worker(((0, 1),), (FullAttentionSpec, FullAttentionSpec))

    desc_ids = worker._get_block_descs_ids_for_shard(
        REMOTE_ENGINE_ID, 1, 10, ([1, 2], [])
    )

    assert desc_ids.tolist() == [1, 2]


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


def _agent_meta(**overrides: Any) -> NixlAgentMetadata:
    """Handshake metadata of a single-region producer shard."""
    kwargs: dict[str, Any] = dict(
        engine_id=REMOTE_ENGINE_ID,
        agent_metadata=b"",
        kv_caches_base_addr=[0x1000],
        device_id=0,
        num_blocks=4,
        block_lens=[128],
        block_strides=[128],
        kv_cache_layout="LBHNC",
        block_size=16,
        ssm_sizes=(0, 0),
        attn_backend_name="FLASH_ATTN",
        physical_blocks_per_logical_kv_block=1,
        registered_layer_names=["layer0"],
    )
    kwargs.update(overrides)
    return NixlAgentMetadata(**kwargs)


@pytest.mark.cpu_test
def test_nixl_agent_metadata_region_members_round_trip():
    """A pooled region's non-representative members must survive the wire, or
    the consumer cannot know they need transferring."""
    meta = _agent_meta(
        registered_layer_names=["layer0", "layer2"],
        region_members=[["layer0", "mamba0"], ["layer2", "mamba1"]],
    )

    decoded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(meta), type=NixlAgentMetadata
    )

    assert NIXL_CONNECTOR_VERSION == 10
    assert decoded.region_members == [["layer0", "mamba0"], ["layer2", "mamba1"]]


@pytest.mark.cpu_test
def test_remote_region_members_defaults_to_one_layer_per_region():
    """A peer that advertises no membership has no pooling, so each region
    holds exactly its representative layer."""
    meta = _agent_meta(registered_layer_names=["layer0", "layer2"])

    members = NixlPullConnectorWorker._remote_region_members(meta)

    assert members == [["layer0"], ["layer2"]]


@pytest.mark.cpu_test
def test_remote_region_members_rejects_a_partial_advertisement():
    """Membership that does not cover every region would silently drop the
    uncovered regions' non-representative members."""
    meta = _agent_meta(
        registered_layer_names=["layer0", "layer2"],
        region_members=[["layer0", "mamba0"]],
    )

    with pytest.raises(RuntimeError, match="region_members"):
        NixlPullConnectorWorker._remote_region_members(meta)


@pytest.mark.cpu_test
def test_shard_desc_layout_maps_members_to_their_kv_groups():
    """The block ids addressing a pooled region are selected per member group,
    so the layout must carry each member's group, not just the region's."""
    worker = object.__new__(NixlPullConnectorWorker)
    worker._layer_name_to_kv_group_index = {"layer0": 0, "mamba0": 1}
    worker._has_mamba = False

    layout = worker._shard_desc_layout(
        _agent_meta(region_members=[["layer0", "mamba0"]])
    )

    assert layout.region_member_groups == ((0, 1),)
    assert layout.ssm_regions_per_layer == 0


@pytest.mark.cpu_test
def test_shard_desc_layout_rejects_an_unknown_member():
    """A member with no local KV group cannot be routed; failing the handshake
    beats emitting descriptors that address the wrong region."""
    worker = object.__new__(NixlPullConnectorWorker)
    worker._layer_name_to_kv_group_index = {"layer0": 0}
    worker._has_mamba = False

    with pytest.raises(RuntimeError, match="no matching local KV cache group"):
        worker._shard_desc_layout(_agent_meta(region_members=[["layer0", "ghost"]]))


@pytest.mark.cpu_test
def test_local_region_indices_resolve_a_pooled_member():
    """A producer stage may advertise a layer that is a non-representative
    member locally; it must still resolve to the region physically holding it."""
    worker = object.__new__(NixlPullConnectorWorker)
    worker.local_seen_layer_names = ["layer0", "layer2"]
    worker._member_to_local_region = {
        "layer0": 0,
        "mamba0": 0,
        "layer2": 1,
        "mamba1": 1,
    }

    assert worker._local_region_indices_for_layer_names(["mamba1", "layer0"]) == [1, 0]


@pytest.mark.cpu_test
def test_local_region_indices_still_reject_an_unknown_layer():
    worker = object.__new__(NixlPullConnectorWorker)
    worker.local_seen_layer_names = ["layer0"]
    worker._member_to_local_region = {"layer0": 0}

    with pytest.raises(RuntimeError, match="no matching local region"):
        worker._local_region_indices_for_layer_names(["ghost"])


@pytest.mark.cpu_test
def test_pull_declares_pp_hma_support_and_push_does_not():
    """The constructor refuses pipeline parallelism with a hybrid KV cache
    unless the transfer mode declares support. Pull resolves pooled regions by
    layer membership; push slices regions per layer, which pooling breaks."""
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
        NixlBaseConnectorWorker,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_worker import (
        NixlPushConnectorWorker,
    )

    assert NixlPullConnectorWorker._supports_pp_hma is True
    assert NixlBaseConnectorWorker._supports_pp_hma is False
    assert NixlPushConnectorWorker._supports_pp_hma is False
