# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from types import SimpleNamespace

from tests.v1.kv_connector.nixl_integration.test_pp_layer_map import (
    _FakeAttentionBackend,
    _meta,
)
from vllm.distributed.kv_transfer.kv_connector.utils import TransferTopology
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlAgentMetadata,
    RemoteMeta,
    ReqMeta,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
    NixlConnectorWorker,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec

REMOTE_ENGINE_ID = "engine"
LOCAL_ENGINE_ID = "local-engine"


class _FakeNixlWrapper:
    def __init__(self) -> None:
        self._next_handle = 0

    def add_remote_agent(self, agent_metadata: bytes) -> str:
        return f"remote-agent-{len(agent_metadata)}-{self._next_handle}"

    def get_xfer_descs(self, blocks_data, memory_type: str) -> list[object]:
        return list(blocks_data)

    def prep_xfer_dlist(self, agent_name: str, descs: list[object]) -> int:
        self._next_handle += 1
        return self._next_handle

    def make_prepped_xfer(self, *args, **kwargs) -> int:
        self._next_handle += 1
        return self._next_handle

    def transfer(self, handle: int) -> None:
        pass

    def send_notif(self, agent_name: str, notif_msg: bytes) -> None:
        pass


def _make_worker(total_layers: int = 32) -> NixlConnectorWorker:
    worker = NixlConnectorWorker.__new__(NixlConnectorWorker)
    worker.engine_id = LOCAL_ENGINE_ID
    worker.tp_rank = 0
    worker.world_size = 1
    worker.block_size = 16
    worker.num_blocks = 4
    worker._logical_num_blocks = worker.num_blocks
    worker._physical_blocks_per_logical_kv_block = 1
    worker.use_mla = False
    worker._has_mamba = False
    worker._is_hma_required = False
    worker._group_spec_types = (FullAttentionSpec,)
    worker.kv_cache_config = SimpleNamespace(kv_cache_groups=[object()])
    worker.transfer_topo = TransferTopology(
        tp_rank=worker.tp_rank,
        tp_size=worker.world_size,
        block_size=worker.block_size,
        engine_id=worker.engine_id,
        is_mla=worker.use_mla,
        is_mamba=False,
        total_num_kv_heads=8,
        attn_backends=[_FakeAttentionBackend],
        tensor_shape=_FakeAttentionBackend.get_kv_cache_shape(
            num_blocks=1,
            block_size=worker.block_size,
            num_kv_heads=1,
            head_size=1,
        ),
    )
    worker.model_config = SimpleNamespace(
        get_total_num_hidden_layers=lambda: total_layers
    )
    worker.kv_cache_layout = "HND"
    worker.host_buffer_kv_cache_layout = "HND"
    worker.use_host_buffer = False
    worker.kv_transfer_config = SimpleNamespace(enable_permute_local_kv=False)
    worker.backend_name = "FLASH_ATTN"
    worker.block_len_per_layer = [1024] * total_layers
    worker.local_registered_layer_indices = list(range(total_layers))
    worker.local_seen_layer_names = [
        f"model.layers.{layer}.self_attn" for layer in range(total_layers)
    ]
    worker._local_layer_name_to_region_indices = defaultdict(list)
    for idx, name in enumerate(worker.local_seen_layer_names):
        worker._local_layer_name_to_region_indices[name].append(idx)
    worker._layer_name_to_kv_group_index = {
        layer_name: 0 for layer_name in worker.local_seen_layer_names
    }
    worker.device_id = 0
    worker.nixl_memory_type = "DRAM"
    worker.nixl_wrapper = _FakeNixlWrapper()
    worker.kv_caches_base_addr = defaultdict(dict)
    worker._local_kv_cache_key = (0, worker.tp_rank)
    worker.kv_caches_base_addr[worker.engine_id][worker._local_kv_cache_key] = [
        100_000 + layer * 10_000 for layer in range(total_layers)
    ]
    worker._remote_agents = defaultdict(dict)
    worker._remote_agent_metadata = defaultdict(dict)
    worker._pp_layer_map = {}
    worker.src_xfer_handles_by_remote = {}
    worker.src_blocks_data_by_remote = {}
    worker.src_xfer_handles_by_shard_tp_ratio = {}
    worker.dst_xfer_side_handles = defaultdict(dict)
    worker._xfer_desc_layouts = {}
    worker.tp_mappings = {}
    worker.dst_num_blocks = {}
    worker._physical_blocks_per_logical = {}
    worker._recving_transfers = defaultdict(list)
    worker._recving_metadata = {}
    worker._failed_recv_reqs = set()
    worker._invalid_block_ids = set()
    worker.enable_permute_local_kv = False
    worker.enable_heterogeneous_attn_post_process = False
    worker.xfer_stats = SimpleNamespace(
        record_failed_notification=lambda: None,
        record_failed_transfer=lambda: None,
    )
    return worker


def _remote_meta(
    worker: NixlConnectorWorker,
    pp_rank: int,
    start_layer: int,
    end_layer: int,
    *,
    pp_size: int,
) -> NixlAgentMetadata:
    meta = _meta(
        pp_rank,
        start_layer,
        end_layer,
        pp_size=pp_size,
        registered_layer_indices=list(range(start_layer, end_layer)),
    )
    meta.num_blocks = worker.num_blocks
    meta.attn_backend_name = worker.backend_name
    meta.kv_caches_base_addr = [
        200_000 + layer * 10_000 for layer in range(start_layer, end_layer)
    ]
    return meta


def _add_two_remote_shards(worker: NixlConnectorWorker) -> list[NixlAgentMetadata]:
    metas = [
        _remote_meta(worker, 0, 0, 16, pp_size=2),
        _remote_meta(worker, 1, 16, 32, pp_size=2),
    ]
    for meta in metas:
        worker.add_remote_agent(
            meta,
            remote_tp_rank=0,
            remote_tp_size=1,
            remote_pp_rank=meta.pp_rank,
            remote_pp_size=meta.pp_size,
        )
    return metas


def test_add_remote_agent_records_both_pp_shard_base_address_keys():
    worker = _make_worker()

    _add_two_remote_shards(worker)

    assert set(worker.kv_caches_base_addr[REMOTE_ENGINE_ID]) == {(0, 0), (1, 0)}


def test_validate_remote_agent_handshake_accepts_synthetic_pp_shard():
    worker = _make_worker()
    meta = _remote_meta(worker, 0, 0, 16, pp_size=2)

    worker.add_remote_agent(
        meta,
        remote_tp_rank=0,
        remote_tp_size=1,
        remote_pp_rank=0,
        remote_pp_size=2,
    )
    worker._validate_remote_agent_handshake(meta, 0, 2, 1)


def test_add_remote_agent_prepares_dst_handles_for_each_pp_shard():
    worker = _make_worker()

    _add_two_remote_shards(worker)

    assert set(worker.dst_xfer_side_handles[REMOTE_ENGINE_ID]) == {
        (0, 0),
        (1, 0),
    }


def test_read_blocks_for_req_appends_one_transfer_per_pp_shard_and_tp_target():
    worker = _make_worker()
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


def test_pp_rank_one_descriptor_ids_are_shard_local():
    worker = _make_worker()
    _add_two_remote_shards(worker)

    remote_desc_ids = worker._get_block_descs_ids_for_shard(
        REMOTE_ENGINE_ID, 1, "remote", ([0],)
    )
    local_desc_ids = worker._get_block_descs_ids_for_shard(
        REMOTE_ENGINE_ID, 1, "local", ([0],)
    )

    assert remote_desc_ids[0] == 0
    assert local_desc_ids[0] == 0
