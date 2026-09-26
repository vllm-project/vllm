# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import contextlib
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
import zmq.asyncio

from tests.v1.attention.utils import dense_kv_cache_views
from vllm import envs
from vllm.config import set_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    _SHARED_REGION_GROUP_ID,
    KVConnectorRole,
    MooncakeConnector,
    MooncakeConnectorMetadata,
    MooncakeConnectorWorker,
    MooncakeXferMetadata,
    MooncakeXferResponse,
    MooncakeXferResponseStatus,
    PullReqMeta,
    SendBlockMeta,
    TransferRegion,
    _align_transfer_regions,
    _block_ids_for_region,
    _coalesce_contiguous_transfer_regions,
    _compute_sender_transfer_plan,
    _has_opaque_packed_row,
    _validate_asymmetric_region_lengths,
    get_mooncake_bootstrap_addr,
    should_launch_bootstrap_server,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (
    MooncakeBootstrapServer,
)
from vllm.utils.network_utils import get_open_port
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheLayout,
    MLAAttentionSpec,
)
from vllm.v1.request import RequestStatus

from .utils import create_request, create_scheduler, create_vllm_config


@pytest.mark.parametrize(
    ("remote_tp_size", "expected_dst_offsets"),
    [
        (1, [0, None, 65536, None, 131072, None, 196608, None]),
        (2, [0, None, 65536, None, 0, None, 65536, None]),
    ],
)
@pytest.mark.parametrize("local_tp_rank", range(8))
def test_sender_plan_replicated_gqa_tp8_to_smaller_tp(
    remote_tp_size,
    expected_dst_offsets,
    local_tp_rank,
):
    expected_dst_offset = expected_dst_offsets[local_tp_rank]
    plan = _compute_sender_transfer_plan(
        local_tp_rank=local_tp_rank,
        local_tp_size=8,
        remote_tp_rank=local_tp_rank // (8 // remote_tp_size),
        remote_tp_size=remote_tp_size,
        local_kv_block_len=65536,
        remote_kv_block_len=262144 // remote_tp_size,
        producer_cache_replicated=True,
        total_num_kv_heads=4,
    )
    if expected_dst_offset is None:
        assert not plan[0]
    else:
        assert plan == (True, 0, expected_dst_offset, 65536)


@pytest.mark.parametrize(
    ("local_tp_size", "expected_src_offsets"),
    [
        (1, [0, 0, 65536, 65536, 131072, 131072, 196608, 196608]),
        (2, [0, 0, 65536, 65536, 0, 0, 65536, 65536]),
        (4, [0] * 8),
    ],
)
@pytest.mark.parametrize("remote_tp_rank", range(8))
def test_sender_plan_gqa_to_replicated_tp8(
    local_tp_size,
    expected_src_offsets,
    remote_tp_rank,
):
    local_head_count = 4 // local_tp_size
    plan = _compute_sender_transfer_plan(
        local_tp_rank=remote_tp_rank // (8 // local_tp_size),
        local_tp_size=local_tp_size,
        remote_tp_rank=remote_tp_rank,
        remote_tp_size=8,
        local_kv_block_len=local_head_count * 65536,
        remote_kv_block_len=65536,
        producer_cache_replicated=False,
        total_num_kv_heads=4,
    )
    assert plan == (True, expected_src_offsets[remote_tp_rank], 0, 65536)


@pytest.mark.parametrize(
    "local_tp,remote_tp,local_len,remote_len,valid",
    [
        (1, 8, 262144, 65536, True),
        (8, 1, 65536, 262144, True),
        (1, 8, 262144, 131072, False),
        (8, 1, 131072, 262144, False),
    ],
)
def test_region_length_validation_checks_replicated_gqa_heads(
    local_tp, remote_tp, local_len, remote_len, valid
):
    """Replicated heads must have matching, whole per-head payloads."""
    local_region = TransferRegion("layer", 0, 0, local_len, local_len)
    remote_region = TransferRegion("layer", 0, 0, remote_len, remote_len)

    assert (
        _validate_asymmetric_region_lengths(
            local_regions=[local_region],
            remote_regions=[remote_region],
            local_tp_size=local_tp,
            remote_tp_size=remote_tp,
            producer_cache_replicated=local_tp > 4,
            total_num_kv_heads=4,
        )
        is None
    ) == valid


def _make_test_kv_cache_config() -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=0,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                [
                    "model.layers.0.self_attn",
                    "model.layers.1.self_attn",
                    "model.layers.0.mla_attn",
                    "model.layers.1.eagle_attn",
                ],
                FullAttentionSpec(
                    block_size=16,
                    num_kv_heads=4,
                    head_size=64,
                    dtype=torch.float16,
                ),
            )
        ],
    )


def _make_packed_mla_kv_cache_config(
    num_blocks: int, *, num_groups: int = 1
) -> KVCacheConfig:
    spec = MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.uint8,
    )
    if num_groups == 1:
        groups = [
            KVCacheGroupSpec(
                ["model.layers.0.mla_attn", "model.layers.1.indexer"],
                spec,
            )
        ]
    else:
        groups = [
            KVCacheGroupSpec(["model.layers.0.mla_attn"], spec),
            KVCacheGroupSpec(["model.layers.1.indexer"], spec),
        ]
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
    )


class FakeMooncakeWrapper:
    """Mock Mooncake TransferEngine for unit testing environments."""

    def __init__(self, *args, **kwargs):
        pass

    def initialize(self, local_hostname, metadata_server, protocol, device_name) -> int:
        return 0

    def get_rpc_port(self) -> int:
        return 12345

    def batch_transfer_sync_write(
        self, target_hostname, buffers, peer_buffer_addresses, lengths
    ) -> int:
        return 0

    def batch_register_memory(self, buffer_addresses, capacities) -> int:
        return 0


def test_align_transfer_regions_uses_layer_name_occurrences():
    """Repeated layer names should align by occurrence order."""
    local_regions = [
        TransferRegion(
            layer_name="model.layers.1.self_attn",
            layer_index=1,
            base_addr=0x1000,
            block_len=256,
            kv_block_len=128,
        ),
        TransferRegion(
            layer_name="model.layers.1.self_attn",
            layer_index=1,
            base_addr=0x1100,
            block_len=256,
            kv_block_len=128,
        ),
    ]
    remote_regions = [
        TransferRegion(
            layer_name="model.layers.0.self_attn",
            layer_index=0,
            base_addr=0xA000,
            block_len=256,
            kv_block_len=128,
        ),
        TransferRegion(
            layer_name="model.layers.1.self_attn",
            layer_index=1,
            base_addr=0xB000,
            block_len=256,
            kv_block_len=128,
        ),
        TransferRegion(
            layer_name="model.layers.1.self_attn",
            layer_index=1,
            base_addr=0xB100,
            block_len=256,
            kv_block_len=128,
        ),
    ]

    aligned_local, aligned_remote, err = _align_transfer_regions(
        local_regions, remote_regions
    )

    assert err is None
    assert [r.base_addr for r in aligned_local] == [0x1000, 0x1100]
    assert [r.base_addr for r in aligned_remote] == [0xB000, 0xB100]


@pytest.mark.asyncio
async def test_build_transfer_params_separates_prefill_pp_layers():
    """Each producer PP stage should send only its registered layer shard."""
    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker.async_zmq_ctx = MagicMock()
    worker.is_kv_consumer = True
    worker.is_kv_producer = True
    worker.tp_rank = 0
    worker.tp_size = 1
    worker.use_mla = False
    worker.kv_cache_config = _make_test_kv_cache_config()
    worker._physical_blocks_per_logical_kv_block = 1
    worker.transfer_topo = SimpleNamespace(
        local_replicates_kv_cache=False,
        total_num_kv_heads=4,
    )

    block_len = 256
    remote_regions = [
        TransferRegion(
            layer_name=f"model.layers.{layer_index}.self_attn",
            layer_index=layer_index,
            base_addr=base_addr,
            block_len=block_len,
            kv_block_len=block_len,
        )
        for layer_index, base_addr in [
            (0, 0xA000),
            (1, 0xB000),
            (2, 0xC000),
            (3, 0xD000),
        ]
    ]
    producer_pp_regions = {
        0: [
            TransferRegion(
                layer_name="model.layers.0.self_attn",
                layer_index=0,
                base_addr=0x1000,
                block_len=block_len,
                kv_block_len=block_len,
            ),
            TransferRegion(
                layer_name="model.layers.1.self_attn",
                layer_index=1,
                base_addr=0x2000,
                block_len=block_len,
                kv_block_len=block_len,
            ),
        ],
        1: [
            TransferRegion(
                layer_name="model.layers.2.self_attn",
                layer_index=2,
                base_addr=0x3000,
                block_len=block_len,
                kv_block_len=block_len,
            ),
            TransferRegion(
                layer_name="model.layers.3.self_attn",
                layer_index=3,
                base_addr=0x4000,
                block_len=block_len,
                kv_block_len=block_len,
            ),
        ],
    }
    expected_by_pp_rank = {
        0: {
            "layers": [0, 1],
            "src_ptrs": [0x1000 + 10 * block_len, 0x2000 + 10 * block_len],
            "dst_ptrs": [0xA000 + 20 * block_len, 0xB000 + 20 * block_len],
        },
        1: {
            "layers": [2, 3],
            "src_ptrs": [0x3000 + 10 * block_len, 0x4000 + 10 * block_len],
            "dst_ptrs": [0xC000 + 20 * block_len, 0xD000 + 20 * block_len],
        },
    }

    transfer_id = "xfer-pp-split"
    send_meta = SendBlockMeta(
        p_req_id="p-req-pp",
        transfer_id=transfer_id,
        local_block_ids=[[10, 11]],
        ready=asyncio.Event(),
    )
    xfer_meta = MooncakeXferMetadata(
        remote_hostname="consumer-host",
        remote_port=54321,
        remote_tp_size=1,
        remote_tp_rank=0,
        req_blocks={"d-req-pp": (transfer_id, [[20, 21]])},
        kv_caches_base_addr=[region.base_addr for region in remote_regions],
        block_lens=[region.block_len for region in remote_regions],
        kv_block_lens=[region.kv_block_len for region in remote_regions],
        registered_layer_names=[region.layer_name for region in remote_regions],
        registered_layer_indices=[region.layer_index for region in remote_regions],
    )

    for pp_rank, local_regions in producer_pp_regions.items():
        aligned_local, aligned_remote, err = _align_transfer_regions(
            local_regions, remote_regions
        )

        assert err is None
        assert [r.layer_index for r in aligned_local] == (
            expected_by_pp_rank[pp_rank]["layers"]
        )
        assert [r.layer_index for r in aligned_remote] == (
            expected_by_pp_rank[pp_rank]["layers"]
        )

        (
            src_ptrs,
            dst_ptrs,
            lengths,
            err_reqs,
            err_msg,
        ) = await worker._build_transfer_params(
            ready_reqs=[("d-req-pp", send_meta)],
            agent_meta=xfer_meta,
            local_regions=aligned_local,
            remote_regions=aligned_remote,
        )

        assert err_reqs == []
        assert err_msg is None
        assert src_ptrs == expected_by_pp_rank[pp_rank]["src_ptrs"]
        assert dst_ptrs == expected_by_pp_rank[pp_rank]["dst_ptrs"]
        assert lengths == [2 * block_len, 2 * block_len]


@pytest.mark.asyncio
async def test_send_kv_to_decode_aligns_consumer_regions_by_layer_metadata(
    monkeypatch,
):
    """Producer sends its PP layer shard to the matching consumer layer address."""
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector", kv_role="kv_producer"
    )

    with set_current_vllm_config(vllm_config), patch_worker_dependencies():
        prefill_connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            _make_test_kv_cache_config(),
        )
        prefill_worker = prefill_connector.connector_worker

        block_len = 4096
        prefill_worker.kv_caches_base_addr = [0x1000]
        prefill_worker.block_len_per_layer = [block_len]
        prefill_worker.kv_block_len_per_layer = [block_len]
        prefill_worker.registered_layer_names = ["model.layers.1.self_attn"]
        prefill_worker.registered_layer_indices = [1]

        class InlineSenderLoop:
            async def run_in_executor(self, executor, func, *args):
                return func(*args)

        origin_sender_loop = prefill_worker.sender_loop
        prefill_worker.sender_loop = InlineSenderLoop()

        transfer_id = "xfer-layer-align"
        send_meta = SendBlockMeta(
            p_req_id="p-req-layer-align",
            transfer_id=transfer_id,
            local_block_ids=[[10]],
            ready=asyncio.Event(),
        )
        prefill_worker.reqs_need_send[transfer_id] = send_meta
        send_meta.ready.set()

        xfer_meta = MooncakeXferMetadata(
            remote_hostname="consumer-host",
            remote_port=54321,
            remote_tp_size=1,
            remote_tp_rank=0,
            req_blocks={"d-req-layer-align": (transfer_id, [[20]])},
            kv_caches_base_addr=[0xA000, 0xB000],
            block_lens=[block_len, block_len],
            kv_block_lens=[block_len, block_len],
            registered_layer_names=[
                "model.layers.0.self_attn",
                "model.layers.1.self_attn",
            ],
            registered_layer_indices=[0, 1],
        )
        mock_socket = AsyncMock(spec=zmq.asyncio.Socket)
        mock_socket.send_multipart = AsyncMock()
        identity = b"consumer-layer-align"

        with patch.object(
            prefill_worker, "_send_blocks", return_value=0
        ) as mock_send_blocks:
            await prefill_worker.send_kv_to_decode(identity, mock_socket, xfer_meta)

        src_ptrs, dst_ptrs, lengths = mock_send_blocks.call_args[0][1:]
        assert src_ptrs == [0x1000 + 10 * block_len]
        assert dst_ptrs == [0xB000 + 20 * block_len]
        assert lengths == [block_len]

        sent_identity, sent_payload = mock_socket.send_multipart.call_args[0][0]
        assert sent_identity == identity
        response = prefill_worker._xfer_resp_decoder.decode(sent_payload)
        assert response.status == MooncakeXferResponseStatus.FINISH
        assert response.ok_reqs == ["d-req-layer-align"]

        prefill_worker.sender_loop = origin_sender_loop
        prefill_worker.shutdown()


def test_basic_interface():
    """Unit test for basic MooncakeConnector interface functionality."""
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector", kv_role="kv_consumer"
    )
    scheduler = create_scheduler(vllm_config)

    # 2 Full Blocks and 1 Half Block.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_EXTERNAL_FULL_BLOCKS = 2
    NUM_TOKENS = int(BLOCK_SIZE * (NUM_EXTERNAL_FULL_BLOCKS + 0.5))

    request = create_request(
        request_id=1,
        block_size=BLOCK_SIZE,
        num_tokens=NUM_TOKENS,
        do_remote_prefill=True,
    )
    request_id = request.request_id
    request.kv_transfer_params.update(
        {
            "transfer_id": request_id,
            "remote_bootstrap_addr": 54321,
        }
    )

    scheduler.add_request(request)

    # Remote Prefill, triggers NixlConnectorMetadata.
    scheduler_output = scheduler.schedule()
    kv_connector_metadata = scheduler_output.kv_connector_metadata
    assert kv_connector_metadata is not None
    assert isinstance(kv_connector_metadata, MooncakeConnectorMetadata)

    assert len(kv_connector_metadata.reqs_to_recv) == 1
    assert request_id in kv_connector_metadata.reqs_to_recv["my-engine-id"]
    req_meta = kv_connector_metadata.reqs_to_recv["my-engine-id"][request_id]

    # local_block_ids is list[list[int]] (per-group); flatten for comparison.
    all_block_ids = [bid for group in req_meta.local_block_ids for bid in group]
    for block_id, block in zip(
        all_block_ids,
        scheduler.kv_cache_manager.coordinator.single_type_managers[0].req_to_blocks[
            request_id
        ],
    ):
        assert block_id == block.block_id


def test_prompt_less_than_block_size():
    """Test that we can handle case where prompt is < block."""
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector", kv_role="kv_consumer"
    )
    scheduler = create_scheduler(vllm_config)

    # Half of a block.
    BLOCK_SIZE = vllm_config.cache_config.block_size
    NUM_TOKENS = int(BLOCK_SIZE * 0.5)

    # Request will have 1 partial remote block.
    request = create_request(
        request_id=1,
        block_size=BLOCK_SIZE,
        num_tokens=NUM_TOKENS,
        do_remote_prefill=True,
        num_remote_blocks=1,
    )
    request.kv_transfer_params.update(
        {
            "transfer_id": request.request_id,
            "remote_bootstrap_addr": 54321,
        }
    )

    scheduler.add_request(request)
    scheduler_output = scheduler.schedule()

    # This request will read async.
    kv_connector_metadata = scheduler_output.kv_connector_metadata
    assert kv_connector_metadata is not None
    assert isinstance(kv_connector_metadata, MooncakeConnectorMetadata)
    assert len(kv_connector_metadata.reqs_to_recv["my-engine-id"]) == 1
    assert len(scheduler_output.scheduled_new_reqs) == 0


@pytest.fixture
def bootstrap_server():
    """Fixture to launch and cleanup a Mooncake Bootstrap HTTP Server."""
    port = get_open_port()
    server = MooncakeBootstrapServer("127.0.0.1", port)
    server.start()
    yield server
    server.shutdown()


@pytest.mark.asyncio
async def test_bootstrap_server(bootstrap_server: MooncakeBootstrapServer):
    """Tests the bootstrap server's api for worker registration and querying.

    Validates DP/TP/PP rank indexing and error handling for duplicate registrations.
    """
    import httpx

    base_url = f"http://127.0.0.1:{bootstrap_server.port}"

    # Query when empty
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{base_url}/query")
        assert response.status_code == 200
        assert response.json() == {}

    # Register multiple PP workers from the same producer engine.
    payload1 = {
        "engine_id": "eng-1",
        "dp_rank": 0,
        "tp_rank": 0,
        "pp_rank": 0,
        "addr": "tcp://1.1.1.1:1111",
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{base_url}/register", json=payload1)
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    payload2 = {
        "engine_id": "eng-1",
        "dp_rank": 0,
        "tp_rank": 0,
        "pp_rank": 1,
        "addr": "tcp://2.2.2.2:2222",
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{base_url}/register", json=payload2)
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    # Query after registration should preserve the PP dimension.
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{base_url}/query")
        assert response.status_code == 200
        data = response.json()
        assert "0" in data
        assert data["0"]["engine_id"] == "eng-1"
        assert data["0"]["worker_addr"]["0"]["0"] == "tcp://1.1.1.1:1111"
        assert data["0"]["worker_addr"]["0"]["1"] == "tcp://2.2.2.2:2222"

    # Test failure: re-registering the same worker
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{base_url}/register", json=payload1)
        assert response.status_code == 400
        assert "is already registered" in response.text

    # Test failure: engine_id mismatch for same dp_rank
    payload3_fail = {
        "engine_id": "eng-2",
        "dp_rank": 0,
        "tp_rank": 1,
        "pp_rank": 0,
        "addr": "tcp://3.3.3.3:3333",
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{base_url}/register", json=payload3_fail)
        assert response.status_code == 400
        assert "Engine ID mismatch" in response.text


def _make_bootstrap_vllm_config(
    *,
    local_engines_only: bool = False,
    data_parallel_rank_local: int = 0,
    data_parallel_index: int = 0,
    nnodes_within_dp: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            local_engines_only=local_engines_only,
            data_parallel_rank_local=data_parallel_rank_local,
            data_parallel_index=data_parallel_index,
            nnodes_within_dp=nnodes_within_dp,
            master_addr="model-parallel-master",
            data_parallel_master_ip="data-parallel-master",
        )
    )


@pytest.mark.parametrize(
    (
        "tp_rank",
        "pp_rank",
        "local_engines_only",
        "data_parallel_rank_local",
        "data_parallel_index",
        "expected",
    ),
    [
        (1, 0, False, 0, 0, False),
        (0, 1, False, 0, 0, False),
        (0, 0, True, 0, 1, True),
        (0, 0, True, 1, 0, False),
        (0, 0, False, 0, 0, True),
        (0, 0, False, 0, 1, False),
    ],
    ids=[
        "nonzero_tp_rank",
        "nonzero_pp_rank",
        "local_engine_rank_zero",
        "local_engine_nonzero_rank",
        "internal_lb_first_dp_engine",
        "internal_lb_nonzero_dp_engine",
    ],
)
def test_should_launch_bootstrap_server_selects_single_owner(
    tp_rank: int,
    pp_rank: int,
    local_engines_only: bool,
    data_parallel_rank_local: int,
    data_parallel_index: int,
    expected: bool,
):
    vllm_config = _make_bootstrap_vllm_config(
        local_engines_only=local_engines_only,
        data_parallel_rank_local=data_parallel_rank_local,
        data_parallel_index=data_parallel_index,
    )
    with (
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_connector.get_tensor_model_parallel_rank",
            return_value=tp_rank,
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
            "mooncake_connector.get_pp_group"
        ) as mock_pp_group,
    ):
        mock_pp_group.return_value.rank_in_group = pp_rank
        assert should_launch_bootstrap_server(vllm_config) is expected


@pytest.mark.parametrize(
    ("local_engines_only", "nnodes_within_dp", "expected_host"),
    [
        (True, 2, "127.0.0.1"),
        (False, 2, "model-parallel-master"),
        (False, 1, "data-parallel-master"),
    ],
    ids=["local_engine", "multi_node_tp_or_pp", "single_node_internal_lb"],
)
def test_get_mooncake_bootstrap_addr_selects_expected_host(
    local_engines_only: bool,
    nnodes_within_dp: int,
    expected_host: str,
):
    vllm_config = _make_bootstrap_vllm_config(
        local_engines_only=local_engines_only,
        nnodes_within_dp=nnodes_within_dp,
    )

    assert get_mooncake_bootstrap_addr(vllm_config) == (
        expected_host,
        envs.VLLM_MOONCAKE_BOOTSTRAP_PORT,
    )


def test_scheduler_request_finished():
    """Tests the scheduler-side logic when a request finishes.

    Differentiates between 'Finished' (requires transfer)
    and 'Aborted' (immediate free).
    """
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector", kv_role="kv_producer"
    )
    scheduler = create_scheduler(vllm_config)
    scheduler_connector = scheduler.get_kv_connector().connector_scheduler

    request = create_request(request_id=1, do_remote_decode=True)
    request.kv_transfer_params["transfer_id"] = request.request_id

    # Case: Capped length (Successful prefill, need to send to decoder)
    request.status = RequestStatus.FINISHED_LENGTH_CAPPED
    delay_free, _ = scheduler_connector.request_finished(request, block_ids=([10, 11],))
    assert delay_free is True
    assert "id-1" in scheduler_connector._reqs_need_send
    assert scheduler_connector._reqs_need_send["id-1"][1] == [[10, 11]]

    # Case: Aborted (No need to transfer, free blocks immediately)
    scheduler_connector._reqs_need_send.clear()
    request.status = RequestStatus.FINISHED_ABORTED
    delay_free, _ = scheduler_connector.request_finished(request, block_ids=([12],))
    assert delay_free is False
    assert len(scheduler_connector._reqs_need_send) == 0
    assert "id-1" in scheduler_connector._reqs_not_processed


@contextlib.contextmanager
def patch_worker_dependencies():
    """Helper to mock all distributed and network dependencies for Worker tests."""
    with (
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.TransferEngine",
            FakeMooncakeWrapper,
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.get_ip",
            return_value="127.0.0.1",
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.get_tensor_model_parallel_rank",
            return_value=0,
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.get_tensor_model_parallel_world_size",
            return_value=1,
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.get_pp_group"
        ) as mock_pp,
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.should_launch_bootstrap_server",
            return_value=False,
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.make_zmq_socket"
        ) as mock_make_zmq,
        patch("httpx.AsyncClient") as mock_async_client,
    ):
        # Mock PP group
        mock_pp_group = MagicMock()
        mock_pp_group.rank_in_group = 0
        mock_pp.return_value = mock_pp_group

        # Mock ZMQ socket
        mock_socket_object = AsyncMock()
        mock_socket_object.setsockopt = MagicMock()
        mock_socket_ctx = MagicMock()
        mock_socket_ctx.__enter__.return_value = mock_socket_object
        mock_make_zmq.return_value = mock_socket_ctx

        # Mock httpx client
        mock_http_client_instance = AsyncMock()
        mock_async_client.return_value = mock_http_client_instance

        yield {
            "mock_make_zmq": mock_make_zmq,
            "mock_socket_object": mock_socket_object,
            "mock_async_client": mock_async_client,
            "mock_http_client": mock_http_client_instance,
        }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("local_pp_size", "local_pp_rank", "expected_addrs"),
    [
        (1, 0, ["tcp://producer-pp0:1234", "tcp://producer-pp1:1234"]),
        (2, 1, ["tcp://producer-pp1:1234"]),
    ],
    ids=["heterogeneous_pp_pulls_all_remote_pp", "matching_pp_pulls_same_rank"],
)
async def test_receive_kv_selects_remote_pp_workers(
    local_pp_size: int,
    local_pp_rank: int,
    expected_addrs: list[str],
):
    """Decode workers should not hard-code producer pp_rank 0."""
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector", kv_role="kv_consumer"
    )

    with set_current_vllm_config(vllm_config), patch_worker_dependencies():
        decode_connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            _make_test_kv_cache_config(),
        )
        decode_worker = decode_connector.connector_worker
        decode_worker.pp_size = local_pp_size
        decode_worker.pp_rank = local_pp_rank
        decode_worker._remote_agents = {
            "p-engine": {
                0: {
                    0: "tcp://producer-pp0:1234",
                    1: "tcp://producer-pp1:1234",
                }
            }
        }
        decode_worker._tp_size["p-engine"] = 1

        pull_metas = {
            "d-req-1": PullReqMeta(
                d_req_id="d-req-1",
                transfer_id="xfer-req-1",
                local_block_ids=[[100, 101]],
                remote_engine_id="p-engine",
                remote_bootstrap_addr="http://bootstrap:33333",
            )
        }
        seen_addrs: list[str] = []

        async def fake_receive(worker_addr: str, metas: dict[str, PullReqMeta]):
            seen_addrs.append(worker_addr)
            for meta in metas.values():
                meta.pull_tasks_count -= 1

        with patch.object(
            decode_worker,
            "receive_kv_from_single_worker",
            side_effect=fake_receive,
        ):
            decode_worker.receive_kv("p-engine", pull_metas)
            await asyncio.sleep(0)

        assert seen_addrs == expected_addrs
        assert pull_metas["d-req-1"].pull_tasks_count == 0
        decode_worker.shutdown()


def test_resolve_need_send_accounts_for_remote_tp_fanout():
    """Producer-side completion waits for every paired consumer TP pull."""
    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker.async_zmq_ctx = MagicMock()
    worker.is_kv_consumer = True
    worker.is_kv_producer = True
    send_meta = SendBlockMeta(
        p_req_id="p-req-1",
        transfer_id="xfer-req-1",
        local_block_ids=[[1]],
        ready=asyncio.Event(),
    )

    worker.resolve_need_send(send_meta, remote_tp_ranks=[0, 1])

    assert send_meta.need_send == 2


@pytest.mark.asyncio
async def test_heterogeneous_pp_waits_for_consumer_without_shared_layers():
    """P4/D2 must retain KV until both D stages finish, including a no-op pull."""
    config = create_vllm_config(kv_connector="MooncakeConnector", kv_role="kv_producer")
    with (
        set_current_vllm_config(config),
        patch_worker_dependencies(),
        patch.object(MooncakeConnectorWorker, "_sync_block_size_with_kernel"),
    ):
        worker = MooncakeConnector(
            config, KVConnectorRole.WORKER, _make_test_kv_cache_config()
        ).connector_worker
        try:
            worker.pp_size = 4
            worker.kv_caches_base_addr = [0x1000]
            worker.block_len_per_layer = [256]
            worker.kv_block_len_per_layer = [256]
            worker.registered_layer_names = ["model.layers.0.self_attn"]
            worker.registered_layer_indices = [0]
            send_meta = SendBlockMeta(
                p_req_id="p-req",
                transfer_id="transfer",
                local_block_ids=[[1]],
                ready=asyncio.Event(),
            )
            send_meta.ready.set()
            worker.reqs_need_send["transfer"] = send_meta
            sock = AsyncMock(spec=zmq.asyncio.Socket, send_multipart=AsyncMock())
            with (
                patch.object(worker, "sender_loop", asyncio.get_running_loop()),
                patch.object(worker, "_send_blocks", return_value=0) as send,
            ):
                for pp_rank in range(2):
                    metadata = MooncakeXferMetadata(
                        remote_hostname="consumer",
                        remote_port=1234 + pp_rank,
                        remote_tp_size=1,
                        remote_tp_rank=0,
                        remote_pp_size=2,
                        req_blocks={"d-req": ("transfer", [[2]])},
                        kv_caches_base_addr=[0x2000],
                        block_lens=[256],
                        kv_block_lens=[256],
                        registered_layer_names=[f"model.layers.{pp_rank}.self_attn"],
                        registered_layer_indices=[pp_rank],
                    )
                    await worker.send_kv_to_decode(b"consumer", sock, metadata)
                    response = worker._xfer_resp_decoder.decode(
                        sock.send_multipart.call_args.args[0][1]
                    )
                    assert response.status == MooncakeXferResponseStatus.FINISH
                    assert response.ok_reqs == ["d-req"]
                    assert ("transfer" in worker.reqs_need_send) == (pp_rank == 0)
                    assert worker.finished_sending_reqs == (
                        set() if pp_rank == 0 else {"p-req"}
                    )
                send.assert_called_once_with("consumer:1234", [0x1100], [0x2200], [256])
        finally:
            worker.shutdown()
            worker.is_kv_consumer = True


@pytest.mark.asyncio
@patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.TransferEngine",
    FakeMooncakeWrapper,
)
async def test_kv_producer(monkeypatch):
    """Simulates a Producer Worker (Prefiller) receiving a transfer request
    from a Consumer (Decoder).

    Verifies memory offset calculation: ptr = base_addr + block_id * block_len.
    """
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector", kv_role="kv_producer"
    )

    with set_current_vllm_config(vllm_config), patch_worker_dependencies():
        prefill_connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            _make_test_kv_cache_config(),
        )
        prefill_worker = prefill_connector.connector_worker
        prefill_worker.kv_caches_base_addr = [0x1000]
        block_len = 4096
        prefill_worker.block_len_per_layer = [block_len]
        prefill_worker.kv_block_len_per_layer = [block_len]
        prefill_worker.registered_layer_names = ["model.layers.0.self_attn"]
        prefill_worker.registered_layer_indices = [0]

        # Override loop to use current test loop
        origin_sender_loop = prefill_worker.sender_loop
        prefill_worker.sender_loop = asyncio.get_event_loop()

        # A request is finished on Producer and ready to be sent.
        transfer_id = "xfer-req-1"
        send_meta = SendBlockMeta(
            p_req_id="p-req-1",
            transfer_id=transfer_id,
            local_block_ids=[[10, 11]],
            ready=asyncio.Event(),
        )
        prefill_worker.reqs_need_send[transfer_id] = send_meta
        send_meta.ready.set()

        # Remote consumer request metadata
        xfer_meta = MooncakeXferMetadata(
            remote_hostname="consumer-host",
            remote_port=54321,
            remote_tp_size=1,
            remote_tp_rank=0,
            req_blocks={"d-req-1": (transfer_id, [[20, 21]])},
            kv_caches_base_addr=[0x2000],
            block_lens=[block_len],
            kv_block_lens=[block_len],
            registered_layer_names=["model.layers.0.self_attn"],
            registered_layer_indices=[0],
        )

        mock_socket = AsyncMock(spec=zmq.asyncio.Socket)
        mock_socket.send_multipart = AsyncMock()
        identity = b"consumer-id"

        with patch.object(
            prefill_worker, "_send_blocks", return_value=0
        ) as mock_send_blocks:

            def expected_transfers(src_base, dst_base, src_blocks, dst_blocks):
                n = len(src_blocks)
                return (
                    [src_base + src_blocks[0] * block_len],
                    [dst_base + dst_blocks[0] * block_len],
                    [n * block_len],
                )

            # Normal case: 2 blocks to 2 blocks
            await prefill_worker.send_kv_to_decode(identity, mock_socket, xfer_meta)
            src, dst, lens = expected_transfers(0x1000, 0x2000, [10, 11], [20, 21])
            mock_send_blocks.assert_called_once_with(
                "consumer-host:54321",
                src,
                dst,
                lens,
            )
            mock_socket.send_multipart.assert_called_once()

            # Verify the response sent back to the consumer
            sent_call = mock_socket.send_multipart.call_args[0][0]
            sent_identity, sent_payload = sent_call
            assert sent_identity == identity
            response = prefill_worker._xfer_resp_decoder.decode(sent_payload)
            assert response.status == MooncakeXferResponseStatus.FINISH
            assert response.ok_reqs == ["d-req-1"]

            # Verify internal state cleanup
            assert transfer_id not in prefill_worker.reqs_need_send
            assert "p-req-1" in prefill_worker.finished_sending_reqs

            # More cases:
            # Consumer only needs 1 block (less than P)
            mock_send_blocks.reset_mock()
            mock_socket.send_multipart.reset_mock()
            prefill_worker.reqs_need_send[transfer_id] = send_meta
            send_meta.sent = 0
            send_meta.ready.set()
            xfer_meta.req_blocks["d-req-1"] = (transfer_id, [[20]])
            # Worker processes the consumer's request
            await prefill_worker.send_kv_to_decode(identity, mock_socket, xfer_meta)
            # Verify transfer parameters are correct: 11 to 20
            src, dst, lens = expected_transfers(0x1000, 0x2000, [11], [20])
            mock_send_blocks.assert_called_once_with(
                "consumer-host:54321",
                src,
                dst,
                lens,
            )
            mock_socket.send_multipart.assert_called_once()

            # Consumer needs 3 blocks (more than P, error case)
            mock_send_blocks.reset_mock()
            mock_socket.send_multipart.reset_mock()
            prefill_worker.reqs_need_send[transfer_id] = send_meta
            send_meta.sent = 0
            send_meta.ready.set()
            xfer_meta.req_blocks["d-req-1"] = (transfer_id, [[20, 21, 22]])
            # Worker processes the consumer's request
            await prefill_worker.send_kv_to_decode(identity, mock_socket, xfer_meta)
            # This should not be called because error.
            mock_send_blocks.assert_not_called()
            mock_socket.send_multipart.assert_called_once()
            _, sent_payload = mock_socket.send_multipart.call_args[0][0]
            response = prefill_worker._xfer_resp_decoder.decode(sent_payload)
            assert response.err_msg == "P num blocks less than D"
            assert response.err_reqs == ["d-req-1"]

            # Timeout
            mock_send_blocks.reset_mock()
            mock_socket.send_multipart.reset_mock()
            prefill_worker.reqs_need_send[transfer_id] = send_meta
            send_meta.sent = 0
            send_meta.ready.clear()
            xfer_meta.req_blocks["d-req-1"] = (transfer_id, [[20, 21]])
            # Worker processes the consumer's request
            await prefill_worker.send_kv_to_decode(identity, mock_socket, xfer_meta)
            # This should not be called because timeout.
            mock_send_blocks.assert_not_called()
            mock_socket.send_multipart.assert_called_once()
            _, sent_payload = mock_socket.send_multipart.call_args[0][0]
            response = prefill_worker._xfer_resp_decoder.decode(sent_payload)
            assert response.err_msg == "Timeout waiting for P side ready."
            assert response.err_reqs == ["d-req-1"]

        # Transfer error
        with patch.object(
            prefill_worker, "_send_blocks", return_value=123
        ) as mock_send_blocks:
            mock_socket.send_multipart.reset_mock()
            prefill_worker.reqs_need_send[transfer_id] = send_meta
            send_meta.sent = 0
            send_meta.ready.set()
            xfer_meta.req_blocks["d-req-1"] = (transfer_id, [[20, 21]])
            # Worker processes the consumer's request
            await prefill_worker.send_kv_to_decode(identity, mock_socket, xfer_meta)
            mock_send_blocks.assert_called_once()
            mock_socket.send_multipart.assert_called_once()
            _, sent_payload = mock_socket.send_multipart.call_args[0][0]
            response = prefill_worker._xfer_resp_decoder.decode(sent_payload)
            assert response.err_msg == "Mooncake transfer engine returned 123"
            assert response.err_reqs == ["d-req-1"]

        # Clean up
        prefill_worker.sender_loop = origin_sender_loop
        prefill_worker.shutdown()


@pytest.mark.asyncio
async def test_kv_consumuer(monkeypatch):
    """Simulates a Consumer Worker (Decoder) initiating a pull from a Producer.

    Verifies that MooncakeXferMetadata is correctly serialized and sent via ZMQ.
    """
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector", kv_role="kv_consumer"
    )

    with set_current_vllm_config(vllm_config), patch_worker_dependencies() as mocks:
        decode_connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            _make_test_kv_cache_config(),
        )
        decode_worker = decode_connector.connector_worker
        decode_worker.kv_caches_base_addr = [0x1000]
        decode_worker.block_len_per_layer = [4096]
        decode_worker.kv_block_len_per_layer = [4096]
        decode_worker.registered_layer_names = ["model.layers.0.self_attn"]
        decode_worker.registered_layer_indices = [0]
        decode_worker.rpc_port = 54321

        # A request to pull data arrives.
        pull_metas = {
            "d-req-1": PullReqMeta(
                d_req_id="d-req-1",
                transfer_id="xfer-req-1",
                local_block_ids=[[100, 101]],
                remote_engine_id="p-engine",
                remote_bootstrap_addr="http://bootstrap:33333",
                pull_tasks_count=1,
            )
        }
        decode_worker._remote_agents = {"p-engine": {0: {0: "tcp://producer:1234"}}}
        decode_worker._tp_size["p-engine"] = 1

        # Mock the response from the producer.
        mock_response = MooncakeXferResponse(
            status=MooncakeXferResponseStatus.FINISH, ok_reqs=["d-req-1"]
        )
        encoded_response = decode_worker._encoder.encode(mock_response)
        mocks["mock_socket_object"].recv.return_value = encoded_response

        # Trigger the receive logic.
        decode_worker.receive_kv("p-engine", pull_metas)
        await asyncio.sleep(1)  # Allow async task to run

        # Verify the metadata sent to the producer.
        mocks["mock_make_zmq"].assert_called_with(
            decode_worker.async_zmq_ctx,
            "tcp://producer:1234",
            zmq.DEALER,
            bind=False,
            linger=0,
        )
        sent_payload = mocks["mock_socket_object"].send.call_args[0][0]
        sent_meta = decode_worker._xfer_meta_decoder.decode(sent_payload)

        assert sent_meta.remote_hostname == "127.0.0.1"
        assert sent_meta.remote_port == 54321
        assert sent_meta.req_blocks["d-req-1"] == ("xfer-req-1", [[100, 101]])
        assert sent_meta.kv_caches_base_addr == [0x1000]
        assert sent_meta.block_lens == [4096]
        assert sent_meta.registered_layer_names == ["model.layers.0.self_attn"]
        assert sent_meta.registered_layer_indices == [0]

        # Verify internal state is updated correctly.
        assert "d-req-1" in decode_worker.finished_recving_reqs

        # Clean up
        decode_worker.shutdown()


@pytest.mark.asyncio
async def test_worker_get_finished_timeout(monkeypatch):
    """Tests the cleanup mechanism for requests."""
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector", kv_role="kv_producer"
    )
    with set_current_vllm_config(vllm_config), patch_worker_dependencies():
        prefill_connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            _make_test_kv_cache_config(),
        )
        prefill_worker = prefill_connector.connector_worker

        # Add an expired request (expire_time is in the past).
        prefill_worker.reqs_need_send["tx-expired"] = SendBlockMeta(
            p_req_id="p-req-expired",
            transfer_id="tx-expired",
            local_block_ids=[[1, 2]],
            ready=MagicMock(),
            expire_time=time.perf_counter() - 100,
        )

        # Add a non-expired request.
        prefill_worker.reqs_need_send["tx-active"] = SendBlockMeta(
            p_req_id="p-req-active",
            transfer_id="tx-active",
            local_block_ids=[[3, 4]],
            ready=MagicMock(),
            expire_time=time.perf_counter() + 100,
        )

        finished_reqs = await prefill_worker.fetch_finished_sending_reqs()

        assert "p-req-expired" in finished_reqs
        assert "p-req-active" not in finished_reqs
        assert "tx-expired" not in prefill_worker.reqs_need_send
        assert "tx-active" in prefill_worker.reqs_need_send


@pytest.mark.parametrize(
    ("layout", "separate_kv_head_groups"),
    [
        (KVCacheLayout.LBHNC, False),
        (KVCacheLayout.BLHNC, False),
        (KVCacheLayout.BHLNC, False),
        # LHBNC gives each head group its own region; the K/V split doubles the
        # head count but the registration shape is driven by the layout.
        (KVCacheLayout.LHBNC, False),
        (KVCacheLayout.LHBNC, True),
    ],
)
def test_register_kv_caches(layout: KVCacheLayout, separate_kv_head_groups: bool):
    """Tests the memory registration logic with the underlying Mooncake engine."""
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector", kv_role="kv_consumer"
    )

    with (
        set_current_vllm_config(vllm_config),
        patch_worker_dependencies(),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.threading.Event"
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.threading.Thread"
        ) as mock_thread,
    ):
        connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            _make_test_kv_cache_config(),
        )
        worker = connector.connector_worker
        mock_thread.return_value.is_alive.return_value = False

        spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=4,
            head_size=64,
            dtype=torch.float16,
            num_head_slots=2 if separate_kv_head_groups else None,
            state_content_bytes=4 * 64 * 2 if separate_kv_head_groups else None,
        )
        layer_names = [
            "model.layers.0.self_attn",
            "model.layers.1.self_attn",
        ]
        for layer_name in layer_names:
            worker._layer_specs[layer_name] = spec
        raw = torch.zeros(2 * 2 * spec.page_size_bytes, dtype=torch.int8)
        tensor1, tensor2 = dense_kv_cache_views(raw, spec, 2, 2, layout)
        kv_caches = dict(zip(layer_names, (tensor1, tensor2)))

        with patch.object(
            worker.engine, "batch_register_memory", return_value=0
        ) as mock_batch_register:
            connector.register_kv_caches(kv_caches)

            mock_batch_register.assert_called_once()
            registered_ptrs, registered_lens = mock_batch_register.call_args[0]
            assert registered_ptrs == [raw.data_ptr()]
            assert registered_lens == [raw.nbytes]

            tensor_blocks = tensor1.shape[0] if tensor1.ndim > 1 else 0
            block_stride = (
                tensor1.stride(0) * tensor1.element_size() if tensor1.ndim > 1 else 0
            )
            storage_nbytes = tensor1.untyped_storage().nbytes()
            storage_is_block_major = (
                tensor_blocks > 0 and tensor_blocks * block_stride == storage_nbytes
            )
            hnc_contiguous = (
                tensor1.ndim == 4
                and tensor1.stride(2) == tensor1.shape[3]
                and tensor1.stride(1) == tensor1.shape[2] * tensor1.shape[3]
            )
            # Block-major non-HNC FA. Contiguous per-layer pages stay separate
            # views. A non-contiguous row (BHLNC) is one allocation-anchored
            # region, and its last block must stay inside the registered buffer.
            if storage_is_block_major and not hnc_contiguous:
                packed_row = storage_nbytes // tensor_blocks
                storage_end = raw.data_ptr() + raw.nbytes
                if tensor1.ndim > 1 and tensor1[0].is_contiguous():
                    assert len(worker.registered_layer_names) == len(kv_caches)
                    for base, stride, kv_len in zip(
                        worker.kv_caches_base_addr,
                        worker.block_len_per_layer,
                        worker.kv_block_len_per_layer,
                    ):
                        assert stride == packed_row
                        assert 0 < kv_len <= packed_row
                        assert (
                            base + (tensor_blocks - 1) * stride + kv_len <= storage_end
                        )
                else:
                    assert worker.kv_caches_base_addr == [raw.data_ptr()]
                    assert worker.block_len_per_layer == [packed_row]
                    assert worker.kv_block_len_per_layer == [packed_row]
                    assert worker.registered_layer_names == [layer_names[0]]
                    assert raw.data_ptr() + tensor_blocks * packed_row <= storage_end
            elif not layout.is_block_compact:
                expected_addrs = [
                    cache[:, head_idx].data_ptr()
                    for cache in (tensor1, tensor2)
                    for head_idx in range(cache.shape[1])
                ]
                head_block_bytes = tensor1.stride(0) * tensor1.element_size()
                assert worker.kv_caches_base_addr == expected_addrs
                assert worker.block_len_per_layer == [head_block_bytes] * len(
                    expected_addrs
                )
                assert worker.kv_block_len_per_layer == [head_block_bytes] * len(
                    expected_addrs
                )
                assert worker.registered_layer_names == [
                    layer_name
                    for layer_name in layer_names
                    for _ in range(tensor1.shape[1])
                ]
            else:
                assert len(worker.block_len_per_layer) == len(kv_caches)
                for bl in worker.block_len_per_layer:
                    assert bl == tensor1.stride(0) * tensor1.element_size()
                assert worker.kv_block_len_per_layer == [spec.page_size_bytes] * 2
                assert worker.registered_layer_names == list(kv_caches)
                assert worker.registered_layer_indices == [0, 1]


def test_register_noncontiguous_packed_allows_matching_pp():
    """Same PP on both sides can still register a non-contiguous packed row.

    A remote PP mismatch is rejected at send time, once the peer PP size is
    known. Rejecting every local PP>1 layout at startup is too strict.
    """
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector", kv_role="kv_consumer"
    )
    with (
        set_current_vllm_config(vllm_config),
        patch_worker_dependencies(),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.threading.Event"
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.threading.Thread"
        ) as mock_thread,
    ):
        connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            _make_test_kv_cache_config(),
        )
        worker = connector.connector_worker
        worker.pp_size = 2
        mock_thread.return_value.is_alive.return_value = False
        spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=4,
            head_size=64,
            dtype=torch.float16,
        )
        layer_names = [
            "model.layers.0.self_attn",
            "model.layers.1.self_attn",
        ]
        for layer_name in layer_names:
            worker._layer_specs[layer_name] = spec
        raw = torch.zeros(2 * 2 * spec.page_size_bytes, dtype=torch.int8)
        tensor1, tensor2 = dense_kv_cache_views(raw, spec, 2, 2, KVCacheLayout.BHLNC)
        assert not tensor1[0].is_contiguous()
        with patch.object(worker.engine, "batch_register_memory", return_value=0):
            connector.register_kv_caches(dict(zip(layer_names, (tensor1, tensor2))))
        assert worker.kv_caches_base_addr == [raw.data_ptr()]
        assert worker.region_row_offsets == [0]


def test_register_kv_caches_supports_mixed_mla_and_eagle_shapes():
    """Mixed MLA+Eagle caches should register by byte length, not shape."""
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector", kv_role="kv_consumer"
    )

    with (
        set_current_vllm_config(vllm_config),
        patch_worker_dependencies(),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.threading.Event"
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.threading.Thread"
        ) as mock_thread,
    ):
        connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            _make_test_kv_cache_config(),
        )
        worker = connector.connector_worker
        mock_thread.return_value.is_alive.return_value = False

        worker.use_mla = True
        worker.transfer_topo.is_mla = True

        # MLA cache tensor: shape[-2] is the block size and each block's
        # byte stride matches the cache spec page size.
        mla_cache = torch.zeros((2, 16, 512), dtype=torch.float16)
        # Eagle3/GQA-like cache tensor: shape[-2] is num_kv_heads, not block size.
        eagle_cache = torch.zeros((2, 16, 8, 64), dtype=torch.float16)
        kv_caches = {
            "model.layers.0.mla_attn": mla_cache,
            "model.layers.1.eagle_attn": eagle_cache,
        }

        with patch.object(
            worker.engine, "batch_register_memory", return_value=0
        ) as mock_batch_register:
            connector.register_kv_caches(kv_caches)

        mock_batch_register.assert_called_once()
        registered_ptrs, registered_lens = mock_batch_register.call_args[0]
        assert registered_ptrs == [mla_cache.data_ptr(), eagle_cache.data_ptr()]
        assert registered_lens == [mla_cache.nbytes, eagle_cache.nbytes]
        assert worker.block_len_per_layer == [
            mla_cache.nbytes // mla_cache.shape[0],
            eagle_cache.nbytes // eagle_cache.shape[0],
        ]
        assert worker.registered_layer_names == [
            "model.layers.0.mla_attn",
            "model.layers.1.eagle_attn",
        ]
        assert worker.registered_layer_indices == [0, 1]
        assert len(worker.kv_caches_base_addr) == 2


def test_packed_and_unpacked_region_lengths_fail_homogeneous_tp_handshake():
    packed = TransferRegion(
        layer_name="model.layers.0.mla_attn",
        layer_index=0,
        base_addr=0x1000,
        block_len=2048,
        kv_block_len=2048,
        group_index=0,
    )
    unpacked = TransferRegion(
        layer_name="model.layers.0.mla_attn",
        layer_index=0,
        base_addr=0x1000,
        block_len=64,
        kv_block_len=64,
        group_index=0,
    )
    err = _validate_asymmetric_region_lengths(
        local_regions=[packed],
        remote_regions=[unpacked],
        local_tp_size=1,
        remote_tp_size=1,
        producer_cache_replicated=False,
    )
    assert err is not None
    assert "length mismatch" in err

    count_err = _validate_asymmetric_region_lengths(
        local_regions=[packed],
        remote_regions=[unpacked, unpacked],
        local_tp_size=1,
        remote_tp_size=1,
        producer_cache_replicated=False,
    )
    assert count_err is not None
    assert "region counts" in count_err


@pytest.mark.parametrize(
    ("num_groups", "expected_group_index"),
    [(1, 0), (2, _SHARED_REGION_GROUP_ID)],
)
def test_register_kv_caches_collapses_shared_mla_storage(
    num_groups, expected_group_index
):
    """Shared backing collapses to one packed region; two groups mark it -1."""
    num_blocks = 4
    packed_row = 256
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector", kv_role="kv_consumer"
    )

    with (
        set_current_vllm_config(vllm_config),
        patch_worker_dependencies(),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.threading.Event"
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.threading.Thread"
        ) as mock_thread,
    ):
        connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            _make_packed_mla_kv_cache_config(num_blocks, num_groups=num_groups),
        )
        worker = connector.connector_worker
        mock_thread.return_value.is_alive.return_value = False

        backing = torch.zeros((num_blocks, packed_row), dtype=torch.uint8)
        kv_caches = {
            "model.layers.0.mla_attn": backing,
            "model.layers.1.indexer": backing,
        }
        with patch.object(
            worker.engine, "batch_register_memory", return_value=0
        ) as mock_batch_register:
            connector.register_kv_caches(kv_caches)

        mock_batch_register.assert_called_once()
        registered_ptrs, registered_lens = mock_batch_register.call_args[0]
        assert registered_ptrs == [backing.data_ptr()]
        assert registered_lens == [backing.untyped_storage().nbytes()]
        assert worker.kv_caches_base_addr == [backing.data_ptr()]
        assert worker.block_len_per_layer == [packed_row]
        assert worker.kv_block_len_per_layer == [packed_row]
        assert worker.registered_layer_names == ["model.layers.0.mla_attn"]
        assert worker.registered_group_indices == [expected_group_index]
        if expected_group_index == _SHARED_REGION_GROUP_ID:
            assert worker.region_shared_groups == [(0, 1)]


def test_block_ids_for_region_flattens_shared_groups():
    ids = [[10, 11], [12]]
    assert _block_ids_for_region(ids, 0) == [10, 11]
    assert _block_ids_for_region(ids, 1) == [12]
    assert _block_ids_for_region(ids, _SHARED_REGION_GROUP_ID, (0, 1)) == [
        10,
        11,
        12,
    ]
    # A group that does not share the allocation stays out.
    assert _block_ids_for_region(
        [[10, 11], [12], [99]], _SHARED_REGION_GROUP_ID, (0, 1)
    ) == [10, 11, 12]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("group_index", "local_ids", "remote_ids", "n_blocks"),
    [
        (0, [[10, 11]], [[20, 21]], 2),
        (_SHARED_REGION_GROUP_ID, [[10, 11], [12]], [[20, 21], [22]], 3),
    ],
)
async def test_build_transfer_params_sends_packed_region_once(
    group_index, local_ids, remote_ids, n_blocks
):
    """Packed region coalesces contiguous blocks, including shared-group flatten."""
    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker.async_zmq_ctx = MagicMock()
    worker.is_kv_consumer = True
    worker.is_kv_producer = True
    worker.tp_rank = 0
    worker.tp_size = 1
    worker.use_mla = True
    worker.kv_cache_config = _make_packed_mla_kv_cache_config(
        num_blocks=4, num_groups=len(local_ids)
    )
    worker._physical_blocks_per_logical_kv_block = 1
    worker.transfer_topo = SimpleNamespace(
        local_replicates_kv_cache=False,
        total_num_kv_heads=1,
    )

    block_len = 256
    region_kw = dict(
        layer_name="model.layers.0.mla_attn",
        layer_index=0,
        block_len=block_len,
        kv_block_len=block_len,
        group_index=group_index,
        shared_group_ids=(0, 1) if group_index == _SHARED_REGION_GROUP_ID else (),
    )
    transfer_id = "xfer-packed"
    send_meta = SendBlockMeta(
        p_req_id="p-packed",
        transfer_id=transfer_id,
        local_block_ids=local_ids,
        ready=asyncio.Event(),
    )
    xfer_meta = MooncakeXferMetadata(
        remote_hostname="consumer-host",
        remote_port=54321,
        remote_tp_size=1,
        remote_tp_rank=0,
        req_blocks={"d-packed": (transfer_id, remote_ids)},
        kv_caches_base_addr=[0xA000],
        block_lens=[block_len],
        kv_block_lens=[block_len],
        registered_layer_names=["model.layers.0.mla_attn"],
        registered_layer_indices=[0],
        registered_group_indices=[group_index],
    )

    (
        src_ptrs,
        dst_ptrs,
        lengths,
        err_reqs,
        err_msg,
    ) = await worker._build_transfer_params(
        ready_reqs=[("d-packed", send_meta)],
        agent_meta=xfer_meta,
        local_regions=[TransferRegion(base_addr=0x1000, **region_kw)],
        remote_regions=[TransferRegion(base_addr=0xA000, **region_kw)],
    )

    assert err_reqs == []
    assert err_msg is None
    assert src_ptrs == [0x1000 + 10 * block_len]
    assert dst_ptrs == [0xA000 + 20 * block_len]
    assert lengths == [n_blocks * block_len]


def test_coalesce_promotes_padding_only_for_a_full_row():
    """A full-row merge includes the padding tail; a partial row does not."""
    page = 1024
    row = 2560

    def region(base: int, row_offset: int) -> TransferRegion:
        return TransferRegion(
            layer_name="layer",
            layer_index=0,
            base_addr=base,
            block_len=row,
            kv_block_len=page,
            row_offset=row_offset,
        )

    full_local, full_remote = _coalesce_contiguous_transfer_regions(
        [region(0, 0), region(page, page)],
        [region(100, 0), region(100 + page, page)],
        promote_full_row=True,
    )
    assert len(full_local) == 1
    assert full_local[0].kv_block_len == row
    assert full_remote[0].kv_block_len == row

    partial_local, _ = _coalesce_contiguous_transfer_regions(
        [region(0, 0), region(page, page)],
        [region(100, 0), region(100 + page, page)],
    )
    assert partial_local[0].kv_block_len == 2 * page


def test_coalesce_spans_padding_between_pages():
    """Unpadded payloads with a padded-page gap still merge inside the row."""
    row = 2560

    def region(base: int) -> TransferRegion:
        return TransferRegion(
            layer_name="layer",
            layer_index=0,
            base_addr=base,
            block_len=row,
            kv_block_len=1000,
        )

    merged, _ = _coalesce_contiguous_transfer_regions(
        [region(0), region(1024)],
        [region(5000), region(6024)],
        promote_full_row=True,
    )
    assert len(merged) == 1
    assert merged[0].kv_block_len == row


def test_coalesce_promotes_each_group_that_starts_at_row_zero():
    """Groups overlay one row, so each row-start run is its own full-row copy."""
    page = 1000
    gap = 1024
    row = 4096

    def region(base: int, group: int, row_offset: int) -> TransferRegion:
        return TransferRegion(
            layer_name=f"group-{group}",
            layer_index=group,
            base_addr=base,
            block_len=row,
            kv_block_len=page,
            group_index=group,
            row_offset=row_offset,
        )

    # Group 1's addresses jump back to the start of the same row.
    local = [
        region(0, 0, 0),
        region(gap, 0, gap),
        region(0, 1, 0),
        region(gap, 1, gap),
    ]
    remote = [
        region(8000, 0, 0),
        region(8000 + gap, 0, gap),
        region(8000, 1, 0),
        region(8000 + gap, 1, gap),
    ]
    merged_local, merged_remote = _coalesce_contiguous_transfer_regions(
        local, remote, promote_full_row=True
    )
    assert [region.group_index for region in merged_local] == [0, 1]
    assert [region.kv_block_len for region in merged_local] == [row, row]
    assert [region.kv_block_len for region in merged_remote] == [row, row]
    assert merged_local[1].base_addr == 0
    assert merged_remote[1].base_addr == 8000


def test_coalesce_does_not_promote_a_mid_row_run():
    """A matched PP slice that does not start at row offset 0 stays partial."""
    page = 1000
    row = 4096

    def region(base: int, row_offset: int) -> TransferRegion:
        return TransferRegion(
            layer_name="layer",
            layer_index=2,
            base_addr=base,
            block_len=row,
            kv_block_len=page,
            row_offset=row_offset,
        )

    merged, _ = _coalesce_contiguous_transfer_regions(
        [region(1024, 1024), region(2048, 2048)],
        [region(9000 + 1024, 1024), region(9000 + 2048, 2048)],
        promote_full_row=True,
    )
    assert len(merged) == 1
    assert merged[0].row_offset == 1024
    assert merged[0].kv_block_len == 1024 + page


def test_opaque_packed_row_is_full_stride_at_offset_zero():
    """Branch B is row-sized at offset 0. A page view at offset 0 is not."""
    assert _has_opaque_packed_row([0], [4096], [4096])
    assert not _has_opaque_packed_row([0, 1024], [1000, 1000], [4096, 4096])
    assert not _has_opaque_packed_row([], [4096], [4096])


def test_coalesce_keeps_a_run_inside_its_starting_row():
    """The row bound is the run start, not the previous slice."""
    row = 2500

    def region(base: int) -> TransferRegion:
        return TransferRegion(
            layer_name="layer",
            layer_index=0,
            base_addr=base,
            block_len=row,
            kv_block_len=1000,
            row_offset=base,
        )

    # The third slice ends at 3000. Checking it against the previous slice
    # (base 1000) would allow it; the run start at 0 does not.
    slices = [region(0), region(1000), region(2000)]
    merged, _ = _coalesce_contiguous_transfer_regions(slices, slices)
    assert len(merged) == 2
    assert merged[0].kv_block_len == 2000
    assert merged[1].base_addr == 2000
    assert merged[1].kv_block_len == 1000


def _layer_name(idx: int) -> str:
    return f"model.layers.{idx}.mla_attn"


def _register_sliced_packed_mla(
    layer_idxs: list[int], *, page: int = 1024, num_blocks: int = 4
):
    """Register column slices of one MLA packed row. Returns worker fields."""
    spec = MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=64,
        dtype=torch.uint8,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec([_layer_name(i) for i in layer_idxs], spec),
        ],
    )
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector", kv_role="kv_producer"
    )
    backing = torch.zeros((num_blocks, page * len(layer_idxs)), dtype=torch.uint8)
    kv_caches = {
        _layer_name(idx): backing[:, i * page : (i + 1) * page]
        for i, idx in enumerate(layer_idxs)
    }
    with (
        set_current_vllm_config(vllm_config),
        patch_worker_dependencies(),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.threading.Event"
        ),
        patch(
            "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.threading.Thread"
        ) as mock_thread,
    ):
        connector = MooncakeConnector(
            vllm_config, KVConnectorRole.WORKER, kv_cache_config
        )
        worker = connector.connector_worker
        mock_thread.return_value.is_alive.return_value = False
        with patch.object(
            worker.engine, "batch_register_memory", return_value=0
        ) as reg:
            connector.register_kv_caches(kv_caches)
        reg.assert_called_once()
        assert reg.call_args[0][0] == [backing.data_ptr()]
        assert worker.region_row_offsets == [i * page for i in range(len(layer_idxs))]
        regions = worker._get_transfer_regions(
            worker.kv_caches_base_addr,
            worker.block_len_per_layer,
            worker.kv_block_len_per_layer,
            worker.registered_layer_names,
            worker.registered_layer_indices,
            worker.registered_group_indices,
            row_offsets=worker.region_row_offsets,
        )
        return worker, kv_cache_config, regions, backing.data_ptr()


@pytest.mark.asyncio
async def test_packed_hetero_pp_keeps_shared_layers():
    """P-PP4/D-PP2 must align the shared MLA slices, not drop them silently."""
    page = 1024
    p_worker, p_config, p_regions, p_base = _register_sliced_packed_mla(
        [2, 3], page=page
    )
    _, _, d_regions, d_base = _register_sliced_packed_mla([0, 1, 2, 3], page=page)

    assert [r.layer_index for r in p_regions] == [2, 3]
    assert [r.layer_index for r in d_regions] == [0, 1, 2, 3]
    assert [r.kv_block_len for r in p_regions] == [page, page]
    assert p_regions[0].block_len == 2 * page
    assert d_regions[0].block_len == 4 * page

    aligned_p, aligned_d, err = _align_transfer_regions(
        p_regions, d_regions, allow_partial_layers=True
    )
    assert err is None
    assert [r.layer_index for r in aligned_p] == [2, 3]
    coalesced_p, coalesced_d = _coalesce_contiguous_transfer_regions(
        aligned_p, aligned_d
    )
    assert len(coalesced_p) == 1
    assert coalesced_p[0].kv_block_len == 2 * page
    assert coalesced_d[0].kv_block_len == 2 * page
    assert coalesced_d[0].base_addr == d_base + 2 * page

    # D's row is wider than the matched span, so blocks stay separate copies.
    block_ids = [[10, 11]]
    transfer_id = "xfer-pp"
    send_meta = SendBlockMeta(
        p_req_id="p-pp",
        transfer_id=transfer_id,
        local_block_ids=block_ids,
        ready=asyncio.Event(),
    )
    xfer_meta = MooncakeXferMetadata(
        remote_hostname="consumer-host",
        remote_port=54321,
        remote_tp_size=1,
        remote_tp_rank=0,
        req_blocks={"d-pp": (transfer_id, block_ids)},
        kv_caches_base_addr=[r.base_addr for r in coalesced_d],
        block_lens=[r.block_len for r in coalesced_d],
        kv_block_lens=[r.kv_block_len for r in coalesced_d],
        registered_layer_names=[r.layer_name for r in coalesced_d],
        registered_layer_indices=[r.layer_index for r in coalesced_d],
        registered_group_indices=[r.group_index for r in coalesced_d],
    )
    p_worker.kv_cache_config = p_config
    src, dst, lengths, err_reqs, err_msg = await p_worker._build_transfer_params(
        ready_reqs=[("d-pp", send_meta)],
        agent_meta=xfer_meta,
        local_regions=coalesced_p,
        remote_regions=coalesced_d,
    )
    assert err_reqs == []
    assert err_msg is None
    assert lengths == [2 * page, 2 * page]
    assert src == [p_base + 10 * (2 * page), p_base + 11 * (2 * page)]
    assert dst == [
        d_base + 2 * page + 10 * (4 * page),
        d_base + 2 * page + 11 * (4 * page),
    ]

    # No shared layers: hetero PP still no-ops without an error.
    _, _, early_d, _ = _register_sliced_packed_mla([0, 1], page=page)
    none_p, none_d, none_err = _align_transfer_regions(
        p_regions, early_d, allow_partial_layers=True
    )
    assert none_err is None
    assert none_p == [] and none_d == []

    # Same PP size must fail closed when the layer sets differ.
    _, _, mismatch_err = _align_transfer_regions(
        p_regions, early_d, allow_partial_layers=False
    )
    assert mismatch_err is not None


@pytest.mark.asyncio
async def test_packed_matching_layers_coalesce_to_one_copy():
    """Identical packed layer sets still emit one copy for contiguous blocks."""
    page = 1024
    worker, config, regions, base = _register_sliced_packed_mla([0, 1], page=page)
    aligned_l, aligned_r, err = _align_transfer_regions(regions, regions)
    assert err is None
    coalesced_l, coalesced_r = _coalesce_contiguous_transfer_regions(
        aligned_l, aligned_r
    )
    assert len(coalesced_l) == 1
    assert coalesced_l[0].kv_block_len == coalesced_l[0].block_len == 2 * page

    block_len = 2 * page
    transfer_id = "xfer-full"
    send_meta = SendBlockMeta(
        p_req_id="p-full",
        transfer_id=transfer_id,
        local_block_ids=[[10, 11]],
        ready=asyncio.Event(),
    )
    xfer_meta = MooncakeXferMetadata(
        remote_hostname="consumer-host",
        remote_port=54321,
        remote_tp_size=1,
        remote_tp_rank=0,
        req_blocks={"d-full": (transfer_id, [[20, 21]])},
        kv_caches_base_addr=[base],
        block_lens=[block_len],
        kv_block_lens=[block_len],
        registered_layer_names=[regions[0].layer_name],
        registered_layer_indices=[0],
        registered_group_indices=[0],
    )
    worker.kv_cache_config = config
    src, dst, lengths, err_reqs, err_msg = await worker._build_transfer_params(
        ready_reqs=[("d-full", send_meta)],
        agent_meta=xfer_meta,
        local_regions=coalesced_l,
        remote_regions=coalesced_r,
    )
    assert err_reqs == []
    assert err_msg is None
    assert src == [base + 10 * block_len]
    assert dst == [base + 20 * block_len]
    assert lengths == [2 * block_len]


@pytest.mark.asyncio
@patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
    "mooncake_connector.TransferEngine",
    FakeMooncakeWrapper,
)
@pytest.mark.parametrize("d_tp_size", [1, 4], ids=["p_tp2_d_tp1", "p_tp2_d_tp4"])
async def test_kv_producer_heterogeneous_tp(monkeypatch, d_tp_size):
    """Tests heterogeneous TP support in the producer transfer path.

    Verifies correct pointer and offset calculation when producer TP=2
    sends to consumer with TP=1 (P>D) or TP=4 (P<D).

    Parametrized cases:
    - P TP=2 > D TP=1: one D rank receives; dst_offset based on P rank
    - P TP=2 < D TP=4: two D ranks receive; src_offset based on D rank
    """
    P_TP_SIZE = 2
    P_TP_RANK = 0
    LOCAL_BLOCK_LEN = 4096

    local_block_len = LOCAL_BLOCK_LEN
    remote_block_len = LOCAL_BLOCK_LEN * P_TP_SIZE // d_tp_size

    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector", kv_role="kv_producer"
    )

    with set_current_vllm_config(vllm_config), patch_worker_dependencies():
        prefill_connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            _make_test_kv_cache_config(),
        )
        prefill_worker = prefill_connector.connector_worker

        # Override TP rank/size to simulate P TP=2
        prefill_worker.tp_rank = P_TP_RANK
        prefill_worker.tp_size = P_TP_SIZE
        prefill_worker._tp_size[prefill_worker.engine_id] = P_TP_SIZE
        prefill_worker.transfer_topo.tp_rank = P_TP_RANK
        prefill_worker.transfer_topo.tp_size = P_TP_SIZE

        prefill_worker.kv_caches_base_addr = [0x1000]
        prefill_worker.block_len_per_layer = [local_block_len]
        prefill_worker.kv_block_len_per_layer = [local_block_len]
        prefill_worker.registered_layer_names = ["model.layers.0.self_attn"]
        prefill_worker.registered_layer_indices = [0]

        origin_sender_loop = prefill_worker.sender_loop
        prefill_worker.sender_loop = asyncio.get_event_loop()

        transfer_id = "xfer-hetero-1"
        local_block_ids = [[10, 11]]
        send_meta = SendBlockMeta(
            p_req_id="p-req-h1",
            transfer_id=transfer_id,
            local_block_ids=local_block_ids,
            ready=asyncio.Event(),
        )
        prefill_worker.reqs_need_send[transfer_id] = send_meta
        send_meta.ready.set()

        # Compute target D ranks using the production code path
        target_d_ranks = prefill_worker.transfer_topo.handshake_target_ranks(d_tp_size)

        mock_socket = AsyncMock(spec=zmq.asyncio.Socket)
        mock_socket.send_multipart = AsyncMock()
        identity = b"consumer-hetero"

        # Assign different remote block IDs per D rank (nested per-group)
        d_rank_remote_blocks = {
            rank: [[20 + i * 10, 21 + i * 10]] for i, rank in enumerate(target_d_ranks)
        }

        with patch.object(
            prefill_worker, "_send_blocks", return_value=0
        ) as mock_send_blocks:
            for d_rank in target_d_ranks:
                remote_block_ids = d_rank_remote_blocks[d_rank]
                xfer_meta = MooncakeXferMetadata(
                    remote_hostname="consumer-host",
                    remote_port=54321,
                    remote_tp_size=d_tp_size,
                    remote_tp_rank=d_rank,
                    req_blocks={
                        f"d-req-h1-r{d_rank}": (
                            transfer_id,
                            remote_block_ids,
                        )
                    },
                    kv_caches_base_addr=[0x2000],
                    block_lens=[remote_block_len],
                    kv_block_lens=[remote_block_len],
                    registered_layer_names=["model.layers.0.self_attn"],
                    registered_layer_indices=[0],
                )

                mock_send_blocks.reset_mock()
                mock_socket.send_multipart.reset_mock()

                await prefill_worker.send_kv_to_decode(identity, mock_socket, xfer_meta)

                # Verify _send_blocks was called
                mock_send_blocks.assert_called_once()
                call_args = mock_send_blocks.call_args[0]
                src_ptrs = call_args[1]
                dst_ptrs = call_args[2]
                lengths = call_args[3]

                # Flatten nested per-group block IDs for assertions
                flat_local = [b for g in local_block_ids for b in g]
                flat_remote = [b for g in remote_block_ids for b in g]
                num_blocks = len(flat_local)

                assert len(src_ptrs) == num_blocks
                assert len(dst_ptrs) == num_blocks
                assert len(lengths) == num_blocks

                if d_tp_size <= P_TP_SIZE:
                    tp_ratio = P_TP_SIZE // d_tp_size
                    expected_src_off = 0
                    expected_dst_off = (P_TP_RANK % tp_ratio) * local_block_len
                    expected_xfer_len = local_block_len
                else:
                    ratio_abs = d_tp_size // P_TP_SIZE
                    expected_src_off = (d_rank % ratio_abs) * remote_block_len
                    expected_dst_off = 0
                    expected_xfer_len = remote_block_len

                local_region_base = 0x1000
                remote_region_base = 0x2000
                for blk_idx, (lblk, rblk) in enumerate(zip(flat_local, flat_remote)):
                    assert src_ptrs[blk_idx] == (
                        local_region_base + lblk * local_block_len + expected_src_off
                    )
                    assert dst_ptrs[blk_idx] == (
                        remote_region_base + rblk * remote_block_len + expected_dst_off
                    )
                    assert lengths[blk_idx] == expected_xfer_len

                # Verify successful response sent back to consumer
                mock_socket.send_multipart.assert_called_once()
                _, sent_payload = mock_socket.send_multipart.call_args[0][0]
                response = prefill_worker._xfer_resp_decoder.decode(sent_payload)
                assert response.status == MooncakeXferResponseStatus.FINISH
                assert response.ok_reqs == [f"d-req-h1-r{d_rank}"]

        # After serving all D ranks, the request should be complete
        assert transfer_id not in prefill_worker.reqs_need_send
        assert "p-req-h1" in prefill_worker.finished_sending_reqs

        prefill_worker.sender_loop = origin_sender_loop
        prefill_worker.shutdown()
