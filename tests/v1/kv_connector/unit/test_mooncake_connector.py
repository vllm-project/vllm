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

from vllm import envs
from vllm.config import set_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
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
    get_mooncake_bootstrap_addr,
    should_launch_bootstrap_server,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (
    MooncakeBootstrapServer,
)
from vllm.utils.network_utils import get_open_port
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.request import RequestStatus

from .utils import create_request, create_scheduler, create_vllm_config


def _make_test_kv_cache_config() -> KVCacheConfig:
    return KVCacheConfig(num_blocks=0, kv_cache_tensors=[], kv_cache_groups=[])


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
    worker.transfer_topo = SimpleNamespace(local_replicates_kv_cache=False)

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
        kv_half = block_len // 2
        prefill_worker.kv_caches_base_addr = [0x1000]
        prefill_worker.block_len_per_layer = [block_len]
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
        assert src_ptrs == [
            0x1000 + 10 * block_len,
            0x1000 + 10 * block_len + kv_half,
        ]
        assert dst_ptrs == [
            0xB000 + 20 * block_len,
            0xB000 + 20 * block_len + kv_half,
        ]
        assert lengths == [kv_half, kv_half]

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
    """
    Tests the bootstrap server's api for worker registration and querying.

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
    """
    Tests the scheduler-side logic when a request finishes.

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
@patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector.TransferEngine",
    FakeMooncakeWrapper,
)
async def test_kv_producer(monkeypatch):
    """
    Simulates a Producer Worker (Prefiller) receiving a transfer request
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
            registered_layer_names=["model.layers.0.self_attn"],
            registered_layer_indices=[0],
        )

        mock_socket = AsyncMock(spec=zmq.asyncio.Socket)
        mock_socket.send_multipart = AsyncMock()
        identity = b"consumer-id"

        with patch.object(
            prefill_worker, "_send_blocks", return_value=0
        ) as mock_send_blocks:
            # With blocks-first layout, each block is virtually split
            # into K and V halves, producing non-coalesced transfers.
            kv_half = block_len // 2

            def expected_split_transfers(src_base, dst_base, src_blocks, dst_blocks):
                """Build expected (src_ptrs, dst_ptrs, lengths) for
                virtual-split K/V transfers."""
                src_ptrs, dst_ptrs, lengths = [], [], []
                for kv_offset in (0, kv_half):
                    for sb, db in zip(src_blocks, dst_blocks):
                        src_ptrs.append(src_base + sb * block_len + kv_offset)
                        dst_ptrs.append(dst_base + db * block_len + kv_offset)
                        lengths.append(kv_half)
                return src_ptrs, dst_ptrs, lengths

            # Normal case: 2 blocks to 2 blocks
            await prefill_worker.send_kv_to_decode(identity, mock_socket, xfer_meta)
            src, dst, lens = expected_split_transfers(
                0x1000, 0x2000, [10, 11], [20, 21]
            )
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
            src, dst, lens = expected_split_transfers(0x1000, 0x2000, [11], [20])
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
    """
    Simulates a Consumer Worker (Decoder) initiating a pull from a Producer.

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


def test_register_kv_caches():
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

        kv_cache_shape = FlashAttentionBackend.get_kv_cache_shape(
            num_blocks=2, block_size=16, num_kv_heads=4, head_size=64
        )
        tensor1 = torch.zeros(*kv_cache_shape, dtype=torch.float16)
        tensor2 = torch.zeros(*kv_cache_shape, dtype=torch.float16)
        kv_caches = {
            "model.layers.0.self_attn": tensor1,
            "model.layers.1.self_attn": tensor2,
        }

        with patch.object(
            worker.engine, "batch_register_memory", return_value=0
        ) as mock_batch_register:
            connector.register_kv_caches(kv_caches)

            mock_batch_register.assert_called_once()
            registered_ptrs, registered_lens = mock_batch_register.call_args[0]
            expected_ptrs = {tensor.data_ptr() for tensor in kv_caches.values()}
            assert set(registered_ptrs) == expected_ptrs
            assert set(registered_lens) == {tensor1.nbytes}

            # Verify block_len_per_layer is set correctly.
            assert len(worker.block_len_per_layer) == len(registered_ptrs)
            for bl in worker.block_len_per_layer:
                assert bl == tensor1.nbytes // tensor1.shape[0]
            assert worker.registered_layer_names == list(kv_caches)
            assert worker.registered_layer_indices == [0, 1]


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

        # MLA cache tensor: shape[-2] is the block size.
        mla_cache = torch.zeros((2, 16, 96), dtype=torch.float16)
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


@pytest.mark.asyncio
@patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
    "mooncake_connector.TransferEngine",
    FakeMooncakeWrapper,
)
@pytest.mark.parametrize("d_tp_size", [1, 4], ids=["p_tp2_d_tp1", "p_tp2_d_tp4"])
async def test_kv_producer_heterogeneous_tp(monkeypatch, d_tp_size):
    """
    Tests heterogeneous TP support in the producer transfer path.

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

                # With blocks-first layout, virtual split halves block
                # lengths and doubles transfer regions (K + V).
                local_kv_block_len = local_block_len // 2
                remote_kv_block_len = remote_block_len // 2

                assert len(src_ptrs) == 2 * num_blocks
                assert len(dst_ptrs) == 2 * num_blocks
                assert len(lengths) == 2 * num_blocks

                # Compute expected offsets using kv_block_len
                if d_tp_size <= P_TP_SIZE:
                    tp_ratio = P_TP_SIZE // d_tp_size
                    expected_src_off = 0
                    expected_dst_off = (P_TP_RANK % tp_ratio) * local_kv_block_len
                    expected_xfer_len = local_kv_block_len
                else:
                    ratio_abs = d_tp_size // P_TP_SIZE
                    expected_src_off = (d_rank % ratio_abs) * remote_kv_block_len
                    expected_dst_off = 0
                    expected_xfer_len = remote_kv_block_len

                # First num_blocks entries are K region,
                # next num_blocks are V region.
                for region_idx in range(2):
                    local_region_base = 0x1000 + region_idx * local_kv_block_len
                    remote_region_base = 0x2000 + region_idx * remote_kv_block_len
                    for blk_idx, (lblk, rblk) in enumerate(
                        zip(flat_local, flat_remote)
                    ):
                        idx = region_idx * num_blocks + blk_idx
                        assert src_ptrs[idx] == (
                            local_region_base
                            + lblk * local_block_len
                            + expected_src_off
                        )
                        assert dst_ptrs[idx] == (
                            remote_region_base
                            + rblk * remote_block_len
                            + expected_dst_off
                        )
                        assert lengths[idx] == expected_xfer_len

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
