# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MooncakeConnector HMA (Hybrid Memory Architecture) support.

Covers sliding-window clipping, multi-group metadata shape, multi-group
send trimming, and group-count invariant checking in _build_transfer_params.
"""

import asyncio
import ctypes
from unittest.mock import patch

import pytest
import torch

from vllm.config import set_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    KVConnectorRole,
    MooncakeConnector,
    MooncakeConnectorMetadata,
    MooncakeConnectorScheduler,
    MooncakeXferMetadata,
    MooncakeXferResponse,
    MooncakeXferResponseStatus,
    PullReqMeta,
    SendBlockMeta,
    TransferRegion,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SparseCacheRole,
    UniformTypeKVCacheSpecs,
)

from .test_mooncake_connector import (
    FakeMooncakeWrapper,
    _xfer_meta,
    mooncake_register_worker,
    patch_worker_dependencies,
)
from .utils import (
    create_request,
    create_vllm_config,
    make_kv_cache_config,
    maybe_update_block_size,
)


# ---------------------------------------------------------------------------
#  test_sw_sizes: blocks_per_sw computed from KVCacheConfig
# ---------------------------------------------------------------------------
@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "swa_enabled,expected_blocks_per_sw",
    [
        # SWA enabled: FullAttentionSpec (0) + SlidingWindowSpec (2048/16=128+1)
        (True, [0, 128 + 1]),
        # SWA disabled: only FullAttentionSpec (0)
        (False, [0]),
    ],
)
def test_sw_sizes(swa_enabled, expected_blocks_per_sw):
    """blocks_per_sw is correctly computed based on SWA enabled/disabled."""
    block_size = 16
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector",
        kv_role="kv_both",
        block_size=block_size,
    )
    # Override so HMA detection works
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = False
    kv_cache_config = make_kv_cache_config(
        block_size=block_size, swa_enabled=swa_enabled, sw_size=2048
    )

    scheduler = MooncakeConnectorScheduler(
        vllm_config=vllm_config,
        engine_id="test-engine",
        kv_cache_config=kv_cache_config,
    )
    assert scheduler.blocks_per_sw == expected_blocks_per_sw


# ---------------------------------------------------------------------------
#  test_is_hma_required: derived from kv_cache_config groups
# ---------------------------------------------------------------------------
@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "swa_enabled,disable_hma,expected_is_hma",
    [
        (True, False, True),  # SWA group present, HMA enabled
        (True, True, False),  # SWA group present, but HMA disabled
        (False, False, False),  # FA only, HMA not needed
    ],
)
def test_is_hma_required(swa_enabled, disable_hma, expected_is_hma):
    """_is_hma_required is correctly derived from kv_cache_config."""
    block_size = 16
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector",
        kv_role="kv_both",
        block_size=block_size,
    )
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = disable_hma
    kv_cache_config = make_kv_cache_config(
        block_size=block_size, swa_enabled=swa_enabled
    )

    scheduler = MooncakeConnectorScheduler(
        vllm_config=vllm_config,
        engine_id="test-engine",
        kv_cache_config=kv_cache_config,
    )
    assert scheduler._is_hma_required is expected_is_hma


# ---------------------------------------------------------------------------
#  test_get_sw_clipped_blocks: sliding-window clipping logic
# ---------------------------------------------------------------------------
@pytest.mark.cpu_test
def test_get_sw_clipped_blocks():
    """get_sw_clipped_blocks clips SWA group but keeps FA group intact."""
    block_size = 16
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector",
        kv_role="kv_both",
        block_size=block_size,
    )
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = False
    # SW=128 tokens → 128/16 = 8 blocks + 1 = 9 blocks_per_sw
    kv_cache_config = make_kv_cache_config(
        block_size=block_size, swa_enabled=True, sw_size=128
    )

    scheduler = MooncakeConnectorScheduler(
        vllm_config=vllm_config,
        engine_id="test-engine",
        kv_cache_config=kv_cache_config,
    )
    assert scheduler.blocks_per_sw == [0, 9]

    # FA group: 20 blocks, SW group: 20 blocks (exceeds window)
    fa_blocks = list(range(20))
    sw_blocks = list(range(100, 120))
    block_ids = (fa_blocks, sw_blocks)

    clipped = scheduler.get_sw_clipped_blocks(block_ids)

    # FA: untouched (blocks_per_sw[0] = 0)
    assert clipped[0] == fa_blocks
    # SW: clipped to last 9 blocks
    assert clipped[1] == sw_blocks[-9:]
    assert len(clipped[1]) == 9


@pytest.mark.cpu_test
def test_get_sw_clipped_blocks_noop_no_hma():
    """get_sw_clipped_blocks is a no-op when HMA is not required."""
    block_size = 16
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector",
        kv_role="kv_both",
        block_size=block_size,
    )
    # FA only → _is_hma_required = False
    kv_cache_config = make_kv_cache_config(block_size=block_size, swa_enabled=False)

    scheduler = MooncakeConnectorScheduler(
        vllm_config=vllm_config,
        engine_id="test-engine",
        kv_cache_config=kv_cache_config,
    )
    assert scheduler._is_hma_required is False

    block_ids = ([1, 2, 3],)
    clipped = scheduler.get_sw_clipped_blocks(block_ids)
    assert clipped == [[1, 2, 3]]


# ---------------------------------------------------------------------------
#  test_metadata_hma_block_ids: MooncakeConnectorMetadata stores per-group IDs
# ---------------------------------------------------------------------------
@pytest.mark.cpu_test
def test_metadata_hma_block_ids():
    """MooncakeConnectorMetadata.add_new_req stores per-group block IDs."""
    metadata = MooncakeConnectorMetadata()

    # FA group: 6 blocks, SW group: 3 blocks (clipped)
    fa_blocks = [0, 1, 2, 3, 4, 5]
    sw_blocks = [10, 11, 12]

    # Test recv path
    metadata.add_new_req(
        request_id="recv-req",
        local_block_ids=[fa_blocks, sw_blocks],
        kv_transfer_params={
            "transfer_id": "recv-req",
            "remote_engine_id": "remote-engine",
            "remote_bootstrap_addr": "http://bootstrap:33333",
        },
        load_remote_cache=True,
    )

    assert "recv-req" in metadata.reqs_to_recv["remote-engine"]
    req_meta = metadata.reqs_to_recv["remote-engine"]["recv-req"]
    assert len(req_meta.local_block_ids) == 2
    assert req_meta.local_block_ids[0] == fa_blocks
    assert req_meta.local_block_ids[1] == sw_blocks

    # Test send path
    metadata.add_new_req(
        request_id="send-req",
        local_block_ids=[fa_blocks, sw_blocks],
        kv_transfer_params={
            "transfer_id": "send-req",
        },
        load_remote_cache=False,
    )

    assert "send-req" in metadata.reqs_to_send
    transfer_id, stored_blocks = metadata.reqs_to_send["send-req"]
    assert transfer_id == "send-req"
    assert len(stored_blocks) == 2
    assert stored_blocks[0] == fa_blocks
    assert stored_blocks[1] == sw_blocks


# ---------------------------------------------------------------------------
#  test_build_transfer_params_multi_group_trimming
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
@patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.mooncake"
    ".mooncake_connector.TransferEngine",
    FakeMooncakeWrapper,
)
async def test_build_transfer_params_multi_group_trimming(monkeypatch):
    """_build_transfer_params trims per-group blocks when local > remote."""
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
    block_size = maybe_update_block_size(16)
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector", kv_role="kv_producer", block_size=block_size
    )
    kv_cache_config = make_kv_cache_config(block_size=block_size, swa_enabled=True)

    with set_current_vllm_config(vllm_config), patch_worker_dependencies():
        connector = MooncakeConnector(
            vllm_config, KVConnectorRole.WORKER, kv_cache_config
        )
        worker = connector.connector_worker
        assert worker is not None

        block_len = 4096
        # Call _build_transfer_params directly (avoids send_kv_to_decode
        # async event loop complexity).
        transfer_id = "xfer-hma-trim"
        send_meta = SendBlockMeta(
            p_req_id="p-trim",
            transfer_id=transfer_id,
            # FA: 4 blocks, SW: 3 blocks (producer has more)
            local_block_ids=[[10, 11, 12, 13], [20, 21, 22]],
            ready=asyncio.Event(),
        )

        xfer_meta = MooncakeXferMetadata(
            remote_hostname="consumer-host",
            remote_port=54321,
            remote_tp_size=1,
            remote_tp_rank=0,
            req_blocks={
                "d-trim": (
                    transfer_id,
                    # FA: 2 blocks, SW: 2 blocks (consumer needs fewer)
                    [[30, 31], [40, 41]],
                )
            },
            kv_caches_base_addr=[0x2000],
            block_lens=[block_len],
            kv_block_lens=[block_len],
        )

        local_regions = [
            TransferRegion(
                layer_name="model.layers.0.self_attn",
                layer_index=0,
                base_addr=0x1000,
                block_len=block_len,
                kv_block_len=block_len,
            ),
        ]
        remote_regions = [
            TransferRegion(
                layer_name="model.layers.0.self_attn",
                layer_index=0,
                base_addr=0x2000,
                block_len=block_len,
                kv_block_len=block_len,
            ),
        ]

        ready_reqs = [("d-trim", send_meta)]
        (
            src_ptrs,
            dst_ptrs,
            lengths,
            err_reqs,
            err_msg,
        ) = await worker._build_transfer_params(
            ready_reqs, xfer_meta, local_regions, remote_regions
        )

        # No errors
        assert err_reqs == []
        assert err_msg is None
        # After trimming: FA [10..13] → last 2 → [12,13]; SW [20..22] → last 2 → [21,22]
        # Flattened: [12,13,21,22] = 4 blocks → coalesced into some transfers
        assert len(src_ptrs) > 0
        assert len(dst_ptrs) == len(src_ptrs)
        assert len(lengths) == len(src_ptrs)

        worker.shutdown()


# ---------------------------------------------------------------------------
#  test_build_transfer_params_group_count_mismatch
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
@patch(
    "vllm.distributed.kv_transfer.kv_connector.v1.mooncake"
    ".mooncake_connector.TransferEngine",
    FakeMooncakeWrapper,
)
async def test_build_transfer_params_group_count_mismatch(monkeypatch):
    """_build_transfer_params reports an error when group counts differ."""
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
    block_size = maybe_update_block_size(16)
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector", kv_role="kv_producer", block_size=block_size
    )
    kv_cache_config = make_kv_cache_config(block_size=block_size, swa_enabled=True)

    with set_current_vllm_config(vllm_config), patch_worker_dependencies():
        connector = MooncakeConnector(
            vllm_config, KVConnectorRole.WORKER, kv_cache_config
        )
        worker = connector.connector_worker
        assert worker is not None

        block_len = 4096
        transfer_id = "xfer-mismatch"
        send_meta = SendBlockMeta(
            p_req_id="p-mismatch",
            transfer_id=transfer_id,
            # Producer has 2 groups
            local_block_ids=[[10, 11], [20, 21]],
            ready=asyncio.Event(),
        )

        # Consumer has only 1 group — group count mismatch
        xfer_meta = MooncakeXferMetadata(
            remote_hostname="consumer-host",
            remote_port=54321,
            remote_tp_size=1,
            remote_tp_rank=0,
            req_blocks={
                "d-mismatch": (transfer_id, [[30, 31]]),
            },
            kv_caches_base_addr=[0x2000],
            block_lens=[block_len],
            kv_block_lens=[block_len],
        )

        local_regions = [
            TransferRegion(
                layer_name="model.layers.0.self_attn",
                layer_index=0,
                base_addr=0x1000,
                block_len=block_len,
                kv_block_len=block_len,
            ),
        ]
        remote_regions = [
            TransferRegion(
                layer_name="model.layers.0.self_attn",
                layer_index=0,
                base_addr=0x2000,
                block_len=block_len,
                kv_block_len=block_len,
            ),
        ]

        ready_reqs = [("d-mismatch", send_meta)]
        (
            src_ptrs,
            dst_ptrs,
            lengths,
            err_reqs,
            err_msg,
        ) = await worker._build_transfer_params(
            ready_reqs, xfer_meta, local_regions, remote_regions
        )

        # Mismatched req is reported via err_reqs/err_msg with no transfers built.
        assert err_reqs == ["d-mismatch"]
        assert err_msg == "KV group count mismatch"
        assert src_ptrs == []
        assert dst_ptrs == []
        assert lengths == []

        worker.shutdown()


# ---------------------------------------------------------------------------
#  test_request_finished_with_hma_groups
# ---------------------------------------------------------------------------
@pytest.mark.cpu_test
def test_request_finished_with_hma_groups():
    """request_finished correctly handles per-group block_ids."""
    block_size = 16
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector",
        kv_role="kv_producer",
        block_size=block_size,
    )
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = False
    kv_cache_config = make_kv_cache_config(
        block_size=block_size, swa_enabled=True, sw_size=128
    )

    scheduler = MooncakeConnectorScheduler(
        vllm_config=vllm_config,
        engine_id="test-engine",
        kv_cache_config=kv_cache_config,
    )

    request = create_request(request_id=1, do_remote_decode=True)
    assert request.kv_transfer_params is not None
    request.kv_transfer_params["transfer_id"] = request.request_id

    from vllm.v1.request import RequestStatus

    request.status = RequestStatus.FINISHED_LENGTH_CAPPED

    # 2 groups: FA with 10 blocks, SW with 20 blocks (will be clipped)
    fa_blocks = list(range(10))
    sw_blocks = list(range(100, 120))
    block_ids = (fa_blocks, sw_blocks)

    delay_free, _ = scheduler.request_finished(request, block_ids)
    assert delay_free is True
    assert request.request_id in scheduler._reqs_need_send

    _, stored_blocks = scheduler._reqs_need_send[request.request_id]
    # FA: untouched
    assert stored_blocks[0] == fa_blocks
    # SW: clipped to last 9 blocks (sw_size=128, block_size=16 → 8+1=9)
    assert stored_blocks[1] == sw_blocks[-9:]


# ---------------------------------------------------------------------------
#  Worker-side load-failure reporting (HMA vs non-HMA)
# ---------------------------------------------------------------------------
def _make_kv_consumer_worker(
    swa_enabled: bool = False, kv_cache_config: KVCacheConfig | None = None
):
    block_size = maybe_update_block_size(16)
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector",
        kv_role="kv_consumer",
        block_size=block_size,
    )
    if kv_cache_config is None:
        kv_cache_config = make_kv_cache_config(
            block_size=block_size, swa_enabled=swa_enabled
        )
    with set_current_vllm_config(vllm_config), patch_worker_dependencies():
        connector = MooncakeConnector(
            vllm_config, KVConnectorRole.WORKER, kv_cache_config
        )
    return connector.connector_worker


def _make_pull_meta(d_req_id: str, local_block_ids: list[list[int]]) -> PullReqMeta:
    return PullReqMeta(
        d_req_id=d_req_id,
        transfer_id="tx-1",
        local_block_ids=local_block_ids,
        remote_engine_id="engine-0",
        remote_bootstrap_addr="127.0.0.1:1234",
    )


# The tests below construct a MooncakeConnectorWorker, whose __init__
# requires an accelerator device, so (like the worker tests in
# test_mooncake_connector.py) they run on the GPU KV-connectors job only.
@pytest.mark.parametrize(
    "swa_enabled,expected_is_hma",
    [
        (True, True),  # multiple groups (FA + SWA)
        (False, False),  # single group, block-level reporting is unambiguous
    ],
)
def test_worker_is_hma_required(swa_enabled, expected_is_hma):
    """Worker-side _is_hma_required mirrors the scheduler's condition."""
    worker = _make_kv_consumer_worker(swa_enabled=swa_enabled)
    assert worker._is_hma_required is expected_is_hma


def test_worker_is_hma_required_multiple_full_attention_groups():
    """Two full-attention groups with different block sizes also select
    request-level failure reporting, matching the scheduler."""
    block_size = maybe_update_block_size(16)
    kv_cache_config = KVCacheConfig(
        num_blocks=100,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer0"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=4,
                    head_size=16,
                    dtype=torch.float16,
                ),
            ),
            KVCacheGroupSpec(
                ["layer1"],
                FullAttentionSpec(
                    block_size=2 * block_size,
                    num_kv_heads=4,
                    head_size=16,
                    dtype=torch.float16,
                ),
            ),
        ],
    )
    worker = _make_kv_consumer_worker(kv_cache_config=kv_cache_config)
    assert worker._is_hma_required


def test_worker_failed_recv_reports_request_level_failure_with_hma():
    """With HMA, load failures report the request, not ambiguous block IDs."""
    worker = _make_kv_consumer_worker(swa_enabled=True)
    assert worker._is_hma_required

    pull_meta = _make_pull_meta("d-req-1", [[1, 2], [3, 4]])
    worker._handle_failed_recv({"p-req-1": pull_meta}, ["p-req-1"], "boom")

    assert worker.get_block_ids_with_load_errors() == set()
    results = worker.get_transfer_results()
    assert results.failed_recving == {"d-req-1"}
    assert results.finished_recving == {"d-req-1"}


def test_worker_failed_recv_reports_block_ids_without_hma():
    """Without HMA, load failures keep reporting block-level errors."""
    worker = _make_kv_consumer_worker(swa_enabled=False)
    assert not worker._is_hma_required

    pull_meta = _make_pull_meta("d-req-1", [[1, 2], [3, 4]])
    worker._handle_failed_recv({"p-req-1": pull_meta}, ["p-req-1"], "boom")

    assert worker.get_block_ids_with_load_errors() == {1, 2, 3, 4}
    results = worker.get_transfer_results()
    assert results.failed_recving == set()
    assert results.finished_recving == {"d-req-1"}


@pytest.mark.parametrize("swa_enabled", [False, True])
@pytest.mark.parametrize("outcomes", [(False, True), (True, False), (False, False)])
def test_failed_receive_waits_for_every_peer(swa_enabled, outcomes):
    """A failed request keeps its destinations until the last peer completes."""
    worker = _make_kv_consumer_worker(swa_enabled=swa_enabled)
    meta = _make_pull_meta("d-req-1", [[1, 2], [3, 4]])
    meta.pull_tasks_count = 2
    error_reports = 0
    try:
        for index, successful in enumerate(outcomes):
            response = MooncakeXferResponse(
                status=MooncakeXferResponseStatus.FINISH,
                ok_reqs=["d-req-1"] if successful else None,
                err_reqs=None if successful else ["d-req-1"],
                err_msg=None if successful else "peer failed",
            )
            worker.process_pulling_result(response, {"d-req-1": meta})
            result = worker.get_transfer_results()
            invalid = worker.get_block_ids_with_load_errors()
            error_reports += bool(result.failed_recving or invalid)
            assert meta.pull_tasks_count == 1 - index
            assert result.finished_recving == ({"d-req-1"} if index else set())
            assert meta.failed == (not all(outcomes[: index + 1]))
            if result.failed_recving:
                assert result.failed_recving == {"d-req-1"}
            if invalid:
                assert invalid == {1, 2, 3, 4}
        assert error_reports == 1
        assert worker.get_transfer_results().finished_recving == set()
    finally:
        worker.shutdown()


def _hisparse_producer_cache_config(*, packed=False):
    specs = {
        name: MLAAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=16,
            dtype=torch.float16,
            cache_role=role,
        )
        for name, role in [
            ("model.layers.0.source", SparseCacheRole.SPARSE),
            ("model.layers.0.indexer", SparseCacheRole.INDEXER),
        ]
    }
    groups = [KVCacheGroupSpec([name], spec) for name, spec in specs.items()]
    if packed:
        groups = [
            KVCacheGroupSpec(
                list(specs),
                UniformTypeKVCacheSpecs(block_size=16, kv_cache_specs=specs),
            )
        ]
    return KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "packed,source_group,descending",
    [(True, 0, False), (True, 1, True), (False, 0, False), (False, 1, True)],
)
def test_hisparse_group_remap_copies_only_requested_pages(
    packed, source_group, descending
):
    """One P group feeds two D groups without copying cached pages or padding."""
    page = 512
    if packed:
        backing = torch.zeros((10, 2 * page), dtype=torch.uint8)
        source, indexer = backing[:, :page], backing[:, page:]
    else:
        source = torch.zeros((10, page), dtype=torch.uint8)
        indexer = torch.zeros_like(source)
    source_ids = [4, 3, 2, 1] if descending else [1, 2, 3, 4]
    for position, block_id in enumerate(source_ids):
        source[block_id].fill_(10 + position)
        indexer[block_id].fill_(100 + position)
    original_source, original_indexer = source.clone(), indexer.clone()

    host = torch.full((10, page), 0xCC, dtype=torch.uint8)
    gpu = torch.full((10, page + 64), 0xCC, dtype=torch.uint8)
    host_ids = [6, 5] if descending else [5, 6]
    indexer_ids = [8, 2, 7]
    expected_host, expected_gpu = host.clone(), gpu.clone()
    for position, block_id in zip([2, 3], host_ids):
        expected_host[block_id].fill_(10 + position)
    for position, block_id in zip([0, 2, 3], indexer_ids):
        expected_gpu[block_id, 32 : 32 + page].fill_(100 + position)

    remote = [
        TransferRegion(
            "model.layers.0.source", 0, host.data_ptr(), page, page, source_group
        ),
        TransferRegion(
            "model.layers.0.indexer",
            0,
            gpu.data_ptr() + 32,
            page + 64,
            page,
            1 - source_group,
        ),
    ]
    destination_ids = [host_ids, indexer_ids]
    positions = [[2, 3], [0, 2, 3]]
    if source_group == 1:
        destination_ids.reverse()
        positions.reverse()
    meta = _xfer_meta(
        remote,
        {"d": ("x", destination_ids)},
        hisparse_protocol=1,
        host_group_indices=[source_group],
        group_block_sizes=[16, 16],
        registered_cache_roles=["sparse", "indexer"],
        req_block_positions={"d": positions},
    )
    send = SendBlockMeta(
        "p", "x", [source_ids], asyncio.Event(), local_block_positions=[[0, 1, 2, 3]]
    )
    config = create_vllm_config(
        kv_connector="MooncakeConnector",
        kv_role="kv_producer",
        hf_overrides={"model_type": "deepseek_v2", "kv_lora_rank": 16},
    )
    with mooncake_register_worker(
        _hisparse_producer_cache_config(packed=True), vllm_config=config
    ) as (worker, _):
        worker.register_kv_caches(
            {"model.layers.0.source": source, "model.layers.0.indexer": indexer}
        )
        local, remote, error = worker._prepare_transfer_regions(meta)
        assert error is None
        sources, destinations, lengths, errors, message = asyncio.run(
            worker._build_transfer_params([("d", send)], meta, local, remote)
        )

    assert errors == [] and message is None
    assert sum(lengths) == 5 * page
    for src, dst, length in zip(sources, destinations, lengths):
        ctypes.memmove(dst, src, length)
    assert torch.equal(host, expected_host)
    assert torch.equal(gpu, expected_gpu)
    assert torch.equal(source, original_source)
    assert torch.equal(indexer, original_indexer)
    # Adjacent full pages can share a descriptor; strided subpages cannot.
    assert lengths == ([page, page] if packed else [2 * page]) + [page] * 3


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "rank,shared,expected_sources",
    [
        (0, True, [1200, 1300, 2400]),
        (1, True, [2400]),
        (1, False, [1200, 1300, 2400]),
    ],
)
def test_hisparse_import_keeps_rank_private_indexer(rank, shared, expected_sources):
    """D's host/indexer prefixes differ; only shared host bytes have an owner."""
    send = SendBlockMeta(
        "p", "x", [[1, 2, 3, 4]], asyncio.Event(), local_block_positions=[[0, 1, 2, 3]]
    )
    meta = MooncakeXferMetadata(
        remote_hostname="d",
        remote_port=1,
        remote_tp_size=2,
        remote_tp_rank=rank,
        req_blocks={"d": ("x", [[5, 6], [7]])},
        kv_caches_base_addr=[],
        block_lens=[],
        kv_block_lens=[],
        hisparse_protocol=1,
        host_group_indices=[0],
        shared_host_owner=0 if shared else -1,
        req_block_positions={"d": [[1, 2], [3]]},
    )
    local = [
        TransferRegion("model.layers.0.source", 0, 1000, 100, 10, 0),
        TransferRegion("model.layers.0.indexer", 0, 2000, 100, 10, 0),
    ]
    remote = [
        TransferRegion("model.layers.0.source", 0, 3000, 100, 10, 0),
        TransferRegion("model.layers.0.indexer", 0, 4000, 100, 10, 1),
    ]
    with mooncake_register_worker(_hisparse_producer_cache_config(), tp_size=2) as (
        worker,
        _,
    ):
        sources, destinations, lengths, errors, message = asyncio.run(
            worker._build_transfer_params([("d", send)], meta, local, remote)
        )
    assert sources == expected_sources
    assert destinations == ([3500, 3600, 4700] if rank == 0 or not shared else [4700])
    assert lengths == [10] * len(expected_sources)
    assert errors == [] and message is None


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "positions,indexer_bytes",
    [([[1, 2], [4]], 10), ([[1, 2], [2, 1]], 10), ([[1, 2]], 10), ([[1, 2], [2]], 11)],
)
def test_hisparse_import_rejects_incomplete_page_plan_atomically(
    positions, indexer_bytes
):
    """A bad indexer plan must discard even previously planned host descriptors."""
    send = SendBlockMeta(
        "p", "x", [[1, 2, 3]], asyncio.Event(), local_block_positions=[[0, 1, 2]]
    )
    meta = MooncakeXferMetadata(
        remote_hostname="d",
        remote_port=1,
        remote_tp_size=1,
        remote_tp_rank=0,
        req_blocks={"d": ("x", [[5, 6], [7]])},
        kv_caches_base_addr=[],
        block_lens=[],
        kv_block_lens=[],
        hisparse_protocol=1,
        host_group_indices=[0],
        req_block_positions={"d": positions},
    )
    local = [
        TransferRegion("model.layers.0.source", 0, 1000, 100, 10, 0),
        TransferRegion("model.layers.0.indexer", 0, 2000, 100, 10, 0),
    ]
    remote = [
        TransferRegion("model.layers.0.source", 0, 3000, 100, 10, 0),
        TransferRegion("model.layers.0.indexer", 0, 4000, 100, indexer_bytes, 1),
    ]
    with mooncake_register_worker(_hisparse_producer_cache_config()) as (worker, _):
        sources, destinations, lengths, errors, message = asyncio.run(
            worker._build_transfer_params([("d", send)], meta, local, remote)
        )
    assert sources == destinations == lengths == []
    assert errors == ["d"] and message


@pytest.mark.cpu_test
@pytest.mark.parametrize("incompatible", [None, "role", "block_size", "tp", "opaque"])
def test_hisparse_semantic_layout_requires_compatible_roles(incompatible):
    """Independent group numbers are valid only under the sparse-MLA contract."""
    local = [
        TransferRegion("model.layers.0.source", 0, 1000, 100, 10, 0),
        TransferRegion("model.layers.0.indexer", 0, 2000, 100, 10, 0),
    ]
    remote = [
        TransferRegion("model.layers.0.source", 0, 3000, 100, 10, 0),
        TransferRegion("model.layers.0.indexer", 0, 4000, 100, 10, 1),
    ]
    meta = MooncakeXferMetadata(
        remote_hostname="d",
        remote_port=1,
        remote_tp_size=2,
        remote_tp_rank=0,
        req_blocks={},
        kv_caches_base_addr=[],
        block_lens=[],
        kv_block_lens=[],
        hisparse_protocol=1,
        host_group_indices=[0],
        shared_host_owner=0,
        group_block_sizes=[16, 16],
        registered_cache_roles=["sparse", "indexer"],
    )
    if incompatible == "role":
        meta.registered_cache_roles[1] = "sparse"
    elif incompatible == "block_size":
        meta.group_block_sizes[1] = 32
    elif incompatible == "tp":
        meta.remote_tp_size = 4
    elif incompatible == "opaque":
        remote[1] = TransferRegion(
            "model.layers.0.indexer", 0, 4000, 100, 10, -1, (0, 1)
        )
    config = create_vllm_config(
        kv_connector="MooncakeConnector",
        hf_overrides={"model_type": "deepseek_v2", "kv_lora_rank": 16},
    )
    # Single-rank construction is sufficient to validate the layout contract.
    meta.remote_tp_size = 2 if incompatible == "tp" else 1
    with mooncake_register_worker(
        _hisparse_producer_cache_config(), vllm_config=config
    ) as (worker, _):
        error = worker._validate_hisparse_layout(meta, local, remote)
    assert (error is None) == (incompatible is None)
