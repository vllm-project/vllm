# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MooncakeConnector HMA (Hybrid Memory Architecture) support.

Covers sliding-window clipping, multi-group metadata shape, multi-group
send trimming, and group-count invariant checking in _build_transfer_params.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.config import set_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    KVConnectorRole,
    MooncakeConnector,
    MooncakeConnectorMetadata,
    MooncakeConnectorScheduler,
    MooncakeConnectorWorker,
    MooncakeXferMetadata,
    SendBlockMeta,
    TransferRegion,
    _align_transfer_regions,
    _common_group_indices_for_regions,
    _select_region_block_ids,
)

from .test_mooncake_connector import FakeMooncakeWrapper, patch_worker_dependencies
from .utils import create_request, create_vllm_config, make_kv_cache_config


def make_transfer_worker() -> MooncakeConnectorWorker:
    worker = MooncakeConnectorWorker.__new__(MooncakeConnectorWorker)
    worker.async_zmq_ctx = SimpleNamespace(term=lambda: None)
    worker.is_kv_consumer = True
    worker.is_kv_producer = True
    worker.tp_rank = 0
    worker.tp_size = 1
    worker.transfer_topo = SimpleNamespace(local_replicates_kv_cache=False)
    return worker


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
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector", kv_role="kv_producer"
    )
    kv_cache_config = make_kv_cache_config(
        block_size=vllm_config.cache_config.block_size, swa_enabled=True
    )

    with set_current_vllm_config(vllm_config), patch_worker_dependencies():
        connector = MooncakeConnector(
            vllm_config, KVConnectorRole.WORKER, kv_cache_config
        )
        worker = connector.connector_worker

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


def test_common_group_indices_treats_missing_metadata_as_all_groups():
    local_region = TransferRegion(
        layer_name="model.layers.4.self_attn",
        layer_index=4,
        base_addr=0x1000,
        block_len=4096,
        kv_block_len=4096,
        group_indices=(0,),
    )
    remote_region = TransferRegion(
        layer_name="model.layers.4.self_attn",
        layer_index=4,
        base_addr=0x2000,
        block_len=4096,
        kv_block_len=4096,
    )
    annotated_remote_region = TransferRegion(
        layer_name="model.layers.4.self_attn",
        layer_index=4,
        base_addr=0x3000,
        block_len=4096,
        kv_block_len=4096,
        group_indices=(0, 2),
    )

    assert _common_group_indices_for_regions(
        local_region,
        remote_region,
        num_groups=3,
    ) == (0, 1, 2)
    assert _common_group_indices_for_regions(
        remote_region,
        local_region,
        num_groups=3,
    ) == (0, 1, 2)
    assert _common_group_indices_for_regions(
        local_region,
        annotated_remote_region,
        num_groups=3,
    ) == (0,)


def test_select_region_block_ids_skips_empty_remote_groups():
    local_block_ids, remote_block_ids, err = _select_region_block_ids(
        local_block_ids_per_group=[
            [10, 11],
            [20, 21],
        ],
        remote_block_ids_per_group=[
            [],
            [120, 121],
        ],
        group_indices=(0, 1),
    )

    assert err is None
    assert local_block_ids == [20, 21]
    assert remote_block_ids == [120, 121]
    assert len(local_block_ids) == len(remote_block_ids)


def test_align_transfer_regions_fans_out_shared_alias_groups():
    local_swa_layer = TransferRegion(
        layer_name="model.layers.4.attn.swa_cache",
        layer_index=4,
        base_addr=0x1000,
        block_len=100,
        kv_block_len=100,
        layer_aliases=(
            "model.layers.4.attn.swa_cache",
            "model.layers.4.attn.compressor.state_cache",
        ),
        layer_indices=(4, 4),
        group_indices=(1, 3),
        alias_group_indices=((1,), (3,)),
    )
    local_layer = TransferRegion(
        layer_name="model.layers.6.attn",
        layer_index=6,
        base_addr=0x2000,
        block_len=100,
        kv_block_len=100,
        layer_aliases=(
            "model.layers.6.attn",
            "model.layers.6.attn.swa_cache",
            "model.layers.5.attn.swa_cache",
            "model.layers.6.attn.compressor.state_cache",
            "model.layers.5.attn.compressor.state_cache",
        ),
        layer_indices=(6, 6, 5, 6, 5),
        group_indices=(0, 1, 2, 3, 4),
        alias_group_indices=((0,), (1,), (2,), (3,), (4,)),
    )
    remote_prev_layer = TransferRegion(
        layer_name="model.layers.4.attn",
        layer_index=4,
        base_addr=0x3000,
        block_len=100,
        kv_block_len=100,
        layer_aliases=(
            "model.layers.4.attn",
            "model.layers.2.attn.swa_cache",
            "model.layers.3.attn.swa_cache",
            "model.layers.4.attn.compressor.state_cache",
            "model.layers.5.attn.compressor.state_cache",
        ),
        layer_indices=(4, 2, 3, 4, 5),
        group_indices=(0, 1, 2, 3, 4),
        alias_group_indices=((0,), (1,), (2,), (3,), (4,)),
    )
    remote_current_layer = TransferRegion(
        layer_name="model.layers.6.attn",
        layer_index=6,
        base_addr=0x4000,
        block_len=100,
        kv_block_len=100,
        layer_aliases=(
            "model.layers.6.attn",
            "model.layers.4.attn.swa_cache",
            "model.layers.5.attn.swa_cache",
            "model.layers.6.attn.compressor.state_cache",
            "model.layers.7.attn.compressor.state_cache",
        ),
        layer_indices=(6, 4, 5, 6, 7),
        group_indices=(0, 1, 2, 3, 4),
        alias_group_indices=((0,), (1,), (2,), (3,), (4,)),
    )
    remote_next_layer = TransferRegion(
        layer_name="model.layers.8.attn",
        layer_index=8,
        base_addr=0x5000,
        block_len=100,
        kv_block_len=100,
        layer_aliases=(
            "model.layers.8.attn",
            "model.layers.6.attn.swa_cache",
            "model.layers.7.attn.swa_cache",
            "model.layers.8.attn.compressor.state_cache",
            "model.layers.9.attn.compressor.state_cache",
        ),
        layer_indices=(8, 6, 7, 8, 9),
        group_indices=(0, 1, 2, 3, 4),
        alias_group_indices=((0,), (1,), (2,), (3,), (4,)),
    )

    local_regions, remote_regions, err = _align_transfer_regions(
        [local_swa_layer, local_layer],
        [remote_prev_layer, remote_current_layer, remote_next_layer],
    )

    assert err is None
    aligned_groups = [
        (
            local_region.layer_name,
            remote_region.layer_name,
            _common_group_indices_for_regions(
                local_region,
                remote_region,
                num_groups=5,
            ),
        )
        for local_region, remote_region in zip(local_regions, remote_regions)
    ]
    assert aligned_groups == [
        (
            "model.layers.4.attn.swa_cache",
            "model.layers.4.attn",
            (3,),
        ),
        (
            "model.layers.4.attn.swa_cache",
            "model.layers.6.attn",
            (1,),
        ),
        (
            "model.layers.6.attn",
            "model.layers.4.attn",
            (4,),
        ),
        (
            "model.layers.6.attn",
            "model.layers.6.attn",
            (0, 2, 3),
        ),
        (
            "model.layers.6.attn",
            "model.layers.8.attn",
            (1,),
        ),
    ]


def test_align_transfer_regions_rejects_unbound_alias_index_match():
    local_region = TransferRegion(
        layer_name="model.layers.10.attn",
        layer_index=10,
        base_addr=0x1000,
        block_len=100,
        kv_block_len=100,
        layer_aliases=(
            "model.layers.10.attn",
            "model.layers.11.attn.swa_cache",
        ),
        layer_indices=(10, 11),
        group_indices=(0, 1),
        alias_group_indices=((0,), (1,)),
    )
    remote_region = TransferRegion(
        layer_name="model.layers.12.attn",
        layer_index=12,
        base_addr=0x2000,
        block_len=100,
        kv_block_len=100,
        layer_aliases=(
            "model.layers.10.attn",
            "model.layers.12.attn.swa_cache",
        ),
        layer_indices=(12, 11),
        group_indices=(0, 1),
        alias_group_indices=((0,), (1,)),
    )

    local_regions, remote_regions, err = _align_transfer_regions(
        [local_region],
        [remote_region],
    )

    assert local_regions == []
    assert remote_regions == []
    assert err is not None
    assert "index mismatch" in err


def test_align_transfer_regions_rejects_duplicate_remote_alias_group():
    local_region_a = TransferRegion(
        layer_name="model.layers.4.attn",
        layer_index=4,
        base_addr=0x1000,
        block_len=100,
        kv_block_len=100,
        layer_aliases=("model.layers.4.attn",),
        layer_indices=(4,),
        group_indices=(0,),
        alias_group_indices=((0,),),
    )
    local_region_b = TransferRegion(
        layer_name="model.layers.4.attn.swa_cache",
        layer_index=4,
        base_addr=0x2000,
        block_len=100,
        kv_block_len=100,
        layer_aliases=("model.layers.4.attn",),
        layer_indices=(4,),
        group_indices=(0,),
        alias_group_indices=((0,),),
    )
    remote_region = TransferRegion(
        layer_name="model.layers.4.attn",
        layer_index=4,
        base_addr=0x3000,
        block_len=100,
        kv_block_len=100,
        layer_aliases=("model.layers.4.attn",),
        layer_indices=(4,),
        group_indices=(0,),
        alias_group_indices=((0,),),
    )

    local_regions, remote_regions, err = _align_transfer_regions(
        [local_region_a, local_region_b],
        [remote_region],
    )

    assert local_regions == []
    assert remote_regions == []
    assert err is not None
    assert "duplicate alias group" in err


@pytest.mark.asyncio
async def test_build_transfer_params_filters_groups_per_shared_alias():
    worker = make_transfer_worker()
    block_len = 100
    transfer_id = "xfer-dsv4-shifted-alias"
    send_meta = SendBlockMeta(
        p_req_id="p-dsv4-shifted-alias",
        transfer_id=transfer_id,
        local_block_ids=[
            [10],
            [20],
            [30],
            [40],
            [50],
        ],
        ready=asyncio.Event(),
    )
    xfer_meta = MooncakeXferMetadata(
        remote_hostname="consumer-host",
        remote_port=54321,
        remote_tp_size=1,
        remote_tp_rank=0,
        req_blocks={
            "d-dsv4-shifted-alias": (
                transfer_id,
                [
                    [110],
                    [120],
                    [130],
                    [140],
                    [150],
                ],
            )
        },
        kv_caches_base_addr=[0x2000, 0x3000],
        block_lens=[block_len, block_len],
    )

    local_region = TransferRegion(
        layer_name="model.layers.30.attn",
        layer_index=30,
        base_addr=0x1000,
        block_len=block_len,
        kv_block_len=block_len,
        layer_aliases=(
            "model.layers.30.attn",
            "model.layers.30.attn.swa_cache",
            "model.layers.31.attn.swa_cache",
            "model.layers.30.attn.compressor.state_cache",
            "model.layers.31.attn.compressor.state_cache",
        ),
        layer_indices=(30, 30, 31, 30, 31),
        group_indices=(0, 1, 2, 3, 4),
        alias_group_indices=((0,), (1,), (2,), (3,), (4,)),
    )
    remote_current_layer = TransferRegion(
        layer_name="model.layers.30.attn",
        layer_index=30,
        base_addr=0x2000,
        block_len=block_len,
        kv_block_len=block_len,
        layer_aliases=(
            "model.layers.30.attn",
            "model.layers.28.attn.swa_cache",
            "model.layers.29.attn.swa_cache",
            "model.layers.30.attn.compressor.state_cache",
            "model.layers.31.attn.compressor.state_cache",
        ),
        layer_indices=(30, 28, 29, 30, 31),
        group_indices=(0, 1, 2, 3, 4),
        alias_group_indices=((0,), (1,), (2,), (3,), (4,)),
    )
    remote_shifted_layer = TransferRegion(
        layer_name="model.layers.32.attn",
        layer_index=32,
        base_addr=0x3000,
        block_len=block_len,
        kv_block_len=block_len,
        layer_aliases=(
            "model.layers.32.attn",
            "model.layers.30.attn.swa_cache",
            "model.layers.31.attn.swa_cache",
            "model.layers.32.attn.compressor.state_cache",
            "model.layers.33.attn.compressor.state_cache",
        ),
        layer_indices=(32, 30, 31, 32, 33),
        group_indices=(0, 1, 2, 3, 4),
        alias_group_indices=((0,), (1,), (2,), (3,), (4,)),
    )

    (
        src_ptrs,
        dst_ptrs,
        lengths,
        err_reqs,
        err_msg,
    ) = await worker._build_transfer_params(
        [("d-dsv4-shifted-alias", send_meta)],
        xfer_meta,
        [local_region, local_region],
        [remote_current_layer, remote_shifted_layer],
    )

    assert err_reqs == []
    assert err_msg is None
    assert lengths == [block_len] * 5
    assert [(ptr - 0x1000) // block_len for ptr in src_ptrs] == [
        10,
        40,
        50,
        20,
        30,
    ]
    assert [
        (ptr - base) // block_len
        for ptr, base in zip(
            dst_ptrs,
            [0x2000, 0x2000, 0x2000, 0x3000, 0x3000],
        )
    ] == [
        110,
        140,
        150,
        120,
        130,
    ]


@pytest.mark.asyncio
async def test_build_transfer_params_filters_groups_per_shared_region():
    worker = make_transfer_worker()
    block_len = 4096
    transfer_id = "xfer-dsv4-shared"
    send_meta = SendBlockMeta(
        p_req_id="p-dsv4-shared",
        transfer_id=transfer_id,
        local_block_ids=[
            [10, 11],
            [20, 21],
            [30, 31],
        ],
        ready=asyncio.Event(),
    )
    xfer_meta = MooncakeXferMetadata(
        remote_hostname="consumer-host",
        remote_port=54321,
        remote_tp_size=1,
        remote_tp_rank=0,
        req_blocks={
            "d-dsv4-shared": (
                transfer_id,
                [
                    [110, 111],
                    [120, 121],
                    [130, 131],
                ],
            )
        },
        kv_caches_base_addr=[0x2000, 0x3000],
        block_lens=[block_len, block_len],
    )

    local_regions = [
        TransferRegion(
            layer_name="model.layers.4.self_attn",
            layer_index=4,
            base_addr=0x1000,
            block_len=block_len,
            kv_block_len=block_len,
            layer_aliases=("model.layers.4.self_attn",),
            layer_indices=(4,),
            group_indices=(0, 1),
        ),
        TransferRegion(
            layer_name="model.layers.4.swa_attn",
            layer_index=4,
            base_addr=0x4000,
            block_len=block_len,
            kv_block_len=block_len,
            layer_aliases=("model.layers.4.swa_attn",),
            layer_indices=(4,),
            group_indices=(2,),
        ),
    ]
    remote_regions = [
        TransferRegion(
            layer_name="model.layers.4.self_attn",
            layer_index=4,
            base_addr=0x2000,
            block_len=block_len,
            kv_block_len=block_len,
            layer_aliases=("model.layers.4.self_attn",),
            layer_indices=(4,),
            group_indices=(0,),
        ),
        TransferRegion(
            layer_name="model.layers.4.swa_attn",
            layer_index=4,
            base_addr=0x3000,
            block_len=block_len,
            kv_block_len=block_len,
            layer_aliases=("model.layers.4.swa_attn",),
            layer_indices=(4,),
            group_indices=(1, 2),
        ),
    ]

    (
        src_ptrs,
        dst_ptrs,
        lengths,
        err_reqs,
        err_msg,
    ) = await worker._build_transfer_params(
        [("d-dsv4-shared", send_meta)],
        xfer_meta,
        local_regions,
        remote_regions,
    )

    assert err_reqs == []
    assert err_msg is None
    assert lengths == [2 * block_len, 2 * block_len]
    assert src_ptrs == [
        0x1000 + 10 * block_len,
        0x4000 + 30 * block_len,
    ]
    assert dst_ptrs == [
        0x2000 + 110 * block_len,
        0x3000 + 130 * block_len,
    ]


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
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector", kv_role="kv_producer"
    )
    kv_cache_config = make_kv_cache_config(
        block_size=vllm_config.cache_config.block_size, swa_enabled=True
    )

    with set_current_vllm_config(vllm_config), patch_worker_dependencies():
        connector = MooncakeConnector(
            vllm_config, KVConnectorRole.WORKER, kv_cache_config
        )
        worker = connector.connector_worker

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
