# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MooncakeConnector HMA (Hybrid Memory Architecture) support.

Covers sliding-window clipping, multi-group metadata shape, multi-group
send trimming, and group-count invariant checking in _build_transfer_params.
"""

import asyncio
from unittest.mock import patch

import pytest

from vllm.config import set_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    KVConnectorRole,
    MooncakeConnector,
    MooncakeConnectorMetadata,
    MooncakeConnectorScheduler,
    MooncakeXferMetadata,
    SendBlockMeta,
    TransferRegion,
)

from .test_mooncake_connector import FakeMooncakeWrapper, patch_worker_dependencies
from .utils import create_request, create_vllm_config, make_kv_cache_config


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
        )

        local_regions = [
            TransferRegion(
                base_addr=0x1000, block_len=block_len, kv_block_len=block_len
            ),
        ]
        remote_regions = [
            TransferRegion(
                base_addr=0x2000, block_len=block_len, kv_block_len=block_len
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
        )

        local_regions = [
            TransferRegion(
                base_addr=0x1000, block_len=block_len, kv_block_len=block_len
            ),
        ]
        remote_regions = [
            TransferRegion(
                base_addr=0x2000, block_len=block_len, kv_block_len=block_len
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
