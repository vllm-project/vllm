# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MooncakeConnector hybrid FA + GDN support.

GDN is represented as a MambaSpec in vLLM, so these tests exercise the
Mooncake MambaSpec path with mamba_type=GDN_ATTN. Mamba2 is intentionally not
validated by this test module.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tests.v1.attention.utils import dense_kv_cache_views
from vllm.config import set_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    KVConnectorRole,
    MooncakeConnector,
    MooncakeConnectorScheduler,
    MooncakeConnectorWorker,
    MooncakeXferMetadata,
    SendBlockMeta,
    TransferRegion,
    _compute_sender_transfer_plan,
    _validate_asymmetric_region_lengths,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheLayout,
    MambaSpec,
)

from .test_mooncake_connector import patch_worker_dependencies
from .utils import create_request, create_vllm_config


def noop_shutdown():
    pass


def make_hybrid_gdn_kv_cache_config(block_size: int) -> KVCacheConfig:
    gdn_spec = MambaSpec(
        block_size=block_size,
        shapes=((6, 3), (1, 2, 2)),
        dtypes=(torch.float16, torch.float16),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )
    assert gdn_spec.mamba_type == MambaAttentionBackendEnum.GDN_ATTN
    return KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["model.layers.0.self_attn"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float16,
                ),
            ),
            KVCacheGroupSpec(
                ["model.layers.1.linear_attn"],
                gdn_spec,
            ),
        ],
    )


def make_hybrid_gdn_scheduler(kv_role: str) -> MooncakeConnectorScheduler:
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector",
        kv_role=kv_role,
    )
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = False
    return MooncakeConnectorScheduler(
        vllm_config=vllm_config,
        engine_id="test-engine",
        kv_cache_config=make_hybrid_gdn_kv_cache_config(
            vllm_config.cache_config.block_size
        ),
    )


@pytest.mark.cpu_test
def test_hybrid_gdn_remote_prefill_uses_mamba_n_minus_one():
    scheduler = make_hybrid_gdn_scheduler(kv_role="kv_consumer")
    request = create_request(num_tokens=10, do_remote_prefill=True)

    num_new_tokens, is_async = scheduler.get_num_new_matched_tokens(
        request, num_computed_tokens=0
    )

    assert num_new_tokens == request.num_prompt_tokens - 1
    assert is_async is True


@pytest.mark.cpu_test
def test_hybrid_gdn_remote_decode_truncates_prefill_once():
    scheduler = make_hybrid_gdn_scheduler(kv_role="kv_producer")
    request = create_request(num_tokens=10, do_remote_decode=True)
    original_tokens = list(request.prompt_token_ids)

    num_new_tokens, is_async = scheduler.get_num_new_matched_tokens(
        request, num_computed_tokens=0
    )

    assert num_new_tokens == 0
    assert is_async is False
    assert request.prompt_token_ids == original_tokens[:-1]
    assert request._all_token_ids == original_tokens[:-1]
    assert request.num_prompt_tokens == len(original_tokens) - 1
    assert request.max_tokens == 1
    assert request.kv_transfer_params["_p_side_truncated"] is True

    scheduler.get_num_new_matched_tokens(request, num_computed_tokens=0)
    assert request.prompt_token_ids == original_tokens[:-1]


def test_register_kv_caches_emits_fa_and_gdn_regions(monkeypatch):
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector",
        kv_role="kv_consumer",
    )
    kv_cache_config = make_hybrid_gdn_kv_cache_config(
        vllm_config.cache_config.block_size
    )

    with set_current_vllm_config(vllm_config), patch_worker_dependencies():
        connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            kv_cache_config,
        )
        worker = connector.connector_worker

        num_blocks = kv_cache_config.num_blocks
        fa_spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
        gdn_spec = kv_cache_config.kv_cache_groups[1].kv_cache_spec
        fa_raw = torch.empty(num_blocks * fa_spec.page_size_bytes, dtype=torch.int8)
        gdn_raw = torch.empty(num_blocks * gdn_spec.page_size_bytes, dtype=torch.int8)
        (fa_cache,) = dense_kv_cache_views(
            fa_raw, fa_spec, num_blocks, 1, KVCacheLayout.LBHNC
        )
        (gdn_cache,) = dense_kv_cache_views(
            gdn_raw, gdn_spec, num_blocks, 1, KVCacheLayout.LBHNC
        )

        worker.register_kv_caches(
            {
                "model.layers.0.self_attn": fa_cache,
                "model.layers.1.linear_attn": gdn_cache,
            }
        )

        assert worker.transfer_topo.is_mamba is True
        assert worker.registered_layer_names == [
            "model.layers.0.self_attn",
            "model.layers.1.linear_attn",
            "model.layers.1.linear_attn",
        ]
        assert worker.block_len_per_layer == [
            fa_spec.page_size_bytes,
            gdn_spec.page_size_bytes,
            gdn_spec.page_size_bytes,
        ]
        assert worker.registered_group_indices == [0, 1, 1]
        # The GDN page packs conv (36 B) then ssm (8 B); each state registers
        # as its own region with real, unpadded byte length.
        assert worker.kv_caches_base_addr == [
            fa_cache.data_ptr(),
            gdn_cache.data_ptr(),
            gdn_cache.data_ptr() + 36,
        ]
        # FA registers whole pages; the GDN states keep their real byte sizes.
        assert worker.kv_block_len_per_layer == [fa_spec.page_size_bytes, 36, 8]

        worker.shutdown()
        worker.shutdown = noop_shutdown
        connector.connector_worker = None


def test_register_kv_caches_deduplicates_shared_backing_memory(monkeypatch):
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector",
        kv_role="kv_consumer",
    )
    kv_cache_config = make_hybrid_gdn_kv_cache_config(
        vllm_config.cache_config.block_size
    )

    with set_current_vllm_config(vllm_config), patch_worker_dependencies():
        connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            kv_cache_config,
        )
        worker = connector.connector_worker

        backing = torch.empty((4, 64), dtype=torch.float16)
        fa_cache = backing[:2, :16]
        gdn_cache = backing[:3]

        with patch.object(
            worker.engine, "batch_register_memory", return_value=0
        ) as batch_register_memory:
            worker.register_kv_caches(
                {
                    "model.layers.0.self_attn": fa_cache,
                    "model.layers.1.linear_attn": gdn_cache,
                }
            )

        assert worker.kv_caches_base_addr == [
            fa_cache.data_ptr(),
            gdn_cache.data_ptr(),
            gdn_cache.data_ptr() + 36,
        ]
        batch_register_memory.assert_called_once()
        registered_ptrs, registered_lens = batch_register_memory.call_args[0]
        assert registered_ptrs == [backing.data_ptr()]
        assert registered_lens == [backing.untyped_storage().nbytes()]

        worker.shutdown()
        worker.shutdown = noop_shutdown
        connector.connector_worker = None


def test_hybrid_gdn_transfer_params_preserve_group_identity(monkeypatch):
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector",
        kv_role="kv_producer",
    )
    kv_cache_config = make_hybrid_gdn_kv_cache_config(
        vllm_config.cache_config.block_size
    )

    with set_current_vllm_config(vllm_config), patch_worker_dependencies():
        connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            kv_cache_config,
        )
        worker = connector.connector_worker

        block_len = 0x100
        transfer_id = "xfer-hybrid-gdn"

        async def build_transfer_params():
            send_meta = SendBlockMeta(
                p_req_id="p-hybrid-gdn",
                transfer_id=transfer_id,
                local_block_ids=[
                    [10, 11],
                    [NULL_BLOCK_ID, 4],
                ],
                ready=asyncio.Event(),
            )
            return await worker._build_transfer_params(
                [("d-hybrid-gdn", send_meta)],
                xfer_meta,
                local_regions,
                remote_regions,
            )

        xfer_meta = MooncakeXferMetadata(
            remote_hostname="consumer-host",
            remote_port=54321,
            remote_tp_size=1,
            remote_tp_rank=0,
            req_blocks={
                "d-hybrid-gdn": (
                    transfer_id,
                    [
                        [30, 31],
                        [NULL_BLOCK_ID, 7],
                    ],
                )
            },
            kv_caches_base_addr=[],
            block_lens=[],
            kv_block_lens=[],
        )

        local_regions = [
            TransferRegion(
                layer_name="model.layers.1.linear_attn",
                layer_index=1,
                base_addr=0x5000,
                block_len=block_len,
                kv_block_len=block_len,
                group_index=1,
            ),
            TransferRegion(
                layer_name="model.layers.0.self_attn",
                layer_index=0,
                base_addr=0x1000,
                block_len=block_len,
                kv_block_len=block_len,
                group_index=0,
            ),
        ]
        remote_regions = [
            TransferRegion(
                layer_name="model.layers.1.linear_attn",
                layer_index=1,
                base_addr=0x6000,
                block_len=block_len,
                kv_block_len=block_len,
                group_index=1,
            ),
            TransferRegion(
                layer_name="model.layers.0.self_attn",
                layer_index=0,
                base_addr=0x2000,
                block_len=block_len,
                kv_block_len=block_len,
                group_index=0,
            ),
        ]

        src_ptrs, dst_ptrs, lengths, err_reqs, err_msg = asyncio.run(
            build_transfer_params()
        )

        assert err_reqs == []
        assert err_msg is None
        assert src_ptrs == [
            0x5000 + 4 * block_len,
            0x1000 + 10 * block_len,
        ]
        assert dst_ptrs == [
            0x6000 + 7 * block_len,
            0x2000 + 30 * block_len,
        ]
        assert lengths == [block_len, 2 * block_len]

        worker.shutdown()
        worker.shutdown = noop_shutdown
        connector.connector_worker = None


def test_logical_to_kernel_block_ids_expands_fa_not_gdn():
    worker = object.__new__(MooncakeConnectorWorker)
    worker.shutdown = noop_shutdown
    worker._physical_blocks_per_logical_kv_block = 17
    worker.kv_cache_config = make_hybrid_gdn_kv_cache_config(block_size=544)

    block_ids = [[2], [2]]
    kernel_block_ids = worker._logical_to_kernel_block_ids(block_ids)

    assert kernel_block_ids == [list(range(34, 51)), [2]]


def test_hybrid_gdn_keeps_packed_fa_and_gdn_regions_whole(
    monkeypatch,
):
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector",
        kv_role="kv_producer",
    )
    kv_cache_config = make_hybrid_gdn_kv_cache_config(
        vllm_config.cache_config.block_size
    )

    with set_current_vllm_config(vllm_config), patch_worker_dependencies():
        connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            kv_cache_config,
        )
        worker = connector.connector_worker

        worker.transfer_topo = SimpleNamespace(is_kv_layout_blocks_first=False)
        regions = worker._get_transfer_regions(
            base_addrs=[0x1000, 0x2000],
            block_lens=[0x100, 0x100],
            kv_block_lens=[0x40, 0x100],
            layer_names=[
                "model.layers.0.self_attn",
                "model.layers.1.linear_attn",
            ],
            layer_indices=[0, 1],
            group_indices=[0, 1],
        )

        assert [
            (region.group_index, region.base_addr, region.kv_block_len)
            for region in regions
        ] == [
            (0, 0x1000, 0x40),
            (1, 0x2000, 0x100),
        ]

        worker.shutdown()
        worker.shutdown = noop_shutdown
        connector.connector_worker = None


def test_get_transfer_regions_tolerates_peer_pp_stage_layers(monkeypatch):
    # A PP-sharded worker's local spec table covers only its own Mamba/GDN
    # layers, but remote metadata lists every layer. Expanding remote regions
    # for a peer stage's layers must not raise KeyError.
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector",
        kv_role="kv_producer",
    )
    kv_cache_config = make_hybrid_gdn_kv_cache_config(
        vllm_config.cache_config.block_size
    )

    with set_current_vllm_config(vllm_config), patch_worker_dependencies():
        connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            kv_cache_config,
        )
        worker = connector.connector_worker

        regions = worker._get_transfer_regions(
            base_addrs=[0x1000, 0x2000, 0x3000],
            block_lens=[0x100, 0x100, 0x100],
            kv_block_lens=[0x40, 0x100, 0x100],
            layer_names=[
                "model.layers.0.self_attn",
                "model.layers.1.linear_attn",
                "model.layers.31.linear_attn",
            ],
            layer_indices=[0, 1, 31],
            group_indices=[0, 1, 1],
        )

        assert [
            (region.group_index, region.base_addr, region.kv_block_len)
            for region in regions
        ] == [
            (0, 0x1000, 0x40),
            (1, 0x2000, 0x100),
            (1, 0x3000, 0x100),
        ]

        worker.shutdown()
        worker.shutdown = noop_shutdown
        connector.connector_worker = None


def test_transfer_plan_full_copy_for_replicated_consumer_kv():
    # P TP2 -> D TP4 with 2 KV heads: every consumer rank holds a full replica
    # of its head group's attention region instead of a slice of the
    # producer's.
    assert _compute_sender_transfer_plan(
        local_tp_rank=0,
        local_tp_size=2,
        remote_tp_rank=3,
        remote_tp_size=4,
        local_kv_block_len=1000,
        remote_kv_block_len=1000,
        producer_cache_replicated=False,
        consumer_kv_replicated=True,
    ) == (True, 0, 0, 1000)
    # Mamba/GDN states shard by head count and keep ratio slicing.
    assert _compute_sender_transfer_plan(
        local_tp_rank=0,
        local_tp_size=2,
        remote_tp_rank=3,
        remote_tp_size=4,
        local_kv_block_len=2000,
        remote_kv_block_len=1000,
        producer_cache_replicated=False,
        consumer_kv_replicated=False,
    ) == (True, 1000, 0, 1000)


def test_validate_region_lengths_allows_replicated_consumer_attention():
    kv_cache_config = make_hybrid_gdn_kv_cache_config(block_size=16)
    local_regions = [
        TransferRegion("model.layers.0.self_attn", 0, 0x1000, 2000, 1000, 0),
        TransferRegion("model.layers.1.linear_attn", 1, 0x5000, 2000, 2000, 1),
    ]
    remote_regions = [
        TransferRegion("model.layers.0.self_attn", 0, 0x2000, 2000, 1000, 0),
        TransferRegion("model.layers.1.linear_attn", 1, 0x6000, 2000, 1000, 1),
    ]
    # Replicated consumer KV: attention regions must match exactly, Mamba/GDN
    # regions keep the TP-ratio rule (local == 2x remote for TP2 -> TP4).
    assert (
        _validate_asymmetric_region_lengths(
            local_regions,
            remote_regions,
            local_tp_size=2,
            remote_tp_size=4,
            producer_cache_replicated=False,
            group_specs=kv_cache_config.kv_cache_groups,
            total_num_kv_heads=2,
        )
        is None
    )

    mismatched_remote = [
        TransferRegion("model.layers.0.self_attn", 0, 0x2000, 2000, 500, 0),
        remote_regions[1],
    ]
    assert "replicated consumer KV" in _validate_asymmetric_region_lengths(
        local_regions,
        mismatched_remote,
        local_tp_size=2,
        remote_tp_size=4,
        producer_cache_replicated=False,
        group_specs=kv_cache_config.kv_cache_groups,
        total_num_kv_heads=2,
    )


@pytest.mark.cpu_test
def test_register_kv_caches_splits_gdn_conv_sub_projections(monkeypatch):
    """DS conv layout: each GDN conv sub-projection is its own region.

    Het-TP slicing splits every region by TP ratio, so the conv page must be
    decomposed into Q/K/V regions; a single conv region would let the ratio
    split cut across projection boundaries and scramble the state.
    """
    from vllm.model_executor.layers.mamba.mamba_utils import (
        get_conv_state_layout,
    )

    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
    monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", "DS")
    get_conv_state_layout.cache_clear()
    try:
        vllm_config = create_vllm_config(
            kv_connector="MooncakeConnector",
            kv_role="kv_consumer",
        )
        kv_cache_config = make_hybrid_gdn_kv_cache_config(
            vllm_config.cache_config.block_size
        )

        with set_current_vllm_config(vllm_config), patch_worker_dependencies():
            connector = MooncakeConnector(
                vllm_config,
                KVConnectorRole.WORKER,
                kv_cache_config,
            )
            worker = connector.connector_worker

            fa_cache = torch.empty((2, 2, 11), dtype=torch.float16)
            gdn_page_bytes = kv_cache_config.kv_cache_groups[
                1
            ].kv_cache_spec.page_size_bytes
            gdn_cache = torch.empty((2, 1, 1, gdn_page_bytes), dtype=torch.int8)

            worker.register_kv_caches(
                {
                    "model.layers.0.self_attn": fa_cache,
                    "model.layers.1.linear_attn": gdn_cache,
                }
            )

            # Fixture GDN spec: conv (6, 3) fp16 -> Q/K/V dims (2, 2, 2),
            # 12 B each in DS layout; ssm (1, 2, 2) fp16 -> 8 B at offset 36.
            assert worker.registered_layer_names == [
                "model.layers.0.self_attn",
                *["model.layers.1.linear_attn"] * 4,
            ]
            assert worker.registered_group_indices == [0, 1, 1, 1, 1]
            assert worker.kv_caches_base_addr == [
                fa_cache.data_ptr(),
                gdn_cache.data_ptr(),
                gdn_cache.data_ptr() + 12,
                gdn_cache.data_ptr() + 24,
                gdn_cache.data_ptr() + 36,
            ]
            fa_page_bytes = kv_cache_config.kv_cache_groups[
                0
            ].kv_cache_spec.page_size_bytes
            assert worker.kv_block_len_per_layer == [fa_page_bytes, 12, 12, 12, 8]

            worker.shutdown()
            worker.shutdown = noop_shutdown
            connector.connector_worker = None
    finally:
        get_conv_state_layout.cache_clear()


def test_gdn_conv_sub_projection_regions_align_across_het_tp(monkeypatch):
    """P TP2 -> D TP4 GDN: per-region ratio slices land inside projections."""
    from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (  # noqa: E501
        derive_mamba_conv_split,
    )
    from vllm.model_executor.layers.mamba.mamba_utils import (
        get_conv_state_layout,
    )

    monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", "DS")
    get_conv_state_layout.cache_clear()
    try:
        # Global GDN dims: key_dim=4 (Q == K), value_dim=8 -> conv_dim=16.
        def gdn_spec(tp: int) -> MambaSpec:
            return MambaSpec(
                block_size=16,
                shapes=((16 // tp, 3), (8 // tp // 2, 2, 2)),
                dtypes=(torch.float16, torch.float16),
                mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
            )

        p_split = derive_mamba_conv_split(gdn_spec(tp=2), local_tp=2)
        d_split = derive_mamba_conv_split(gdn_spec(tp=4), local_tp=4)
        assert p_split.local_proj_dims == (2, 2, 4)
        assert d_split.local_proj_dims == (1, 1, 2)

        # Every sub-projection region obeys the TP-ratio rule (P len == 2x D
        # len), and each D rank slices its own half of each P sub-projection.
        row_bytes = 3 * 2  # conv_rows x fp16
        for p_dim, d_dim in zip(p_split.local_proj_dims, d_split.local_proj_dims):
            p_len = p_dim * row_bytes
            d_len = d_dim * row_bytes
            assert p_len == 2 * d_len
            for d_rank in range(4):
                assert _compute_sender_transfer_plan(
                    local_tp_rank=d_rank // 2,
                    local_tp_size=2,
                    remote_tp_rank=d_rank,
                    remote_tp_size=4,
                    local_kv_block_len=p_len,
                    remote_kv_block_len=d_len,
                    producer_cache_replicated=False,
                    consumer_kv_replicated=False,
                ) == (True, (d_rank % 2) * d_len, 0, d_len)
    finally:
        get_conv_state_layout.cache_clear()
