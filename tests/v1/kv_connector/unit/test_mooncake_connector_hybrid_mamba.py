# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for MooncakeConnector hybrid FA + GDN support.

GDN is represented as a MambaSpec in vLLM, so these tests exercise the
Mooncake MambaSpec path with mamba_type=GDN_ATTN. Mamba2 is intentionally not
validated by this test module.
"""

import asyncio
import ctypes
from collections import OrderedDict
from math import prod
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
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheLayout,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

from .test_mooncake_connector import patch_worker_dependencies
from .utils import create_request, create_vllm_config, maybe_update_block_size


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
def test_hybrid_gdn_remote_decode_truncates_prefill_before_cache_lookup():
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector",
        kv_role="kv_producer",
    )
    vllm_config.scheduler_config.disable_hybrid_kv_cache_manager = False
    connector = MooncakeConnector(
        vllm_config,
        KVConnectorRole.SCHEDULER,
        make_hybrid_gdn_kv_cache_config(vllm_config.cache_config.block_size),
    )
    scheduler = connector.connector_scheduler
    assert scheduler is not None
    request = create_request(num_tokens=10, do_remote_decode=True)
    original_tokens = list(request.prompt_token_ids)

    connector.on_new_request(request)
    assert request.prompt_token_ids == original_tokens[:-1]
    assert request._all_token_ids == original_tokens[:-1]
    assert request.num_prompt_tokens == len(original_tokens) - 1
    assert request.max_tokens == 1
    assert request.kv_transfer_params["_p_side_truncated"] is True

    # Re-adding or rescheduling the request must not truncate another token.
    scheduler.on_new_request(request)
    assert request.prompt_token_ids == original_tokens[:-1]

    # Prefix-cache matching must not mutate the request after its local lookup.
    with patch.object(
        scheduler,
        "_truncate_mamba_request_for_prefill",
        side_effect=AssertionError("late Mamba truncation"),
    ):
        num_new_tokens, is_async = scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens=0
        )

    assert num_new_tokens == 0
    assert is_async is False


def test_register_kv_caches_emits_fa_and_gdn_regions(monkeypatch):
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector",
        kv_role="kv_consumer",
        block_size=maybe_update_block_size(16),
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
        gdn_block_stride = gdn_spec.page_size_bytes + 64
        gdn_raw = torch.empty(num_blocks, gdn_block_stride, dtype=torch.int8)
        (fa_cache,) = dense_kv_cache_views(
            fa_raw, fa_spec, num_blocks, 1, KVCacheLayout.LBHNC
        )
        gdn_cache = gdn_raw[:, : gdn_spec.page_size_bytes]

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
        # Both GDN states sit inside the padded GDN block, so they keep its
        # stride; their transfer lengths are checked below.
        assert worker.block_len_per_layer == [
            fa_spec.page_size_bytes,
            gdn_block_stride,
            gdn_block_stride,
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


def test_register_kv_caches_keeps_gdn_states_inside_a_packed_block(monkeypatch):
    """A GDN page inside a larger packed block (another layer's pages share
    each block id's row) must not copy the row stride from the page's offset,
    which would run into the next block id, owned by another request."""
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
        gdn_spec = kv_cache_config.kv_cache_groups[1].kv_cache_spec
        offset, stride = 64, 128
        assert offset + gdn_spec.page_size_bytes <= stride
        packed = torch.empty(num_blocks * stride, dtype=torch.int8)
        gdn_cache = packed.view(num_blocks, stride)[
            :, offset : offset + gdn_spec.page_size_bytes
        ]

        worker.register_kv_caches({"model.layers.1.linear_attn": gdn_cache})

        base = packed.data_ptr() + offset
        assert worker.kv_caches_base_addr == [base, base + 36]
        assert worker.block_len_per_layer == [stride, stride]
        assert worker.kv_block_len_per_layer == [36, 8]

        worker.shutdown()
        worker.shutdown = noop_shutdown
        connector.connector_worker = None


def test_register_kv_caches_scales_attention_len_to_kernel_block(monkeypatch):
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector",
        kv_role="kv_consumer",
        block_size=maybe_update_block_size(16),
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
        factor = 4
        worker._physical_blocks_per_logical_kv_block = factor

        fa_spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
        physical_page_bytes = fa_spec.page_size_bytes // factor
        fa_cache = torch.empty(
            kv_cache_config.num_blocks * factor,
            physical_page_bytes,
            dtype=torch.uint8,
        )

        worker.register_kv_caches({"model.layers.0.self_attn": fa_cache})

        assert worker.block_len_per_layer == [physical_page_bytes]
        assert worker.kv_block_len_per_layer == [physical_page_bytes]

        worker.shutdown()
        worker.shutdown = noop_shutdown
        connector.connector_worker = None


@pytest.mark.asyncio
async def test_build_transfer_params_uses_non_overlapping_physical_pages():
    kv_cache_config = make_hybrid_gdn_kv_cache_config(block_size=16)
    worker = object.__new__(MooncakeConnectorWorker)
    worker.shutdown = noop_shutdown
    # __new__ skips __init__, where these caches are created.
    worker._prepared_transfer_regions = {}
    worker._layout_by_peer = OrderedDict()
    worker._acked_layout_peers = set()
    worker.use_mla = False
    worker.engine = SimpleNamespace(batch_register_memory=lambda *_: 0)
    worker.is_kv_consumer = True
    worker.tp_rank = 0
    worker.tp_size = 1
    worker.transfer_topo = SimpleNamespace(
        local_replicates_kv_cache=False,
        total_num_kv_heads=1,
    )
    worker.kv_cache_config = kv_cache_config
    worker._layer_specs = {
        layer_name: group.kv_cache_spec
        for group in kv_cache_config.kv_cache_groups
        for layer_name in group.layer_names
    }
    worker._layer_group_indices = {
        layer_name: group_index
        for group_index, group in enumerate(kv_cache_config.kv_cache_groups)
        for layer_name in group.layer_names
    }

    ratio = 4
    worker._physical_blocks_per_logical_kv_block = ratio
    fa_spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
    physical_page_bytes = fa_spec.page_size_bytes // ratio
    fa_cache = torch.empty(
        kv_cache_config.num_blocks * ratio,
        physical_page_bytes,
        dtype=torch.uint8,
    )
    worker.register_kv_caches({"model.layers.0.self_attn": fa_cache})

    local_regions = worker._get_transfer_regions(
        worker.kv_caches_base_addr,
        worker.block_len_per_layer,
        worker.kv_block_len_per_layer,
        worker.registered_layer_names,
        worker.registered_layer_indices,
        worker.registered_group_indices,
    )
    remote_base_addr = 0x100000
    remote_regions = [
        TransferRegion(
            layer_name="model.layers.0.self_attn",
            layer_index=0,
            base_addr=remote_base_addr,
            block_len=physical_page_bytes,
            kv_block_len=physical_page_bytes,
            group_index=0,
        )
    ]

    local_logical_block = 2
    remote_logical_block = 5
    transfer_id = "xfer-physical-attention-pages"
    send_meta = SendBlockMeta(
        p_req_id="p-physical-attention-pages",
        transfer_id=transfer_id,
        local_block_ids=[[local_logical_block], []],
        ready=asyncio.Event(),
    )
    xfer_meta = MooncakeXferMetadata(
        remote_hostname="consumer-host",
        remote_port=54321,
        remote_tp_size=1,
        remote_tp_rank=0,
        req_blocks={
            "d-physical-attention-pages": (
                transfer_id,
                [[remote_logical_block], []],
            )
        },
        kv_caches_base_addr=[],
        block_lens=[],
        kv_block_lens=[],
    )

    (
        src_ptrs,
        dst_ptrs,
        lengths,
        err_reqs,
        err_msg,
    ) = await worker._build_transfer_params(
        [("d-physical-attention-pages", send_meta)],
        xfer_meta,
        local_regions,
        remote_regions,
    )

    assert err_reqs == []
    assert err_msg is None
    assert src_ptrs == [
        fa_cache.data_ptr() + local_logical_block * ratio * physical_page_bytes
    ]
    assert dst_ptrs == [
        remote_base_addr + remote_logical_block * ratio * physical_page_bytes
    ]
    assert lengths == [ratio * physical_page_bytes]

    def split_contiguous_blocks(src_ids, dst_ids):
        return [[block_id] for block_id in src_ids], [
            [block_id] for block_id in dst_ids
        ]

    with patch(
        "vllm.distributed.kv_transfer.kv_connector.v1.mooncake."
        "mooncake_connector.group_concurrent_contiguous",
        side_effect=split_contiguous_blocks,
    ):
        (
            src_ptrs,
            dst_ptrs,
            lengths,
            err_reqs,
            err_msg,
        ) = await worker._build_transfer_params(
            [("d-physical-attention-pages", send_meta)],
            xfer_meta,
            local_regions,
            remote_regions,
        )

    expected_src_ptrs = [
        fa_cache.data_ptr()
        + (local_logical_block * ratio + offset) * physical_page_bytes
        for offset in range(ratio)
    ]
    expected_dst_ptrs = [
        remote_base_addr + (remote_logical_block * ratio + offset) * physical_page_bytes
        for offset in range(ratio)
    ]
    assert err_reqs == []
    assert err_msg is None
    assert src_ptrs == expected_src_ptrs
    assert dst_ptrs == expected_dst_ptrs
    assert lengths == [physical_page_bytes] * ratio
    assert all(
        next_ptr - ptr == physical_page_bytes
        for ptr, next_ptr in zip(src_ptrs, src_ptrs[1:])
    )
    assert all(
        next_ptr - ptr == physical_page_bytes
        for ptr, next_ptr in zip(dst_ptrs, dst_ptrs[1:])
    )


def test_register_kv_caches_deduplicates_shared_backing_memory(monkeypatch):
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector",
        kv_role="kv_consumer",
        block_size=maybe_update_block_size(16),
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

        fa_spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
        fa_cache_width = (
            fa_spec.page_size_bytes
            // torch.tensor([], dtype=torch.float16).element_size()
        )
        backing = torch.empty((4, fa_cache_width), dtype=torch.float16)
        fa_cache = backing[:2]
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
        block_size=maybe_update_block_size(16),
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
        block_size=maybe_update_block_size(16),
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


def test_transfer_rule_follows_plain_head_mapping_for_hybrid_attention(monkeypatch):
    """Hybrid attention follows the head mapping of plain models; Mamba/GDN
    states shard by TP ratio, or copy whole when replicated across TP. The rule
    comes from each layer's own spec, also when the group's spec wraps its
    layers in a UniformTypeKVCacheSpecs, as hybrid models such as Qwen3.8 do."""
    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector",
        kv_role="kv_consumer",
    )
    block_size = vllm_config.cache_config.block_size
    base = make_hybrid_gdn_kv_cache_config(block_size)
    gdn_spec = base.kv_cache_groups[1].kv_cache_spec
    # Main attention shards the model's KV heads across TP (TP1 here).
    attention_spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=vllm_config.model_config.get_total_num_kv_heads(),
        head_size=1,
        dtype=torch.float16,
    )
    # Auxiliary caches that every rank holds whole.
    indexer_ring_spec = CircularBufferSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=8,
        head_size_v=0,
        dtype=torch.bfloat16,
    )
    compressed_key_spec = MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.bfloat16,
    )
    replicated_spec = MambaSpec(
        block_size=block_size,
        shapes=((4, 3),),
        dtypes=(torch.float16,),
        mamba_type=MambaAttentionBackendEnum.SHORT_CONV,
        tp_replicated=True,
    )

    def wrapped(name: str, spec) -> KVCacheGroupSpec:
        return KVCacheGroupSpec(
            [name],
            UniformTypeKVCacheSpecs(block_size=block_size, kv_cache_specs={name: spec}),
        )

    kv_cache_config = KVCacheConfig(
        num_blocks=base.num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            wrapped("model.layers.0.self_attn", attention_spec),
            wrapped("model.layers.1.linear_attn", gdn_spec),
            wrapped("model.layers.2.ple", replicated_spec),
            wrapped("model.layers.3.indexer_ring", indexer_ring_spec),
            wrapped("model.layers.3.compressed_key", compressed_key_spec),
        ],
    )

    with set_current_vllm_config(vllm_config), patch_worker_dependencies():
        connector = MooncakeConnector(
            vllm_config,
            KVConnectorRole.WORKER,
            kv_cache_config,
        )
        worker = connector.connector_worker

        assert worker._transfer_rule("model.layers.0.self_attn") == (
            False,
            worker.transfer_topo.total_num_kv_heads,
        )
        assert worker._transfer_rule("model.layers.1.linear_attn") == (False, None)
        assert worker._transfer_rule("model.layers.2.ple") == (True, None)
        assert worker._transfer_rule("model.layers.3.indexer_ring") == (True, None)
        assert worker._transfer_rule("model.layers.3.compressed_key") == (True, None)

        # At a TP size equal to the KV-head count, a rank holds one KV head,
        # as many as the single-head indexer ring; the ring stays replicated.
        worker.tp_size = worker.transfer_topo.total_num_kv_heads
        assert worker._transfer_rule("model.layers.3.indexer_ring") == (True, None)
        assert worker._transfer_rule("model.layers.3.compressed_key") == (True, None)

        worker.shutdown()
        worker.shutdown = noop_shutdown
        connector.connector_worker = None

    # P TP2 -> D TP4 with 2 KV heads: each consumer rank holds a whole replica
    # of the head its producer rank owns, which the head mapping copies whole.
    for remote_tp_rank in range(4):
        assert _compute_sender_transfer_plan(
            local_tp_rank=remote_tp_rank // 2,
            local_tp_size=2,
            remote_tp_rank=remote_tp_rank,
            remote_tp_size=4,
            local_kv_block_len=1000,
            remote_kv_block_len=1000,
            producer_cache_replicated=False,
            total_num_kv_heads=2,
        ) == (True, 0, 0, 1000)
    # Mamba/GDN states keep ratio slicing.
    assert _compute_sender_transfer_plan(
        local_tp_rank=0,
        local_tp_size=2,
        remote_tp_rank=1,
        remote_tp_size=4,
        local_kv_block_len=2000,
        remote_kv_block_len=1000,
        producer_cache_replicated=False,
    ) == (True, 1000, 0, 1000)


def test_validate_region_lengths_per_rule_for_replicated_consumer_attention():
    # P TP2 -> D TP4 with 2 KV heads. The worker validates each rule's regions
    # separately: attention by head mapping (one whole head per rank), Mamba/GDN
    # by TP ratio (local == 2x remote).
    attention_local = [
        TransferRegion("model.layers.0.self_attn", 0, 0x1000, 2000, 1000, 0)
    ]
    attention_remote = [
        TransferRegion("model.layers.0.self_attn", 0, 0x2000, 2000, 1000, 0)
    ]
    mamba_local = [
        TransferRegion("model.layers.1.linear_attn", 1, 0x5000, 2000, 2000, 1)
    ]
    mamba_remote = [
        TransferRegion("model.layers.1.linear_attn", 1, 0x6000, 2000, 1000, 1)
    ]
    assert (
        _validate_asymmetric_region_lengths(
            attention_local,
            attention_remote,
            local_tp_size=2,
            remote_tp_size=4,
            producer_cache_replicated=False,
            total_num_kv_heads=2,
        )
        is None
    )
    assert (
        _validate_asymmetric_region_lengths(
            mamba_local,
            mamba_remote,
            local_tp_size=2,
            remote_tp_size=4,
            producer_cache_replicated=False,
        )
        is None
    )
    mismatched_remote = [
        TransferRegion("model.layers.0.self_attn", 0, 0x2000, 2000, 500, 0)
    ]
    assert (
        _validate_asymmetric_region_lengths(
            attention_local,
            mismatched_remote,
            local_tp_size=2,
            remote_tp_size=4,
            producer_cache_replicated=False,
            total_num_kv_heads=2,
        )
        is not None
    )


def test_transfer_plan_shards_mamba_when_only_attention_is_replicated():
    # Falcon-H1 P TP4 -> D TP2: 2 attention KV heads replicate across the
    # producer's 4 ranks, but the 24-head Mamba2 states still shard. A global
    # producer-replication flag would make only producer ranks 0/2 send their
    # mamba regions and silently drop half the mamba heads on the consumer.
    #
    # Attention region: replicated producer -> ranks 0 and 2 send whole
    # copies, rank 1 stays silent.
    assert _compute_sender_transfer_plan(
        local_tp_rank=1,
        local_tp_size=4,
        remote_tp_rank=0,
        remote_tp_size=2,
        local_kv_block_len=1000,
        remote_kv_block_len=1000,
        producer_cache_replicated=True,
        total_num_kv_heads=2,
    ) == (False, 0, 0, 1000)
    assert _compute_sender_transfer_plan(
        local_tp_rank=2,
        local_tp_size=4,
        remote_tp_rank=1,
        remote_tp_size=2,
        local_kv_block_len=1000,
        remote_kv_block_len=1000,
        producer_cache_replicated=True,
        total_num_kv_heads=2,
    ) == (True, 0, 0, 1000)
    # Mamba region: producer not replicated -> every rank pushes its shard
    # into the paired consumer rank's region (rank 1 -> dst offset 1000).
    assert _compute_sender_transfer_plan(
        local_tp_rank=1,
        local_tp_size=4,
        remote_tp_rank=0,
        remote_tp_size=2,
        local_kv_block_len=1000,
        remote_kv_block_len=2000,
        producer_cache_replicated=False,
    ) == (True, 0, 1000, 1000)


def test_validate_region_lengths_shards_mamba_when_producer_attn_replicated():
    # P TP4 -> D TP2 with 2 KV heads: attention replicates on the producer
    # (4 > 2 KV heads) and copies whole heads, while the Mamba group still
    # shards, so its consumer region must be tp_ratio times the producer's. A
    # single engine-wide replication rule would skip the Mamba check.
    attention_local = [
        TransferRegion("model.layers.0.self_attn", 0, 0x1000, 2000, 1000, 0)
    ]
    attention_remote = [
        TransferRegion("model.layers.0.self_attn", 0, 0x2000, 2000, 1000, 0)
    ]
    mamba_local = [
        TransferRegion("model.layers.1.linear_attn", 1, 0x5000, 2000, 1000, 1)
    ]
    assert (
        _validate_asymmetric_region_lengths(
            attention_local,
            attention_remote,
            local_tp_size=4,
            remote_tp_size=2,
            producer_cache_replicated=True,
            total_num_kv_heads=2,
        )
        is None
    )
    assert (
        _validate_asymmetric_region_lengths(
            mamba_local,
            [TransferRegion("model.layers.1.linear_attn", 1, 0x6000, 2000, 2000, 1)],
            local_tp_size=4,
            remote_tp_size=2,
            producer_cache_replicated=False,
        )
        is None
    )
    err = _validate_asymmetric_region_lengths(
        mamba_local,
        [TransferRegion("model.layers.1.linear_attn", 1, 0x6000, 2000, 1000, 1)],
        local_tp_size=4,
        remote_tp_size=2,
        producer_cache_replicated=False,
    )
    assert err is not None and "TP ratio" in err


def test_hetero_tp_keeps_sharded_gdn_states_uncoalesced(monkeypatch):
    """Adjacent conv and ssm regions must not merge under heterogeneous TP,
    even when the attention KV is replicated: the ratio split of a merged
    region would cut across the conv/ssm boundary."""
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
        gdn_spec = kv_cache_config.kv_cache_groups[1].kv_cache_spec
        stride = 128
        packed = torch.empty(kv_cache_config.num_blocks * stride, dtype=torch.int8)
        gdn_cache = packed.view(-1, stride)[:, : gdn_spec.page_size_bytes]
        worker.register_kv_caches({"model.layers.1.linear_attn": gdn_cache})
        assert worker.kv_block_len_per_layer == [36, 8]
        worker._producer_cache_is_replicated = lambda: True

        # The consumer at twice the TP holds half of each state.
        remote_base = 0x100000
        meta = MooncakeXferMetadata(
            remote_hostname="consumer-host",
            remote_port=54321,
            remote_tp_size=worker.tp_size * 2,
            remote_tp_rank=0,
            req_blocks={},
            kv_caches_base_addr=[remote_base, remote_base + 18],
            block_lens=[stride, stride],
            kv_block_lens=[18, 4],
            registered_layer_names=["model.layers.1.linear_attn"] * 2,
            registered_layer_indices=[1, 1],
            registered_group_indices=[1, 1],
            registered_shared_group_ids=[[1], [1]],
            registered_row_offsets=[-1, -1],
        )
        local_regions, remote_regions, err = worker._prepare_transfer_regions(meta)
        assert err is None
        assert [region.kv_block_len for region in local_regions] == [36, 8]
        assert [region.kv_block_len for region in remote_regions] == [18, 4]

        worker.shutdown()
        worker.shutdown = noop_shutdown
        connector.connector_worker = None


def test_register_kv_caches_keeps_replicated_conv_state_whole(monkeypatch):
    """Under DS, only sharded conv states split into sub-projections; a
    TP-replicated short conv (e.g. a per-layer embedding cache) registers as
    one region, and the sub-projection split does not support it."""
    from vllm.model_executor.layers.mamba.mamba_utils import (
        get_conv_state_layout,
    )

    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
    monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", "DS")
    get_conv_state_layout.cache_clear()
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector",
        kv_role="kv_consumer",
    )
    spec = MambaSpec(
        block_size=vllm_config.cache_config.block_size,
        shapes=((4, 3),),
        dtypes=(torch.float16,),
        mamba_type=MambaAttentionBackendEnum.SHORT_CONV,
        tp_replicated=True,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["model.layers.0.ple"], spec)],
    )
    try:
        with set_current_vllm_config(vllm_config), patch_worker_dependencies():
            connector = MooncakeConnector(
                vllm_config,
                KVConnectorRole.WORKER,
                kv_cache_config,
            )
            worker = connector.connector_worker
            raw = torch.empty(16 * spec.page_size_bytes, dtype=torch.int8)
            cache = raw.view(16, spec.page_size_bytes)
            worker.register_kv_caches({"model.layers.0.ple": cache})

            assert worker.kv_caches_base_addr == [cache.data_ptr()]
            assert worker.kv_block_len_per_layer == [4 * 3 * 2]

            worker.shutdown()
            worker.shutdown = noop_shutdown
            connector.connector_worker = None
    finally:
        get_conv_state_layout.cache_clear()


def test_hetero_tp_sd_conv_layout_requires_a_derivable_split(monkeypatch):
    """SD conv states reshard per row and sub-projection, so heterogeneous TP
    needs a mamba type whose sub-projections can be derived."""
    from vllm.model_executor.layers.mamba.mamba_utils import (
        get_conv_state_layout,
    )

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
        try:
            monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", "SD")
            get_conv_state_layout.cache_clear()
            # The fixture's GDN conv state (6, 3) is DS-shaped; SD is (3, 6).
            worker._layer_specs["model.layers.1.linear_attn"] = MambaSpec(
                block_size=vllm_config.cache_config.block_size,
                shapes=((3, 6), (1, 2, 2)),
                dtypes=(torch.float16, torch.float16),
                mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
            )
            assert worker._hetero_tp_mamba_layout_error(worker.tp_size * 2) is None

            # A sharded short conv has no sub-projection split to reshard by.
            worker._layer_specs["model.layers.2.conv"] = MambaSpec(
                block_size=vllm_config.cache_config.block_size,
                shapes=((3, 4),),
                dtypes=(torch.float16,),
                mamba_type=MambaAttentionBackendEnum.SHORT_CONV,
            )
            err = worker._hetero_tp_mamba_layout_error(worker.tp_size * 2)
            assert err is not None and "model.layers.2.conv" in err
            # Homogeneous TP copies whole states, so any SD state is fine.
            assert worker._hetero_tp_mamba_layout_error(worker.tp_size) is None

            monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", "DS")
            get_conv_state_layout.cache_clear()
            assert worker._hetero_tp_mamba_layout_error(worker.tp_size * 2) is None
        finally:
            get_conv_state_layout.cache_clear()
            worker.shutdown()
            worker.shutdown = noop_shutdown
            connector.connector_worker = None


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

            fa_page_bytes = kv_cache_config.kv_cache_groups[
                0
            ].kv_cache_spec.page_size_bytes
            fa_cache = torch.empty((2, fa_page_bytes), dtype=torch.uint8)
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
                ) == (True, (d_rank % 2) * d_len, 0, d_len)
    finally:
        get_conv_state_layout.cache_clear()


# Global conv sub-projection dims per mamba type, and the head-sharded ssm
# state (heads, a, b). Mamba2's x:B:C is 2:1:1; GDN's Q:K:V is 1:1:2.
_SD_RESHARD_MODELS = {
    MambaAttentionBackendEnum.GDN_ATTN: ((8, 8, 16), (4, 4, 2)),
    MambaAttentionBackendEnum.MAMBA2: ((16, 8, 8), (4, 4, 2)),
}
# conv_kernel - 1 + num_spec, with one speculative token.
_SD_CONV_ROWS = 4


def _sd_reshard_spec(mamba_type, tp: int, block_size: int) -> MambaSpec:
    proj_dims, (heads, a, b) = _SD_RESHARD_MODELS[mamba_type]
    return MambaSpec(
        block_size=block_size,
        shapes=((_SD_CONV_ROWS, sum(proj_dims) // tp), (heads // tp, a, b)),
        dtypes=(torch.float16, torch.float16),
        mamba_type=mamba_type,
    )


def _sd_shard_page(conv, ssm, mamba_type, tp: int, rank: int) -> torch.Tensor:
    """One rank's page bytes: SD conv rows [proj0_r | proj1_r | ...], then
    its heads of the ssm state."""
    proj_dims, _ = _SD_RESHARD_MODELS[mamba_type]
    pieces = []
    start = 0
    for dim in proj_dims:
        shard = dim // tp
        pieces.append(conv[:, start + rank * shard : start + (rank + 1) * shard])
        start += dim
    heads = ssm.shape[0] // tp
    return torch.cat(
        [
            torch.cat(pieces, dim=1).contiguous().view(torch.int8).flatten(),
            ssm[rank * heads : (rank + 1) * heads]
            .contiguous()
            .view(torch.int8)
            .flatten(),
        ]
    )


async def _plan_and_apply_sd_reshard(
    worker: MooncakeConnectorWorker,
    *,
    p_tp: int,
    p_rank: int,
    producer_cache: torch.Tensor,
    consumer_pages: list[torch.Tensor],
    d_spec: MambaSpec,
    layer_name: str,
    requests: dict[str, tuple[int, int]],
) -> None:
    """Plan one producer rank's sends to its consumer ranks for all requests
    in one batch, and apply the planned copies."""
    d_tp = len(consumer_pages)
    worker.tp_size, worker.tp_rank = p_tp, p_rank
    worker.register_kv_caches({layer_name: producer_cache})
    assert worker._hetero_tp_mamba_layout_error(d_tp) is None
    local_regions = worker._get_transfer_regions(
        worker.kv_caches_base_addr,
        worker.block_len_per_layer,
        worker.kv_block_len_per_layer,
        worker.registered_layer_names,
        worker.registered_layer_indices,
        worker.registered_group_indices,
    )
    ratio = max(p_tp, d_tp) // min(p_tp, d_tp)
    d_ranks = (
        [p_rank // ratio]
        if p_tp >= d_tp
        else range(p_rank * ratio, (p_rank + 1) * ratio)
    )
    d_conv_bytes = prod(d_spec.shapes[0]) * get_dtype_size(d_spec.dtypes[0])
    for d_rank in d_ranks:
        consumer = consumer_pages[d_rank]
        block_len = consumer.stride(0)
        remote_regions = [
            TransferRegion(
                layer_name=layer_name,
                layer_index=0,
                base_addr=consumer.data_ptr() + offset,
                block_len=block_len,
                kv_block_len=length,
                group_index=0,
            )
            for offset, length in (
                (0, d_conv_bytes),
                (d_conv_bytes, d_spec.page_size_bytes - d_conv_bytes),
            )
        ]
        ready_reqs = [
            (
                req_id,
                SendBlockMeta(
                    p_req_id=f"p-{req_id}",
                    transfer_id=f"xfer-{req_id}",
                    local_block_ids=[[p_block]],
                    ready=asyncio.Event(),
                ),
            )
            for req_id, (p_block, _) in requests.items()
        ]
        xfer_meta = MooncakeXferMetadata(
            remote_hostname="consumer-host",
            remote_port=54321,
            remote_tp_size=d_tp,
            remote_tp_rank=d_rank,
            req_blocks={
                req_id: (f"xfer-{req_id}", [[d_block]])
                for req_id, (_, d_block) in requests.items()
            },
            kv_caches_base_addr=[],
            block_lens=[],
            kv_block_lens=[],
        )
        (
            src_ptrs,
            dst_ptrs,
            lengths,
            err_reqs,
            err_msg,
        ) = await worker._build_transfer_params(
            ready_reqs, xfer_meta, local_regions, remote_regions
        )
        assert err_reqs == [] and err_msg is None
        for src, dst, length in zip(src_ptrs, dst_ptrs, lengths):
            ctypes.memmove(dst, src, length)


@pytest.mark.cpu_test
@pytest.mark.asyncio
@pytest.mark.parametrize("mamba_type", list(_SD_RESHARD_MODELS))
@pytest.mark.parametrize("p_tp,d_tp", [(2, 1), (1, 2), (4, 2), (2, 4)])
async def test_hetero_tp_reshards_sd_conv_state(monkeypatch, mamba_type, p_tp, d_tp):
    """Heterogeneous TP with the SD conv layout: every consumer page equals
    its own rank's shard of the producer state, for every request of a batch.
    """
    from vllm.model_executor.layers.mamba.mamba_utils import (
        get_conv_state_layout,
    )

    monkeypatch.setenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT", "5")
    monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", "SD")
    get_conv_state_layout.cache_clear()
    # The planner does not depend on the role; a consumer worker registers
    # without a bootstrap server.
    vllm_config = create_vllm_config(
        kv_connector="MooncakeConnector",
        kv_role="kv_consumer",
    )
    block_size = vllm_config.cache_config.block_size
    layer_name = "model.layers.0.linear_attn"
    num_blocks = 8
    proj_dims, ssm_shape = _SD_RESHARD_MODELS[mamba_type]
    torch.manual_seed(0)
    global_conv = torch.randn(num_blocks, _SD_CONV_ROWS, sum(proj_dims)).half()
    global_ssm = torch.randn(num_blocks, *ssm_shape).half()

    def pages(tp: int, rank: int, fill: bool) -> torch.Tensor:
        spec = _sd_reshard_spec(mamba_type, tp, block_size)
        cache = torch.zeros(num_blocks, spec.page_size_bytes, dtype=torch.int8)
        if fill:
            for block in range(num_blocks):
                page = _sd_shard_page(
                    global_conv[block], global_ssm[block], mamba_type, tp, rank
                )
                cache[block, : page.numel()] = page
        return cache

    consumer_pages = [pages(d_tp, rank, fill=False) for rank in range(d_tp)]
    # (producer block, consumer block) per request, planned in one batch.
    requests = {"req-a": (3, 5), "req-b": (6, 2), "req-c": (1, 7)}
    try:
        for p_rank in range(p_tp):
            p_spec = _sd_reshard_spec(mamba_type, p_tp, block_size)
            kv_cache_config = KVCacheConfig(
                num_blocks=num_blocks,
                kv_cache_tensors=[],
                kv_cache_groups=[KVCacheGroupSpec([layer_name], p_spec)],
            )
            with set_current_vllm_config(vllm_config), patch_worker_dependencies():
                connector = MooncakeConnector(
                    vllm_config, KVConnectorRole.WORKER, kv_cache_config
                )
            worker = connector.connector_worker
            try:
                await _plan_and_apply_sd_reshard(
                    worker,
                    p_tp=p_tp,
                    p_rank=p_rank,
                    producer_cache=pages(p_tp, p_rank, fill=True),
                    consumer_pages=consumer_pages,
                    d_spec=_sd_reshard_spec(mamba_type, d_tp, block_size),
                    layer_name=layer_name,
                    requests=requests,
                )
            finally:
                worker.shutdown()
                worker.shutdown = noop_shutdown
                connector.connector_worker = None

        for d_rank, consumer in enumerate(consumer_pages):
            for p_block, d_block in requests.values():
                expected = _sd_shard_page(
                    global_conv[p_block], global_ssm[p_block], mamba_type, d_tp, d_rank
                )
                assert torch.equal(consumer[d_block, : expected.numel()], expected)
            untouched = sorted(
                set(range(num_blocks)) - {d for _, d in requests.values()}
            )
            assert not consumer[untouched].any()
    finally:
        get_conv_state_layout.cache_clear()
