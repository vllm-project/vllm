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
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import pytest
import torch

from tests.v1.attention.utils import dense_kv_cache_views
from vllm.config import set_current_vllm_config
from vllm.distributed.kv_transfer.kv_connector.utils import TransferTopology
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    KVConnectorRole,
    MooncakeConnector,
    MooncakeConnectorScheduler,
    MooncakeConnectorWorker,
    MooncakeXferMetadata,
    SendBlockMeta,
    TransferRegion,
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
from .utils import create_request, create_vllm_config, maybe_update_block_size


@pytest.fixture(autouse=True)
def conv_layout(monkeypatch):
    from vllm.model_executor.layers.mamba.mamba_utils import get_conv_state_layout

    monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", "DS")
    get_conv_state_layout.cache_clear()
    yield
    get_conv_state_layout.cache_clear()


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
        assert (
            worker.registered_layer_names
            == ["model.layers.0.self_attn"] + ["model.layers.1.linear_attn"] * 4
        )
        assert worker.kv_block_len_per_layer == [fa_spec.page_size_bytes, 12, 12, 12, 8]
        assert worker.registered_group_indices == [0, 1, 1, 1, 1]
        assert worker.kv_caches_base_addr == [fa_cache.data_ptr()] + [
            gdn_cache.data_ptr() + offset for offset in (0, 12, 24, 36)
        ]

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
        local_replicates_kv_cache=False, total_num_kv_heads=1
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

        assert worker.kv_caches_base_addr == [fa_cache.data_ptr()] + [
            gdn_cache.data_ptr() + offset for offset in (0, 12, 24, 36)
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


_ATTN_LAYER = "model.layers.0.self_attn"
_GDN_LAYER = "model.layers.1.linear_attn"
_CANARY = 205


def _global_gdn_state(block):
    # Different projections and history rows expose accidental whole-page slices.
    conv = torch.arange(3 * 32, dtype=torch.float16).reshape(3, 32) + block * 100
    ssm = torch.arange(4 * 4 * 2, dtype=torch.float32).reshape(4, 4, 2) + block * 1000
    return conv, ssm


def _gdn_payload(block, rank, tp, layout, replicated=False):
    conv, ssm = _global_gdn_state(block)
    if not replicated:
        conv = torch.cat(
            [part.chunk(tp, dim=1)[rank] for part in conv.split((8, 8, 16), dim=1)],
            dim=1,
        )
        ssm = ssm.chunk(tp, dim=0)[rank]
    if layout == "DS":
        conv = conv.t()
    return torch.cat(
        [state.contiguous().view(torch.uint8).flatten() for state in (conv, ssm)]
    )


def _mamba1_payload(block, rank, tp, layout):
    conv = torch.arange(3 * 32, dtype=torch.float16).reshape(3, 32) + block * 100
    ssm = torch.arange(32 * 4, dtype=torch.float32).reshape(32, 4) + block * 1000
    conv = conv.chunk(tp, dim=1)[rank]
    ssm = ssm.chunk(tp, dim=0)[rank]
    if layout == "DS":
        conv = conv.t()
    return torch.cat(
        [state.contiguous().view(torch.uint8).flatten() for state in (conv, ssm)]
    )


def _rawpage_worker(
    tp,
    rank,
    layout,
    *,
    replicated=False,
    second_layer=False,
    mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    extra_state=False,
):
    shard_tp = 1 if replicated else tp
    width = 32 // shard_tp
    conv_shape = (3, width) if layout == "SD" else (width, 3)
    if mamba_type == MambaAttentionBackendEnum.MAMBA1:
        # Mamba1 conv state holds x only; the SSM state is (x, state_size).
        shapes = (conv_shape, (width, 4))
    else:
        shapes = (conv_shape, (4 // shard_tp, 4, 2))
    dtypes = (torch.float16, torch.float32)
    if extra_state:
        shapes += ((1,),)
        dtypes += (torch.float16,)
    spec = MambaSpec(
        block_size=16,
        shapes=shapes,
        dtypes=dtypes,
        mamba_type=mamba_type,
        tp_replicated=replicated,
        page_size_padded=1024,
    )
    groups = [
        KVCacheGroupSpec(
            [_ATTN_LAYER],
            FullAttentionSpec(
                block_size=16,
                num_kv_heads=max(1, 2 // tp),
                head_size=2,
                dtype=torch.float16,
            ),
        ),
        KVCacheGroupSpec([_GDN_LAYER], spec),
    ]
    if second_layer:
        groups.append(
            KVCacheGroupSpec(
                ["model.layers.2.linear_attn"],
                MambaSpec(
                    block_size=16,
                    shapes=((3, 16) if layout == "SD" else (16, 3), (2, 4, 2)),
                    dtypes=(torch.float16, torch.float32),
                    mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
                    tp_replicated=True,
                    page_size_padded=1024,
                ),
            )
        )
    worker = object.__new__(MooncakeConnectorWorker)
    worker.shutdown = noop_shutdown
    worker.tp_size, worker.tp_rank, worker.pp_size = tp, rank, 1
    worker.use_mla = False
    worker.is_kv_consumer = True  # Avoid starting networking threads.
    worker.engine = SimpleNamespace(batch_register_memory=lambda *_: 0)
    worker._physical_blocks_per_logical_kv_block = 1
    worker._prepared_transfer_regions = {}
    worker._layout_by_peer = OrderedDict()
    worker._acked_layout_peers = set()
    worker.kv_cache_config = KVCacheConfig(
        num_blocks=6, kv_cache_tensors=[], kv_cache_groups=groups
    )
    worker._layer_specs = {
        name: group.kv_cache_spec for group in groups for name in group.layer_names
    }
    worker._layer_group_indices = {
        name: i for i, group in enumerate(groups) for name in group.layer_names
    }
    worker.transfer_topo = TransferTopology(
        tp_rank=rank,
        tp_size=tp,
        block_size=16,
        engine_id="rawpage",
        is_mla=False,
        is_mamba=True,
        total_num_kv_heads=2,
        attn_backends=[],
    )
    allocations, caches = {}, {}
    for name, layer_spec in worker._layer_specs.items():
        page = layer_spec.page_size_bytes
        raw = torch.full((6, page + 80), _CANARY, dtype=torch.uint8)
        # The page starts inside its storage, and its stride includes padding.
        caches[name] = raw[:, 32 : 32 + page].view(6, 1, 1, page)
        allocations[name] = raw
    worker.register_kv_caches(caches)
    return worker, allocations, caches


def _wire_metadata(worker, request_blocks):
    meta = MooncakeXferMetadata(
        remote_hostname="consumer",
        remote_port=1234,
        remote_tp_size=worker.tp_size,
        remote_tp_rank=worker.tp_rank,
        req_blocks={"request": ("transfer", request_blocks)},
        kv_caches_base_addr=worker.kv_caches_base_addr,
        block_lens=worker.block_len_per_layer,
        kv_block_lens=worker.kv_block_len_per_layer,
        registered_layer_names=worker.registered_layer_names,
        registered_layer_indices=worker.registered_layer_indices,
        registered_group_indices=worker.registered_group_indices,
        registered_shared_group_ids=[list(ids) for ids in worker.region_shared_groups],
        registered_row_offsets=worker.region_row_offsets,
        mamba_transfer_geometry=worker._get_mamba_transfer_geometry(),
    )
    return msgspec.msgpack.decode(
        msgspec.msgpack.encode(meta), type=MooncakeXferMetadata
    )


def _attention_payload(block, rank, tp):
    heads = torch.arange(2 * 128, dtype=torch.uint8).reshape(2, 128) + block
    if tp > 2:
        return heads[rank // (tp // 2)].flatten()
    return heads.chunk(tp, dim=0)[rank].flatten()


@pytest.mark.parametrize("layout", ["SD", "DS"])
@pytest.mark.parametrize(("p_tp", "d_tp"), [(1, 2), (2, 1), (4, 2), (2, 2)])
@pytest.mark.asyncio
async def test_gdn_rawpages_reshard_states_and_preserve_canaries(
    monkeypatch, layout, p_tp, d_tp
):
    """Transfer real descriptors and compare with independently sharded states."""
    monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", layout)
    producers = [
        _rawpage_worker(p_tp, rank, layout, second_layer=True) for rank in range(p_tp)
    ]
    consumers = [
        _rawpage_worker(d_tp, rank, layout, second_layer=True) for rank in range(d_tp)
    ]
    second = "model.layers.2.linear_attn"
    for rank, (_, _, caches) in enumerate(producers):
        for block in range(6):
            caches[_ATTN_LAYER][block].flatten().copy_(
                _attention_payload(block, rank, p_tp)
            )
            state = _gdn_payload(block, rank, p_tp, layout)
            caches[_GDN_LAYER][block].flatten()[: state.numel()].copy_(state)
            caches[second][block].flatten()[:160].copy_(
                torch.arange(160, dtype=torch.uint8) + block
            )
    for d_rank, (consumer, allocations, _) in enumerate(consumers):
        expected = {name: raw.clone() for name, raw in allocations.items()}
        for source, destination in zip([2, 3], [1, 2]):
            payload = _attention_payload(source, d_rank, d_tp)
            expected[_ATTN_LAYER][destination, 32 : 32 + payload.numel()] = payload
        for source, destination in zip([3, 4], [2, 3]):
            payload = _gdn_payload(source, d_rank, d_tp, layout)
            expected[_GDN_LAYER][destination, 32 : 32 + payload.numel()] = payload
            expected[second][destination, 32:192] = (
                torch.arange(160, dtype=torch.uint8) + source
            )
        meta = _wire_metadata(consumer, [[1, 2], [2, 3], [2, 3]])
        for producer, _, _ in producers:
            if d_rank not in producer.transfer_topo.handshake_target_ranks(d_tp):
                continue
            local, remote, error = producer._prepare_transfer_regions(meta)
            assert error is None
            send = SendBlockMeta(
                p_req_id="producer",
                transfer_id="transfer",
                local_block_ids=[[2, 3], [3, 4], [3, 4]],
                ready=asyncio.Event(),
            )
            src, dst, lengths, errors, error = await producer._build_transfer_params(
                [("request", send)], meta, local, remote
            )
            assert not errors and error is None
            for src_ptr, dst_ptr, length in zip(src, dst, lengths):
                ctypes.memmove(dst_ptr, src_ptr, length)
        for name, raw in allocations.items():
            assert torch.equal(raw, expected[name]), name


@pytest.mark.parametrize("layout", ["SD", "DS"])
@pytest.mark.parametrize(("p_tp", "d_tp"), [(1, 2), (2, 1)])
@pytest.mark.asyncio
async def test_mamba1_rawpages_reshard_states_and_preserve_canaries(
    monkeypatch, layout, p_tp, d_tp
):
    """Mamba1 pages decompose into one conv shard plus the SSM state."""
    monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", layout)
    producers = [
        _rawpage_worker(p_tp, rank, layout, mamba_type=MambaAttentionBackendEnum.MAMBA1)
        for rank in range(p_tp)
    ]
    consumers = [
        _rawpage_worker(d_tp, rank, layout, mamba_type=MambaAttentionBackendEnum.MAMBA1)
        for rank in range(d_tp)
    ]
    for rank, (_, _, caches) in enumerate(producers):
        for block in range(6):
            caches[_ATTN_LAYER][block].flatten().copy_(
                _attention_payload(block, rank, p_tp)
            )
            state = _mamba1_payload(block, rank, p_tp, layout)
            caches[_GDN_LAYER][block].flatten()[: state.numel()].copy_(state)
    for d_rank, (consumer, allocations, _) in enumerate(consumers):
        expected = {name: raw.clone() for name, raw in allocations.items()}
        for source, destination in zip([2, 3], [1, 2]):
            payload = _attention_payload(source, d_rank, d_tp)
            expected[_ATTN_LAYER][destination, 32 : 32 + payload.numel()] = payload
        for source, destination in zip([3, 4], [2, 3]):
            payload = _mamba1_payload(source, d_rank, d_tp, layout)
            expected[_GDN_LAYER][destination, 32 : 32 + payload.numel()] = payload
        meta = _wire_metadata(consumer, [[1, 2], [2, 3]])
        for producer, _, _ in producers:
            if d_rank not in producer.transfer_topo.handshake_target_ranks(d_tp):
                continue
            local, remote, error = producer._prepare_transfer_regions(meta)
            assert error is None
            send = SendBlockMeta(
                p_req_id="producer",
                transfer_id="transfer",
                local_block_ids=[[2, 3], [3, 4]],
                ready=asyncio.Event(),
            )
            src, dst, lengths, errors, error = await producer._build_transfer_params(
                [("request", send)], meta, local, remote
            )
            assert not errors and error is None
            for src_ptr, dst_ptr, length in zip(src, dst, lengths):
                ctypes.memmove(dst_ptr, src_ptr, length)
        for name, raw in allocations.items():
            assert torch.equal(raw, expected[name]), name


@pytest.mark.parametrize("layout", ["SD", "DS"])
@pytest.mark.asyncio
async def test_replicated_gdn_has_one_writer_independent_of_attention(
    monkeypatch, layout
):
    monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", layout)
    consumer, allocations, _ = _rawpage_worker(2, 0, layout, replicated=True)
    meta = _wire_metadata(consumer, [[], [2]])
    writes = []
    for rank in range(4):
        producer, _, caches = _rawpage_worker(4, rank, layout, replicated=True)
        if consumer.tp_rank not in producer.transfer_topo.handshake_target_ranks(2):
            continue
        payload = _gdn_payload(3, 0, 1, layout, replicated=True)
        caches[_GDN_LAYER][3].flatten()[: payload.numel()].copy_(payload)
        local, remote, error = producer._prepare_transfer_regions(meta)
        assert error is None
        send = SendBlockMeta(
            p_req_id="producer",
            transfer_id="transfer",
            local_block_ids=[[], [3]],
            ready=asyncio.Event(),
        )
        src, dst, lengths, errors, error = await producer._build_transfer_params(
            [("request", send)], meta, local, remote
        )
        assert not errors and error is None
        writes.extend(lengths)
        for src_ptr, dst_ptr, length in zip(src, dst, lengths):
            ctypes.memmove(dst_ptr, src_ptr, length)
    assert sum(writes) == payload.numel()
    expected = torch.full_like(allocations[_GDN_LAYER], _CANARY)
    expected[2, 32 : 32 + payload.numel()] = payload
    assert torch.equal(allocations[_GDN_LAYER], expected)


@pytest.mark.cpu_test
def test_equal_tp_mamba_merge_needs_no_peer_geometry():
    """Equal TP keeps the pre-geometry contract and merges sub-regions."""
    producer, _, _ = _rawpage_worker(2, 0, "DS")
    consumer, _, _ = _rawpage_worker(2, 0, "DS")
    meta = _wire_metadata(consumer, [[], [1]])
    meta.mamba_transfer_geometry = {}

    local, remote, error = producer._prepare_transfer_regions(meta)

    assert error is None
    mamba_regions = [region for region in local if region.group_index == 1]
    assert len(mamba_regions) == 1
    assert mamba_regions[0].kv_block_len == 160


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("mamba_type", "extra_state", "expected_error"),
    [
        (MambaAttentionBackendEnum.MAMBA2, False, "unsupported Mamba state type"),
        (MambaAttentionBackendEnum.GDN_ATTN, True, "requires conv and SSM states"),
    ],
)
def test_unequal_tp_rejects_unsupported_mamba_states(
    mamba_type, extra_state, expected_error
):
    """Unknown state layouts must fail before producing transfer descriptors."""
    producer, _, _ = _rawpage_worker(
        1, 0, "DS", mamba_type=mamba_type, extra_state=extra_state
    )
    consumer, _, _ = _rawpage_worker(
        2, 0, "DS", mamba_type=mamba_type, extra_state=extra_state
    )
    local, remote, error = producer._prepare_transfer_regions(
        _wire_metadata(consumer, [[], [1]])
    )
    assert local == remote == []
    assert expected_error in error


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    [
        (None, None, "omitted Mamba transfer geometry"),
        ("conv_layout", "SD", "incompatible Mamba transfer geometry"),
        (
            "dtypes",
            ["torch.float32", "torch.float32"],
            "incompatible Mamba transfer geometry",
        ),
        ("tp_replicated", True, "incompatible Mamba transfer geometry"),
        ("shapes", [[16, 4], [2, 4, 2]], "incompatible Mamba state shapes"),
    ],
    ids=["old-peer", "layout", "dtype", "replication", "shape"],
)
def test_prepared_regions_revalidate_changed_peer_geometry(
    field, value, expected_error
):
    """A cached valid peer must not hide missing or incompatible state geometry."""
    producer, _, _ = _rawpage_worker(1, 0, "DS")
    consumer, _, _ = _rawpage_worker(2, 0, "DS")
    meta = _wire_metadata(consumer, [[], [1]])
    prepared = producer._prepare_transfer_regions(meta)
    assert prepared[2] is None
    geometry = meta.mamba_transfer_geometry[_GDN_LAYER]
    if field is None:
        meta.mamba_transfer_geometry = {}
    else:
        meta.mamba_transfer_geometry[_GDN_LAYER] = msgspec.structs.replace(
            geometry, **{field: value}
        )
    local, remote, error = producer._prepare_transfer_regions(meta)
    assert local == remote == []
    assert expected_error in error
    meta.mamba_transfer_geometry = {_GDN_LAYER: geometry}
    restored = producer._prepare_transfer_regions(meta)
    assert restored[0] is prepared[0]
    assert restored[1] is prepared[1]
    assert restored[2] is None
