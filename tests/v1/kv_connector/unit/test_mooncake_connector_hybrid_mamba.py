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

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
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
    MambaSpec,
)

from .utils import create_request


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
    block_size = 16
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        kv_transfer_config=SimpleNamespace(kv_role=kv_role),
        scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
    )
    return MooncakeConnectorScheduler(
        vllm_config=vllm_config,
        engine_id="test-engine",
        kv_cache_config=make_hybrid_gdn_kv_cache_config(block_size),
    )


def make_hybrid_gdn_worker(kv_role: str = "kv_producer") -> MooncakeConnectorWorker:
    kv_cache_config = make_hybrid_gdn_kv_cache_config(block_size=16)
    worker = object.__new__(MooncakeConnectorWorker)
    worker.vllm_config = SimpleNamespace(speculative_config=None)
    worker.model_config = SimpleNamespace(
        get_total_num_hidden_layers=lambda: 2,
    )
    worker.use_mla = False
    worker.tp_rank = 0
    worker.tp_size = 1
    worker.is_kv_producer = kv_role == "kv_producer"
    worker.is_kv_consumer = kv_role == "kv_consumer"
    worker.engine = SimpleNamespace(batch_register_memory=lambda *_: 0)
    worker.transfer_topo = SimpleNamespace(
        get_transfer_cache_regions=lambda cache, layer_spec: [cache],
        virtually_split_kv_in_blocks=False,
        local_replicates_kv_cache=False,
    )
    worker.block_len_per_layer = []
    worker.kv_block_len_per_layer = []
    worker.registered_layer_names = []
    worker.registered_layer_indices = []
    worker.registered_group_indices = []
    worker.registered_layer_aliases = []
    worker.registered_layer_index_aliases = []
    worker.registered_logical_group_indices = []
    worker.registered_alias_group_indices = []
    worker.seen_base_addresses = []
    worker.kv_caches_base_addr = []
    worker.device_kv_caches = {}
    worker.kv_cache_config = kv_cache_config
    worker._physical_blocks_per_logical_kv_block = 1
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
    worker._layer_logical_group_indices = {
        layer_name: [group_index]
        for group_index, group in enumerate(kv_cache_config.kv_cache_groups)
        for layer_name in group.layer_names
    }
    worker.shutdown = noop_shutdown
    return worker


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


def test_register_kv_caches_emits_fa_and_gdn_regions():
    worker = make_hybrid_gdn_worker(kv_role="kv_consumer")
    fa_cache = torch.empty((2, 2, 11), dtype=torch.float16)
    gdn_conv_state = torch.empty((2, 22), dtype=torch.float16)
    gdn_ssm_state = torch.empty((2, 4), dtype=torch.float16)

    worker.register_kv_caches(
        {
            "model.layers.0.self_attn": fa_cache,
            "model.layers.1.linear_attn": (gdn_conv_state, gdn_ssm_state),
        }
    )

    assert worker.kv_cache_config.has_mamba_layers is True
    assert worker.registered_layer_names == [
        "model.layers.0.self_attn",
        "model.layers.1.linear_attn",
    ]
    assert worker.registered_group_indices == [0, 1]
    assert worker.kv_caches_base_addr == [
        fa_cache.data_ptr(),
        gdn_conv_state.data_ptr(),
    ]


def test_register_kv_caches_deduplicates_shared_backing_memory():
    worker = make_hybrid_gdn_worker(kv_role="kv_consumer")
    backing = torch.empty((4, 64), dtype=torch.float16)
    fa_cache = backing[:2, :16]
    gdn_conv_state = backing[:3]
    gdn_ssm_state = torch.empty((3, 4), dtype=torch.float16)

    with patch.object(
        worker.engine, "batch_register_memory", return_value=0
    ) as batch_register_memory:
        worker.register_kv_caches(
            {
                "model.layers.0.self_attn": fa_cache,
                "model.layers.1.linear_attn": (gdn_conv_state, gdn_ssm_state),
            }
        )

    assert worker.kv_caches_base_addr == [
        fa_cache.data_ptr(),
        gdn_conv_state.data_ptr(),
    ]
    batch_register_memory.assert_called_once()
    registered_ptrs, registered_lens = batch_register_memory.call_args[0]
    assert registered_ptrs == [backing.data_ptr()]
    assert registered_lens == [backing.untyped_storage().nbytes()]


def test_hybrid_gdn_transfer_params_preserve_group_identity():
    worker = make_hybrid_gdn_worker(kv_role="kv_producer")
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


def test_logical_to_kernel_block_ids_expands_fa_not_gdn():
    worker = object.__new__(MooncakeConnectorWorker)
    worker.shutdown = noop_shutdown
    worker._physical_blocks_per_logical_kv_block = 17
    worker.kv_cache_config = make_hybrid_gdn_kv_cache_config(block_size=544)

    block_ids = [[2], [2]]
    kernel_block_ids = worker._logical_to_kernel_block_ids(block_ids)

    assert kernel_block_ids == [list(range(34, 51)), [2]]


def test_hybrid_gdn_splits_fa_regions_but_keeps_gdn_state_whole():
    worker = make_hybrid_gdn_worker(kv_role="kv_producer")
    worker.transfer_topo = SimpleNamespace(virtually_split_kv_in_blocks=True)
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
        (0, 0x1040, 0x40),
        (1, 0x2000, 0x100),
    ]
