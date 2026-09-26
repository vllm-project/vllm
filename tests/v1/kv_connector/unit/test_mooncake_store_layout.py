# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Mooncake Store payload layouts."""

import ctypes
import random

import pytest
import torch
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import BlockHash

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    ChunkedTokenDatabase,
    KeyMetadata,
    LBHNCStoreLayout,
    LBNHCStoreLayout,
    PoolKey,
    RankLocalStoreLayout,
    endpoint_neutral_region_metadata,
)

BLOCK_SIZE = 128


def _rank_local_layout(num_regions: int, num_block_lens: int) -> RankLocalStoreLayout:
    metadata = KeyMetadata("test-model", 1, 0, 0, 0)
    layout = RankLocalStoreLayout(metadata, BLOCK_SIZE, BLOCK_SIZE)
    layout.set_kv_caches_base_addr(
        [0x7F00_0000_0000 + i * (1 << 30) for i in range(num_regions)]
    )
    layout.set_block_len([30_208 + 512 * i for i in range(num_block_lens)])
    return layout


def _rank_local_reference(
    layout: RankLocalStoreLayout, start: int, end: int, block_ids: list[int]
) -> tuple[list[int], list[int], int]:
    block_id = block_ids[start // layout.block_size]
    length = len(layout.block_len)
    addrs = [
        base_addr + block_id * layout.block_len[index % length]
        for index, base_addr in enumerate(layout.kv_caches_base_addr)
    ]
    sizes = [
        layout.block_len[index % length] * cdiv(end - start, layout.block_size)
        for index in range(len(layout.kv_caches_base_addr))
    ]
    return addrs, sizes, block_id


@pytest.mark.parametrize("num_regions,num_block_lens", [(96, 96), (96, 2), (1, 1)])
def test_rank_local_descriptors_match_reference(num_regions: int, num_block_lens: int):
    layout = _rank_local_layout(num_regions, num_block_lens)
    rng = random.Random(0)
    block_ids = [rng.randrange(0, 1 << 20) for _ in range(300)]
    chunks = []
    block = 0
    while block < len(block_ids) - 4:
        span = rng.choice([1, 1, 1, 2, 4])
        chunks.append((block * BLOCK_SIZE, (block + span) * BLOCK_SIZE))
        block += span + rng.choice([0, 1])

    addrs, sizes, selected_blocks = layout.prepare_values(
        chunks, block_ids, [0] * len(chunks)
    )

    for chunk, chunk_addrs, chunk_sizes, block_id in zip(
        chunks, addrs, sizes, selected_blocks, strict=True
    ):
        assert (chunk_addrs, chunk_sizes, block_id) == _rank_local_reference(
            layout, *chunk, block_ids
        )
        assert all(type(addr) is int for addr in chunk_addrs)
        assert type(block_id) is int


def test_rank_local_descriptors_handle_empty_and_invalid_chunks():
    layout = _rank_local_layout(4, 4)
    assert layout.prepare_values([], [1, 2, 3], []) == ([], [], [])
    with pytest.raises(AssertionError):
        layout.prepare_values([(0, BLOCK_SIZE + 1)], [0, 1], [0])


def test_semantic_region_key_is_endpoint_neutral():
    prefill = KeyMetadata(
        "test-model",
        tp_rank=2,
        pcp_rank=0,
        dcp_rank=2,
        pp_rank=0,
        group_id=1,
        cache_prefix="experiment",
        store_namespace="@schema:v1",
    )
    decode = KeyMetadata(
        "test-model",
        tp_rank=2,
        pcp_rank=0,
        dcp_rank=2,
        pp_rank=1,
        group_id=5,
        cache_prefix="experiment",
        store_namespace="@schema:v1",
    )

    prefill_region = endpoint_neutral_region_metadata(prefill, "layer.4:target_conv")
    decode_region = endpoint_neutral_region_metadata(decode, "layer.4:target_conv")

    assert (
        PoolKey(prefill_region, "abc").to_string()
        == PoolKey(decode_region, "abc").to_string()
    )
    assert (
        "@pp_rank:-1@group:-1@region:layer.4:target_conv@abc"
        in PoolKey(prefill_region, "abc").to_string()
    )


def test_legacy_key_format_is_unchanged_without_region_id():
    metadata = KeyMetadata("test-model", 1, 2, 3, 4, group_id=5)
    assert (
        PoolKey(metadata, "abc").to_string()
        == "test-model@tp_rank:1@pcp2@dcp3@pp_rank:4@group:5@abc"
    )


def test_semantic_region_database_separates_content_length_from_stride():
    database = ChunkedTokenDatabase(
        KeyMetadata("test-model", 1, 0, 1, 1, group_id=7),
        block_size=16,
    )
    region = ChunkedTokenDatabase.from_semantic_region(
        database,
        region_id="layer.9:base_recurrent",
        base_addr=0x1000,
        block_stride=4096,
        content_len=768,
    )

    assert region.key_for(BlockHash(b"hash")).startswith(
        "test-model@tp_rank:1@pcp0@dcp1@pp_rank:-1@group:-1"
        "@region:layer.9:base_recurrent@"
    )
    assert region.prepare_value_for_block(3) == ([0x1000 + 3 * 4096], [768])

    region.store_layout.set_block_stride([0])
    assert region.prepare_value_for_block(3) == ([0x1000], [768])


@pytest.mark.parametrize(
    ("block_stride", "content_len"),
    [(63, 64), (0, 0), (-1, 1)],
)
def test_semantic_region_database_rejects_invalid_geometry(
    block_stride: int, content_len: int
):
    database = ChunkedTokenDatabase(KeyMetadata("test-model", 0, 0, 0, 0), 16)
    with pytest.raises(ValueError, match="stride"):
        ChunkedTokenDatabase.from_semantic_region(
            database,
            region_id="layer.0:page",
            base_addr=0x1000,
            block_stride=block_stride,
            content_len=content_len,
        )


@pytest.mark.parametrize(
    ("layout_cls", "store_format"),
    [
        (LBHNCStoreLayout, "tp_shared_lbhnc"),
        (LBNHCStoreLayout, "tp_shared_lbnhc"),
    ],
)
def test_tp_shared_layout_owns_store_namespace(layout_cls, store_format):
    assert (
        layout_cls.shared_namespace(4, 2)
        == f"@store_tp:4@store_pp:2@store_format:{store_format}"
    )


def test_rank_local_database_api_rejects_tp_shared_layout():
    metadata = KeyMetadata("test-model", 0, 0, 0, 0)
    layout = LBHNCStoreLayout(
        metadata,
        block_size=16,
        hash_block_size=16,
        local_tp_size=2,
        store_tp_size=4,
        tp_rank=0,
        num_kv_heads=8,
    )
    database = ChunkedTokenDatabase(metadata, 16, store_layout=layout)

    with pytest.raises(RuntimeError, match="rank-local"):
        database.prepare_value_for_block(0)


@pytest.mark.parametrize(
    ("layout_cls", "producer_strides", "consumer_strides"),
    [
        (LBHNCStoreLayout, (128, 64, 4, 1), (256, 64, 4, 1)),
        (LBNHCStoreLayout, (128, 4, 8, 1), (256, 4, 16, 1)),
    ],
)
def test_tp_shared_layout_round_trip_across_tp_sizes(
    layout_cls, producer_strides, consumer_strides
):
    block_size = 16
    shape = (1, 2, block_size, 4)
    stored: dict[int, bytes] = {}
    producer_tensors = []

    for tp_rank in range(4):
        tensor = torch.empty_strided(shape, producer_strides, dtype=torch.float16)
        tensor.copy_(
            torch.arange(128, dtype=torch.float16).view(shape) + tp_rank * 1000
        )
        producer_tensors.append(tensor)
        metadata = KeyMetadata("test-model", tp_rank, 0, 0, 0)
        layout = layout_cls(
            metadata,
            block_size,
            block_size,
            local_tp_size=4,
            store_tp_size=4,
            tp_rank=tp_rank,
            num_kv_heads=8,
        )
        layout.register_kv_caches([tensor], 1)
        addrs, sizes, _ = layout.prepare_values([(0, block_size)], [0], [tp_rank])
        stored[tp_rank] = b"".join(
            ctypes.string_at(addr, size)
            for addr, size in zip(addrs[0], sizes[0], strict=True)
        )

    for tp_rank in range(2):
        tensor = torch.empty_strided(
            (1, 4, block_size, 4), consumer_strides, dtype=torch.float16
        )
        tensor.zero_()
        metadata = KeyMetadata("test-model", tp_rank, 0, 0, 0)
        layout = layout_cls(
            metadata,
            block_size,
            block_size,
            local_tp_size=2,
            store_tp_size=4,
            tp_rank=tp_rank,
            num_kv_heads=8,
        )
        layout.register_kv_caches([tensor], 1)
        shard_ids = layout.local_shard_ids
        addrs, sizes, _ = layout.prepare_values(
            [(0, block_size)] * len(shard_ids), [0], shard_ids
        )
        for shard_id, shard_addrs, shard_sizes in zip(
            shard_ids, addrs, sizes, strict=True
        ):
            offset = 0
            for addr, size in zip(shard_addrs, shard_sizes, strict=True):
                ctypes.memmove(addr, stored[shard_id][offset : offset + size], size)
                offset += size

        expected = torch.cat(producer_tensors[tp_rank * 2 : tp_rank * 2 + 2], dim=1)
        torch.testing.assert_close(tensor, expected)
