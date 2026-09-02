# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Mooncake Store payload layouts."""

import ctypes
import os
import random

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import (
    AttentionStoreLayout,
    BHLNCStoreLayout,
    BLHNCStoreLayout,
    BLNHCStoreLayout,
    ChunkedTokenDatabase,
    KeyMetadata,
    LBHNCStoreLayout,
    LBNHCStoreLayout,
    LHBNCStoreLayout,
    MambaStoreLayout,
    RankLocalStoreLayout,
)
from vllm.model_executor.layers.mamba.mamba_utils import get_conv_state_layout
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_layout import KVCacheLayout

BLOCK_SIZE = 128


@pytest.fixture(autouse=True)
def _reset_conv_state_layout_cache():
    get_conv_state_layout.cache_clear()
    yield
    get_conv_state_layout.cache_clear()


def _attention_specs(
    num_layers: int, block_size: int, local_heads: int, content_size: int = 4
):
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    return tuple(
        FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=local_heads,
            head_size=content_size // 2,
            dtype=torch.float16,
        )
        for _ in range(num_layers)
    )


def _make_state_store_layout(
    spec, *, local_tp_size: int, tp_rank: int
) -> tuple[MambaStoreLayout, torch.Tensor]:
    tensor = torch.empty((2, 1, 1, spec.state_content_size_bytes), dtype=torch.uint8)
    metadata = KeyMetadata(
        "test-model",
        tp_rank,
        0,
        0,
        0,
        group_id=1,
        store_namespace=(
            "@store_tp:4@store_pp:1@store_format:mamba_state@store_schema:test"
        ),
    )
    layout = MambaStoreLayout(
        metadata,
        block_size=16,
        hash_block_size=16,
        local_tp_size=local_tp_size,
        store_tp_size=4,
        tp_rank=tp_rank,
        layer_specs=(spec,),
    )
    layout.register_kv_caches([tensor], 2)
    return layout, tensor


def _make_gdn_store_layout(
    *, local_tp_size: int, tp_rank: int
) -> tuple[MambaStoreLayout, torch.Tensor]:
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.kv_cache_interface import MambaSpec

    local_factor = 4 // local_tp_size
    conv_shape = (6 * local_factor, 3)
    if os.environ.get("VLLM_SSM_CONV_STATE_LAYOUT") == "SD":
        conv_shape = conv_shape[::-1]
    spec = MambaSpec(
        block_size=16,
        shapes=(conv_shape, (local_factor, 2, 2)),
        dtypes=(torch.uint8, torch.uint8),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )
    return _make_state_store_layout(spec, local_tp_size=local_tp_size, tp_rank=tp_rank)


def _make_mamba2_store_layout(
    *, local_tp_size: int, tp_rank: int
) -> tuple[MambaStoreLayout, torch.Tensor]:
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.kv_cache_interface import MambaSpec

    local_heads = 4 // local_tp_size
    effective_groups = local_tp_size
    conv_shape = ((8 + 2 * effective_groups * 2) // local_tp_size, 3)
    if os.environ.get("VLLM_SSM_CONV_STATE_LAYOUT") == "SD":
        conv_shape = conv_shape[::-1]
    spec = MambaSpec(
        block_size=16,
        shapes=(
            conv_shape,
            (local_heads, 2, 2),
        ),
        dtypes=(torch.uint8, torch.uint8),
        mamba_type=MambaAttentionBackendEnum.MAMBA2,
    )
    return _make_state_store_layout(spec, local_tp_size=local_tp_size, tp_rank=tp_rank)


def _descriptors_for_block(
    layout: MambaStoreLayout, shard_id: int
) -> tuple[list[int], list[int]]:
    addrs, sizes, _ = layout.prepare_values([(0, layout.block_size)], [0], [shard_id])
    return addrs[0], sizes[0]


def _read_segments(addrs: list[int], sizes: list[int]) -> bytes:
    return b"".join(
        ctypes.string_at(addr, size) for addr, size in zip(addrs, sizes, strict=True)
    )


def test_tp_shared_layout_loads_partial_prefix_from_physical_block():
    metadata = KeyMetadata("test-model", 0, 0, 0, 0)
    layout = LBHNCStoreLayout(
        metadata,
        16,
        16,
        local_tp_size=2,
        store_tp_size=4,
        tp_rank=0,
        layer_specs=_attention_specs(1, 16, 4),
    )
    tensor = torch.empty((2, 4, 16, 4), dtype=torch.float16)
    layout.register_kv_caches([tensor], 2)
    shard_ids = layout.local_shard_ids

    addrs, sizes, block_ids = layout.prepare_values(
        [(0, 8)] * len(shard_ids), [1], shard_ids
    )

    assert len(addrs) == len(sizes) == len(shard_ids)
    assert block_ids == [1] * len(shard_ids)


def _assert_state_store_round_trip(layout_factory):
    block_hash = BlockHash(b"h")
    for producer_tp, consumer_tp in ((4, 2), (2, 4)):
        stored: dict[int, bytes] = {}
        keys: dict[int, str] = {}
        schemas: set[str] = set()
        for tp_rank in range(producer_tp):
            layout, cache = layout_factory(local_tp_size=producer_tp, tp_rank=tp_rank)
            schemas.add(
                MambaStoreLayout.schema_fingerprint(layout.layer_specs, producer_tp, 4)
            )
            cache.zero_()
            for shard_id in layout.local_shard_ids:
                addrs, sizes = _descriptors_for_block(layout, shard_id)
                for segment_index, (addr, size) in enumerate(
                    zip(addrs, sizes, strict=True)
                ):
                    value = 32 + segment_index * 16
                    if segment_index in (0, 3):
                        value += shard_id
                    ctypes.memset(addr, value, size)
                stored[shard_id] = _read_segments(addrs, sizes)
                keys[shard_id] = layout.key_for(shard_id, block_hash)

        for tp_rank in range(consumer_tp):
            layout, cache = layout_factory(local_tp_size=consumer_tp, tp_rank=tp_rank)
            schemas.add(
                MambaStoreLayout.schema_fingerprint(layout.layer_specs, consumer_tp, 4)
            )
            cache.zero_()
            for shard_id in layout.local_shard_ids:
                assert layout.key_for(shard_id, block_hash) == keys[shard_id]
                addrs, sizes = _descriptors_for_block(layout, shard_id)
                offset = 0
                for addr, size in zip(addrs, sizes, strict=True):
                    ctypes.memmove(addr, stored[shard_id][offset : offset + size], size)
                    offset += size
                assert _read_segments(addrs, sizes) == stored[shard_id]
        assert set(stored) == set(range(4))
        assert len(schemas) == 1


@pytest.mark.parametrize("conv_layout", ["DS", "SD"])
def test_gdn_store_shards_round_trip_in_both_tp_directions(monkeypatch, conv_layout):
    monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", conv_layout)
    _assert_state_store_round_trip(_make_gdn_store_layout)


@pytest.mark.parametrize("conv_layout", ["DS", "SD"])
def test_mamba2_replicated_groups_round_trip_in_both_tp_directions(
    monkeypatch, conv_layout
):
    monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", conv_layout)
    _assert_state_store_round_trip(_make_mamba2_store_layout)


@pytest.mark.parametrize(
    ("conv_layout", "sizes", "shard0_offsets", "shard1_offsets"),
    [
        ("DS", [6, 6, 6, 4], [0, 12, 24, 36], [6, 18, 30, 40]),
        (
            "SD",
            [2] * 9 + [4],
            [0, 4, 8, 12, 16, 20, 24, 28, 32, 36],
            [2, 6, 10, 14, 18, 22, 26, 30, 34, 40],
        ),
    ],
)
def test_gdn_store_shard_segments_preserve_projection_boundaries(
    monkeypatch, conv_layout, sizes, shard0_offsets, shard1_offsets
):
    monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", conv_layout)
    layout, cache = _make_gdn_store_layout(local_tp_size=2, tp_rank=0)

    shard0_addrs, shard0_sizes = _descriptors_for_block(layout, 0)
    shard1_addrs, shard1_sizes = _descriptors_for_block(layout, 1)

    assert shard0_sizes == shard1_sizes == sizes
    assert [addr - cache.data_ptr() for addr in shard0_addrs] == shard0_offsets
    assert [addr - cache.data_ptr() for addr in shard1_addrs] == shard1_offsets


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


def test_rank_local_database_api_rejects_tp_shared_layout():
    metadata = KeyMetadata("test-model", 0, 0, 0, 0)
    layout = LBHNCStoreLayout(
        metadata,
        block_size=16,
        hash_block_size=16,
        local_tp_size=2,
        store_tp_size=4,
        tp_rank=0,
        layer_specs=_attention_specs(1, 16, 4),
    )
    database = ChunkedTokenDatabase(metadata, 16, store_layout=layout)

    with pytest.raises(RuntimeError, match="rank-local"):
        database.prepare_value_for_block(0)


_SHARED_LAYOUTS = [
    (KVCacheLayout.LBHNC, LBHNCStoreLayout),
    (KVCacheLayout.LBNHC, LBNHCStoreLayout),
    (KVCacheLayout.BLHNC, BLHNCStoreLayout),
    (KVCacheLayout.BLNHC, BLNHCStoreLayout),
    (KVCacheLayout.LHBNC, LHBNCStoreLayout),
    (KVCacheLayout.BHLNC, BHLNCStoreLayout),
]


def _physical_layer_views(
    layout: KVCacheLayout,
    num_layers: int,
    num_blocks: int,
    num_heads: int,
    block_size: int,
    content_size: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    logical_shape = (num_layers, num_blocks, num_heads, block_size, content_size)
    axis_order = layout.value
    physical = torch.empty(
        tuple(logical_shape[axis] for axis in axis_order), dtype=torch.float16
    )
    logical = physical.permute(tuple(axis_order.index(axis) for axis in range(5)))
    return physical, list(logical.unbind(0))


@pytest.mark.parametrize(
    ("producer_layout", "producer_cls", "consumer_layout", "consumer_cls"),
    [(layout, layout_cls, layout, layout_cls) for layout, layout_cls in _SHARED_LAYOUTS]
    + [
        (
            KVCacheLayout.LBHNC,
            LBHNCStoreLayout,
            KVCacheLayout.BLHNC,
            BLHNCStoreLayout,
        ),
        (
            KVCacheLayout.LHBNC,
            LHBNCStoreLayout,
            KVCacheLayout.BHLNC,
            BHLNCStoreLayout,
        ),
        (
            KVCacheLayout.LBNHC,
            LBNHCStoreLayout,
            KVCacheLayout.BLNHC,
            BLNHCStoreLayout,
        ),
    ],
)
def test_tp_shared_layout_round_trip_across_tp_sizes(
    producer_layout: KVCacheLayout,
    producer_cls,
    consumer_layout: KVCacheLayout,
    consumer_cls,
):
    block_size = 16
    num_layers = 2
    stored: dict[int, bytes] = {}
    producer_layers = []

    for tp_rank in range(4):
        physical, layers = _physical_layer_views(
            producer_layout, num_layers, 1, 2, block_size, 4
        )
        for layer_index, layer in enumerate(layers):
            layer.copy_(
                torch.arange(layer.numel(), dtype=torch.float16).view_as(layer)
                + tp_rank * 1000
                + layer_index * 100
            )
        producer_layers.append(layers)
        metadata = KeyMetadata("test-model", tp_rank, 0, 0, 0)
        layout = producer_cls(
            metadata,
            block_size,
            block_size,
            local_tp_size=4,
            store_tp_size=4,
            tp_rank=tp_rank,
            layer_specs=_attention_specs(num_layers, block_size, 2),
        )
        layout.register_kv_caches(layers, 1)
        addrs, sizes, _ = layout.prepare_values([(0, block_size)], [0], [tp_rank])
        stored[tp_rank] = b"".join(
            ctypes.string_at(addr, size)
            for addr, size in zip(addrs[0], sizes[0], strict=True)
        )

    for tp_rank in range(2):
        physical, layers = _physical_layer_views(
            consumer_layout, num_layers, 1, 4, block_size, 4
        )
        physical.zero_()
        metadata = KeyMetadata("test-model", tp_rank, 0, 0, 0)
        layout = consumer_cls(
            metadata,
            block_size,
            block_size,
            local_tp_size=2,
            store_tp_size=4,
            tp_rank=tp_rank,
            layer_specs=_attention_specs(num_layers, block_size, 4),
        )
        layout.register_kv_caches(layers, 1)
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

        for layer_index, layer in enumerate(layers):
            expected = torch.cat(
                [
                    rank_layers[layer_index]
                    for rank_layers in producer_layers[tp_rank * 2 : tp_rank * 2 + 2]
                ],
                dim=1,
            )
            torch.testing.assert_close(layer, expected)


@pytest.mark.parametrize(
    ("cache_layout", "layout_cls"),
    [
        (KVCacheLayout.LBHNC, LBHNCStoreLayout),
        (KVCacheLayout.LBNHC, LBNHCStoreLayout),
    ],
)
def test_attention_store_layout_round_trip_across_local_block_sizes(
    cache_layout, layout_cls
):
    metadata = KeyMetadata("test-model", 0, 0, 0, 0)
    chunks = [(0, 400), (400, 800)]

    _, (small_cache,) = _physical_layer_views(cache_layout, 1, 2, 2, 400, 4)
    small_cache.copy_(
        torch.arange(small_cache.numel(), dtype=torch.float16).view_as(small_cache)
    )
    small_layout = layout_cls(
        metadata,
        block_size=400,
        hash_block_size=16,
        local_tp_size=1,
        store_tp_size=1,
        tp_rank=0,
        layer_specs=_attention_specs(1, 400, 2),
        store_chunk_size=400,
    )
    small_layout.register_kv_caches([small_cache], num_blocks=2)
    addrs, sizes, _ = small_layout.prepare_values(chunks, [0, 1], [0, 0])
    stored = [_read_segments(a, s) for a, s in zip(addrs, sizes, strict=True)]

    _, (large_cache,) = _physical_layer_views(cache_layout, 1, 1, 2, 800, 4)
    large_cache.zero_()
    large_layout = layout_cls(
        metadata,
        block_size=800,
        hash_block_size=16,
        local_tp_size=1,
        store_tp_size=1,
        tp_rank=0,
        layer_specs=_attention_specs(1, 800, 2),
        store_chunk_size=400,
    )
    large_layout.register_kv_caches([large_cache], num_blocks=1)
    addrs, sizes, _ = large_layout.prepare_values(chunks, [0], [0, 0])
    for value, value_addrs, value_sizes in zip(stored, addrs, sizes, strict=True):
        offset = 0
        for addr, size in zip(value_addrs, value_sizes, strict=True):
            ctypes.memmove(addr, value[offset : offset + size], size)
            offset += size
    expected = torch.cat((small_cache[0], small_cache[1]), dim=1)
    torch.testing.assert_close(large_cache[0], expected)

    large_values = [_read_segments(a, s) for a, s in zip(addrs, sizes, strict=True)]
    _, (restored_small,) = _physical_layer_views(cache_layout, 1, 2, 2, 400, 4)
    restored_small.zero_()
    restored_layout = layout_cls(
        metadata,
        block_size=400,
        hash_block_size=16,
        local_tp_size=1,
        store_tp_size=1,
        tp_rank=0,
        layer_specs=_attention_specs(1, 400, 2),
        store_chunk_size=400,
    )
    restored_layout.register_kv_caches([restored_small], num_blocks=2)
    addrs, sizes, _ = restored_layout.prepare_values(chunks, [0, 1], [0, 0])
    for value, value_addrs, value_sizes in zip(large_values, addrs, sizes, strict=True):
        offset = 0
        for addr, size in zip(value_addrs, value_sizes, strict=True):
            ctypes.memmove(addr, value[offset : offset + size], size)
            offset += size
    torch.testing.assert_close(restored_small, small_cache)


def test_attention_store_chunk_is_independent_of_k3_local_pages():
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    fingerprints = set()
    for block_size in (1536, 11776):
        spec = MLAAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
        )
        chunk_size = AttentionStoreLayout.resolve_store_chunk_size(
            (spec,),
            requested_store_chunk_size=512,
        )
        assert chunk_size == 512
        fingerprints.add(
            AttentionStoreLayout.schema_fingerprint((spec,), 1, 1, chunk_size)
        )

    assert len(fingerprints) == 1


def test_tp_shared_layout_handles_kernel_blocked_compressed_states():
    from vllm.v1.kv_cache_interface import MLAAttentionSpec, group_kernel_blocks

    spec = MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.uint8,
        tokens_per_state=4,
        model_version="deepseek_v4",
    )
    cache = torch.arange(2 * 4 * 8, dtype=torch.uint8).view(8, 1, 1, 8)
    cache = group_kernel_blocks(cache, num_blocks=2)
    metadata = KeyMetadata("test-model", 0, 0, 0, 0)
    layout = LBHNCStoreLayout(
        metadata,
        block_size=16,
        hash_block_size=4,
        local_tp_size=1,
        store_tp_size=1,
        tp_rank=0,
        layer_specs=(spec,),
    )

    layout.register_kv_caches([cache], num_blocks=2)
    addrs, sizes, _ = layout.prepare_values([(0, 16)], [0], [0])

    assert sizes == [[8, 8, 8, 8]]
    assert _read_segments(addrs[0], sizes[0]) == bytes(range(32))
