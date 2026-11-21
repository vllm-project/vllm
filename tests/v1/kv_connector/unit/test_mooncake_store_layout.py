# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Mooncake Store payload layouts."""

import ctypes
import os
import random

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake_store.layout import (  # noqa: E501
    AttentionStoreLayout,
    HeadMajorStoreLayout,
    KeyMetadata,
    MambaStoreLayout,
    RankLocalStoreLayout,
    TokenMajorStoreLayout,
)
from vllm.model_executor.layers.mamba.mamba_utils import get_conv_state_layout
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import group_kernel_blocks
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
    from vllm.v1.kv_cache_interface import AttentionSpec

    return tuple(
        AttentionSpec(
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
    layout = HeadMajorStoreLayout(
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


@pytest.mark.parametrize("num_regions", [1, 3])
def test_rank_local_descriptors_match_reference(num_regions: int):
    num_blocks = 8
    views = [
        torch.zeros((num_blocks, 16 + 8 * i), dtype=torch.uint8)[:, 4:-4]
        for i in range(num_regions)
    ]
    layout = RankLocalStoreLayout(
        KeyMetadata("test-model", 1, 0, 0, 0), BLOCK_SIZE, BLOCK_SIZE // 2
    )
    layout.register_kv_caches(views, num_blocks)
    rng = random.Random(0)
    block_ids = [rng.randrange(num_blocks) for _ in range(300)]
    chunks = []
    block = 0
    while block < len(block_ids):
        span = rng.choice([BLOCK_SIZE // 2, BLOCK_SIZE])
        chunks.append((block * BLOCK_SIZE, block * BLOCK_SIZE + span))
        block += rng.choice([1, 1, 1, 2, 4])

    addrs, sizes, selected_blocks = layout.prepare_values(
        chunks, block_ids, [0] * len(chunks)
    )

    for chunk, chunk_addrs, chunk_sizes, block_id in zip(
        chunks, addrs, sizes, selected_blocks, strict=True
    ):
        expected_block = block_ids[chunk[0] // BLOCK_SIZE]
        assert block_id == expected_block
        assert chunk_addrs == [view[expected_block].data_ptr() for view in views]
        assert chunk_sizes == [view[expected_block].nbytes for view in views]
        assert all(type(addr) is int for addr in chunk_addrs)
        assert type(block_id) is int


def test_rank_local_descriptors_handle_empty_and_invalid_chunks():
    cache = torch.zeros((4, 16), dtype=torch.uint8)
    layout = RankLocalStoreLayout(
        KeyMetadata("test-model", 0, 0, 0, 0), BLOCK_SIZE, BLOCK_SIZE // 2
    )
    layout.register_kv_caches([cache], 4)
    assert layout.prepare_values([], [1, 2, 3], []) == ([], [], [])
    for chunk in (
        (0, BLOCK_SIZE // 2 + 1),
        (0, 2 * BLOCK_SIZE),
        (BLOCK_SIZE // 2, 3 * BLOCK_SIZE // 2),
        (0, 0),
    ):
        with pytest.raises(AssertionError):
            layout.prepare_values([chunk], [0, 1], [0])


@pytest.mark.parametrize(
    "component_slice,kernel_blocks_per_manager,pre_grouped",
    [
        (slice(16, 24), 1, False),
        (slice(None, None, 2), 1, False),
        (slice(0, 16), 2, False),
        (slice(0, 16), 2, True),
    ],
    ids=["offset", "strided", "kernel-blocks", "manager-blocks"],
)
def test_component_layout_round_trip_preserves_neighboring_blocks(
    component_slice, kernel_blocks_per_manager, pre_grouped
):
    """Copy a component without changing padding or neighboring blocks."""
    count = kernel_blocks_per_manager
    source = torch.arange(5 * count * 32).to(torch.uint8).view(5 * count, 32)
    destination = torch.full_like(source, 255)
    expected = destination.clone()
    expected[count : 2 * count, component_slice] = source[
        3 * count : 4 * count, component_slice
    ]
    layouts = [
        RankLocalStoreLayout(KeyMetadata("test-model", 0, 0, 0, 0), 16, 16)
        for _ in range(2)
    ]
    for layout, cache in zip(layouts, (source, destination), strict=True):
        view = cache[: 4 * count, component_slice]
        if pre_grouped:
            view = group_kernel_blocks(view, 4)
        layout.register_kv_caches([view], 4)
    source_addrs, sizes, _ = layouts[0].prepare_values([(0, 16)], [3], [0])
    destination_addrs, destination_sizes, _ = layouts[1].prepare_values(
        [(0, 16)], [1], [0]
    )
    assert sizes == destination_sizes
    for src, dst, size in zip(
        source_addrs[0], destination_addrs[0], sizes[0], strict=True
    ):
        ctypes.memmove(dst, ctypes.string_at(src, size), size)
    assert torch.equal(destination, expected)


def test_component_layout_deduplicates_shared_views_in_input_order():
    raw = torch.zeros(4, 32, dtype=torch.uint8)
    first, second = raw[:, 16:24], raw[:, :8]
    layout = RankLocalStoreLayout(KeyMetadata("test-model", 0, 0, 0, 0), 16, 16)
    layout.register_kv_caches([first, second, first, raw[:, :0], second], 4)
    assert layout.prepare_values([(0, 16)], [3], [0]) == (
        [[first[3].data_ptr(), second[3].data_ptr()]],
        [[8, 8]],
        [3],
    )


@pytest.mark.parametrize(
    "regions,expected_sizes",
    [
        ([(0, 32), (8, 32), (16, 32)], [24]),
        ([(0, 32), (16, 32)], [8, 8]),
        ([(8, 32), (0, 32)], [8, 8]),
        ([(0, 32), (8, 16)], [8, 8]),
        ([(0, 8), (8, 8)], [8, 8]),
    ],
    ids=["adjacent", "gap", "reversed", "different-strides", "block-boundary"],
)
def test_component_layout_coalesces_regions_without_changing_payload(
    regions, expected_sizes
):
    raw = torch.arange(128, dtype=torch.uint8)
    views = [raw.as_strided((4, 8), (stride, 1), offset) for offset, stride in regions]
    layout = RankLocalStoreLayout(KeyMetadata("test-model", 0, 0, 0, 0), 16, 16)
    layout.register_kv_caches(views, 4)
    addrs, sizes, blocks = layout.prepare_values([(0, 16), (16, 32)], [3, 1], [0, 0])
    assert sizes == [expected_sizes, expected_sizes]
    for addresses, lengths, block in zip(addrs, sizes, blocks, strict=True):
        assert _read_segments(addresses, lengths) == b"".join(
            view[block].numpy().tobytes() for view in views
        )


def test_component_layout_does_not_coalesce_across_registered_storage_bounds():
    raw = bytearray(range(16))
    views = [
        torch.frombuffer(raw, dtype=torch.uint8, count=8, offset=offset).as_strided(
            (1, 8), (16, 1)
        )
        for offset in (0, 8)
    ]
    assert views[0].data_ptr() + 8 == views[1].data_ptr()
    layout = RankLocalStoreLayout(KeyMetadata("test-model", 0, 0, 0, 0), 16, 16)
    layout.register_kv_caches(views, 1)
    assert layout.prepare_values([(0, 16)], [0], [0]) == (
        [[view.data_ptr() for view in views]],
        [[8, 8]],
        [0],
    )


def test_rank_local_padded_kernel_blocks_round_trip_preserves_padding():
    # Merged DSA cache (Kimi K3.1): 64-token kernel blocks of 64 x 788 B, each
    # padded to 50512 B, 17 per manager block.
    num_blocks, ratio, unit, padded = 4, 17, 64 * 788, 50_512
    source = torch.arange(num_blocks * ratio * padded).to(torch.uint8)
    destination = torch.full_like(source, 255)
    expected = destination.clone().view(num_blocks * ratio, padded)
    expected[ratio : 2 * ratio, :unit] = source.view(-1, padded)[-ratio:, :unit]
    layouts = [
        RankLocalStoreLayout(KeyMetadata("test-model", 0, 0, 0, 0), 1088, 1088)
        for _ in range(2)
    ]
    for layout, raw in zip(layouts, (source, destination), strict=True):
        cache = raw.as_strided((num_blocks * ratio, 1, 64, 788), (padded, unit, 788, 1))
        layout.register_kv_caches([cache], num_blocks)
        # Gaps may belong to another component: only transfer live bytes.
        assert layout.kv_caches_base_addr == [
            raw.data_ptr() + i * padded for i in range(ratio)
        ]
        assert layout.block_stride == [ratio * padded] * ratio
        assert layout.block_len == [unit] * ratio
    addrs, sizes, _ = layouts[0].prepare_values([(0, 1088)], [num_blocks - 1], [0])
    assert addrs[0][-1] + sizes[0][-1] == (
        source.data_ptr() + source.numel() - (padded - unit)
    )
    dst_addrs, dst_sizes, _ = layouts[1].prepare_values([(0, 1088)], [1], [0])
    assert sizes == dst_sizes
    for src, dst, size in zip(addrs[0], dst_addrs[0], sizes[0], strict=True):
        ctypes.memmove(dst, src, size)
    assert torch.equal(destination.view_as(expected), expected)


def test_rank_local_splits_interleaved_blocks_per_head():
    # K and V of all blocks in separate planes: a block is not stride-compact.
    num_blocks = 4
    kv = torch.zeros(2, num_blocks, BLOCK_SIZE, 64, dtype=torch.uint8)
    layout = RankLocalStoreLayout(
        KeyMetadata("test-model", 0, 0, 0, 0), BLOCK_SIZE, BLOCK_SIZE
    )

    layout.register_kv_caches([kv.permute(1, 0, 2, 3)], num_blocks)

    assert layout.kv_caches_base_addr == [kv[0].data_ptr(), kv[1].data_ptr()]
    assert layout.block_len == [BLOCK_SIZE * 64] * 2


def test_rank_local_layout_skips_block_padding():
    """A state padded up to a larger page, like a Mamba state padded to the
    attention page size, is stored without the padding."""
    num_layers, num_blocks, page, state = 2, 3, 64, 40
    # Layer-outermost buffer; each layer views its state at the start of
    # every page.
    raw = torch.zeros(num_layers * num_blocks * page, dtype=torch.int8)
    views = [
        torch.as_strided(
            raw,
            (num_blocks, 1, 1, state),
            (page, state, state, 1),
            layer * num_blocks * page,
        )
        for layer in range(num_layers)
    ]
    layout = RankLocalStoreLayout(
        KeyMetadata("test-model", 0, 0, 0, 0), BLOCK_SIZE, BLOCK_SIZE
    )
    layout.register_kv_caches(views, num_blocks)

    assert layout.kv_caches_base_addr == [view.data_ptr() for view in views]
    assert layout.block_stride == [page] * num_layers
    assert layout.block_len == [state] * num_layers
    addrs, sizes, _ = layout.prepare_values(
        [(0, BLOCK_SIZE), (BLOCK_SIZE, 2 * BLOCK_SIZE)], [2, 0], [0, 0]
    )
    assert addrs == [
        [view.data_ptr() + 2 * page for view in views],
        [view.data_ptr() for view in views],
    ]
    # Separate chunks leave every block's padding untouched.
    assert sizes == [[state] * num_layers, [state] * num_layers]
    addrs, sizes, _ = layout.prepare_values([(0, BLOCK_SIZE)], [1], [0])
    assert addrs == [[view.data_ptr() + page for view in views]]
    assert sizes == [[state] * num_layers]


_SHARED_LAYOUTS = [
    (KVCacheLayout.LBHNC, HeadMajorStoreLayout),
    (KVCacheLayout.LBNHC, TokenMajorStoreLayout),
    (KVCacheLayout.BLHNC, HeadMajorStoreLayout),
    (KVCacheLayout.BLNHC, TokenMajorStoreLayout),
    (KVCacheLayout.LHBNC, HeadMajorStoreLayout),
    (KVCacheLayout.BHLNC, HeadMajorStoreLayout),
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
            HeadMajorStoreLayout,
            KVCacheLayout.BLHNC,
            HeadMajorStoreLayout,
        ),
        (
            KVCacheLayout.LHBNC,
            HeadMajorStoreLayout,
            KVCacheLayout.BHLNC,
            HeadMajorStoreLayout,
        ),
        (
            KVCacheLayout.LBNHC,
            TokenMajorStoreLayout,
            KVCacheLayout.BLNHC,
            TokenMajorStoreLayout,
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
        (KVCacheLayout.LBHNC, HeadMajorStoreLayout),
        (KVCacheLayout.LBNHC, TokenMajorStoreLayout),
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


@pytest.mark.parametrize("layout_cls", [HeadMajorStoreLayout, TokenMajorStoreLayout])
@pytest.mark.parametrize("chunk_size", [3, 4])
def test_chunk_descriptors_preserve_order_across_kernel_blocks(layout_cls, chunk_size):
    cache = torch.empty((3, 3, 4, 4, 4), dtype=torch.float16)
    if layout_cls is TokenMajorStoreLayout:
        cache = cache.transpose(2, 3)
    cache.copy_(torch.arange(cache.numel(), dtype=cache.dtype).reshape(cache.shape))
    layout = layout_cls(
        KeyMetadata("test-model", 1, 0, 0, 0),
        block_size=12,
        hash_block_size=1,
        local_tp_size=2,
        store_tp_size=4,
        tp_rank=1,
        layer_specs=_attention_specs(1, 12, 4),
        store_chunk_size=chunk_size,
    )
    layout.register_kv_caches([cache], 3)
    starts = [12 + chunk_size, 0, 24 + 2 * chunk_size, 12 + chunk_size] * 8
    shards = [3, 2, 3, 3] * 8
    chunks = [(start, start + chunk_size) for start in starts]
    addrs, sizes, blocks = layout.prepare_values(chunks, [2, 0, 1], shards)

    assert blocks == [0, 2, 1, 0] * 8
    assert layout.prepare_values(chunks[:4], [2, 0, 1], shards[:4]) == (
        addrs[:4],
        sizes[:4],
        blocks[:4],
    )
    for start, shard, block, addresses, lengths in zip(
        starts, shards, blocks, addrs, sizes, strict=True
    ):
        logical = cache[block].permute(1, 0, 2, 3).reshape(4, 12, 4)
        head_start = (shard - 2) * 2
        offset = start % 12
        expected = logical[head_start : head_start + 2, offset : offset + chunk_size]
        if layout_cls is TokenMajorStoreLayout:
            expected = expected.transpose(0, 1)
        expected_bytes = expected.contiguous().numpy().tobytes()
        assert _read_segments(addresses, lengths) == expected_bytes


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
    layout = HeadMajorStoreLayout(
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
