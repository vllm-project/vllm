# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from types import SimpleNamespace

import pytest
import torch

from tests.v1.attention.utils import dense_kv_cache_views
from vllm.v1.core.kv_cache_utils import KVBlockTail
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheLayout,
    MLAAttentionSpec,
    SlidingWindowSpec,
    SparseCacheRole,
)
from vllm.v1.worker import utils as worker_utils
from vllm.v1.worker.utils import (
    AttentionGroup,
    KVBlockZeroer,
    _zero_kv_blocks_kernel,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "spec",
    [
        SlidingWindowSpec(
            block_size=2,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.uint8,
            sliding_window=4,
        ),
        ChunkedLocalAttentionSpec(
            block_size=2,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.uint8,
            attention_chunk_size=4,
        ),
    ],
    ids=["sliding-window", "chunked-local"],
)
def test_attention_blocks_are_zeroed(spec):
    device = torch.device("cuda")
    storage = torch.ones((4, 1, 2, 2), dtype=torch.uint8, device=device)
    layer_name = "draft.self_attn"
    zeroer = KVBlockZeroer(
        device,
        attn_groups_iter=[AttentionGroup(None, [layer_name], spec, 0)],
        kernel_block_sizes=[2],
        static_forward_context={
            layer_name: SimpleNamespace(kv_cache=storage),
        },
        num_blocks=4,
    )

    zeroer.zero_block_ids([1])
    torch.accelerator.synchronize()

    expected = torch.ones_like(storage)
    expected[1] = 0
    assert torch.equal(storage, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_layers_in_one_group_may_use_different_kernel_pages_per_block():
    """Derive kernel pages per block from each layer's allocation."""
    device = torch.device("cuda")
    num_blocks = 4
    spec = SlidingWindowSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.int32,
        sliding_window=2,
    )
    # Two kernel pages per logical block.
    wide = torch.ones((num_blocks * 2, 3), dtype=torch.int32, device=device)
    # One kernel page per logical block, carved out of a larger allocation so
    # an over-strided write lands in the guard region instead of faulting or
    # silently hitting another tensor.
    narrow_backing = torch.ones((num_blocks * 3, 5), dtype=torch.int32, device=device)
    narrow = narrow_backing[:num_blocks]

    zeroer = KVBlockZeroer(
        device,
        attn_groups_iter=[
            AttentionGroup(
                None,
                ["wide", "narrow"],
                spec,
                0,
            )
        ],
        kernel_block_sizes=[1],
        static_forward_context={
            "wide": SimpleNamespace(kv_cache=wide),
            "narrow": SimpleNamespace(kv_cache=narrow),
        },
        num_blocks=num_blocks,
    )

    zeroer.zero_block_ids([num_blocks - 1])
    torch.accelerator.synchronize()

    expected_wide = torch.ones_like(wide)
    expected_wide[2 * (num_blocks - 1) :] = 0
    assert torch.equal(wide, expected_wide)

    expected_narrow = torch.ones_like(narrow_backing)
    expected_narrow[num_blocks - 1] = 0
    assert torch.equal(narrow_backing, expected_narrow)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_block_ids_are_not_overwritten_while_copy_is_in_flight():
    device = torch.device("cuda")
    num_blocks = 4
    page_size_el = 4
    storage = torch.ones((num_blocks, page_size_el), dtype=torch.int32, device=device)

    # Build the minimal zeroer state directly so the test can focus on the
    # in-flight copy behavior without constructing model attention groups.
    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = device
    zeroer._meta = (
        torch.tensor([storage.data_ptr()], dtype=torch.uint64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        page_size_el // page_size_el,  # max_chunks = 1
        page_size_el,  # blk_size
        1,  # n_segs
    )

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # Keep the first nonblocking H2D copy pending while the host submits the
        # second call. Each call must stage from its own pinned source so the
        # first copy is not corrupted before it runs.
        torch.cuda._sleep(10_000_000)
        zeroer.zero_block_ids([1])
        zeroer.zero_block_ids([2])
    stream.synchronize()

    assert torch.all(storage[0] == 1)
    assert torch.all(storage[1] == 0)
    assert torch.all(storage[2] == 0)
    assert torch.all(storage[3] == 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_non_uniform_page_sizes():
    """Two segments with different page sizes (e.g. MLA + DSA indexer)."""
    device = torch.device("cuda")
    num_blocks = 4
    page_size_a = 10496  # int32 elements
    page_size_b = 2112

    storage_a = torch.ones((num_blocks, page_size_a), dtype=torch.int32, device=device)
    storage_b = torch.ones((num_blocks, page_size_b), dtype=torch.int32, device=device)

    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = device

    seg_page_sizes = [page_size_a, page_size_b]
    max_ps = max(seg_page_sizes)

    blk_size = min(1 << (max_ps - 1).bit_length(), 1024)

    zeroer._meta = (
        torch.tensor(
            [storage_a.data_ptr(), storage_b.data_ptr()],
            dtype=torch.uint64,
            device=device,
        ),
        torch.tensor(seg_page_sizes, dtype=torch.int64, device=device),
        torch.tensor(seg_page_sizes, dtype=torch.int64, device=device),
        (max_ps + blk_size - 1) // blk_size,
        blk_size,
        2,
    )

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        zeroer.zero_block_ids([1, 2])
    stream.synchronize()

    for storage in (storage_a, storage_b):
        assert torch.all(storage[0] == 1)
        assert torch.all(storage[1] == 0)
        assert torch.all(storage[2] == 0)
        assert torch.all(storage[3] == 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_packed_segment_zeros_only_its_last_block_page():
    """A packed KV segment steps by block stride but clears only its page."""
    device = torch.device("cuda")
    num_blocks = 4
    block_stride_el = 12
    page_size_el = 4
    page_offset_el = 3
    backing = torch.ones(
        (num_blocks, block_stride_el), dtype=torch.int32, device=device
    )

    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = device
    zeroer._meta = (
        torch.tensor(
            [backing.data_ptr() + page_offset_el * backing.element_size()],
            dtype=torch.uint64,
            device=device,
        ),
        torch.tensor([block_stride_el], dtype=torch.int64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        1,
        page_size_el,
        1,
    )

    zeroer.zero_block_ids([num_blocks - 1])
    torch.accelerator.synchronize()

    expected = torch.ones_like(backing)
    expected[-1, page_offset_el : page_offset_el + page_size_el] = 0
    assert torch.equal(backing, expected)


def test_large_dsv4_launch_geometry(monkeypatch):
    """Keep the failing DSV4 shape efficient and within launch limits."""
    device = torch.device("cpu")
    n_blocks, n_segs = 6870, 181
    layer_names = [f"layer.{i}" for i in range(n_segs)]
    page_sizes = [9344 if i % 2 == 0 else 292 for i in range(n_segs)]
    spec = SlidingWindowSpec(
        block_size=1,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.int32,
        sliding_window=1,
    )
    storages = {
        name: torch.ones((1, page_size), dtype=torch.int32)
        for name, page_size in zip(layer_names, page_sizes)
    }
    zeroer = KVBlockZeroer(
        device,
        attn_groups_iter=[
            AttentionGroup(None, [name], spec, group_id)
            for group_id, name in enumerate(layer_names)
        ],
        kernel_block_sizes=[1] * n_segs,
        static_forward_context={
            name: SimpleNamespace(kv_cache=storage)
            for name, storage in storages.items()
        },
        num_blocks=1,
    )

    assert zeroer._meta is not None
    _, _, seg_page_sizes, max_chunks, blk_size, n_segs = zeroer._meta
    assert seg_page_sizes.tolist() == page_sizes
    assert (max_chunks, blk_size, n_segs) == (10, 1024, 181)

    captured_grids = []

    class FakeKernel:
        def __getitem__(self, grid):
            captured_grids.append(grid)
            return lambda *args, **kwargs: None

    monkeypatch.setattr(worker_utils, "_zero_kv_blocks_kernel", FakeKernel())
    monkeypatch.setattr(
        worker_utils,
        "async_tensor_h2d",
        lambda values, **kwargs: torch.tensor(values, dtype=torch.int64),
    )

    zeroer.zero_block_ids(list(range(n_blocks)))

    old_max_chunks = max(page_sizes) // 4
    assert math.prod((n_blocks, n_segs, old_max_chunks)) > 2**31 - 1
    assert captured_grids == [(n_blocks, n_segs, max_chunks)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_warmup_compiles_for_all_block_counts():
    """After warmup, no launch should trigger a first-request JIT compile.

    The block count is carried by the launch grid, so changing it must reuse
    the warmup's compiled kernel.
    """
    device = torch.device("cuda")
    num_blocks = 64
    page_size_el = 4
    storage = torch.ones((num_blocks, page_size_el), dtype=torch.int32, device=device)

    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = device
    zeroer._meta = (
        torch.tensor([storage.data_ptr()], dtype=torch.uint64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        1,  # max_chunks
        page_size_el,  # blk_size
        1,  # n_segs
    )
    zeroer.seg_tail_layouts = [{}]

    def compiled_variants() -> set:
        return {
            key
            for caches in _zero_kv_blocks_kernel.device_caches.values()
            for key in caches[0]
        }

    zeroer.warmup(num_blocks)
    torch.accelerator.synchronize()
    warmed = compiled_variants()
    assert warmed

    for n_blocks in (1, 2, 3, 16, 32):
        zeroer.zero_block_ids(list(range(n_blocks)))
    torch.accelerator.synchronize()

    assert compiled_variants() == warmed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_warmup_respects_available_block_count():
    """An empty KV cache must not be warmed with out-of-range block IDs."""
    device = torch.device("cuda")
    page_size_el = 4
    storage = torch.ones((1, page_size_el), dtype=torch.int32, device=device)

    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.device = device
    zeroer._meta = (
        torch.tensor([storage.data_ptr()], dtype=torch.uint64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        torch.tensor([page_size_el], dtype=torch.int64, device=device),
        1,
        page_size_el,
        1,
    )

    zeroer.warmup(0)
    torch.accelerator.synchronize()

    assert torch.all(storage == 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("layout", list(KVCacheLayout))
def test_zeroes_exactly_one_block_per_layer(layout: KVCacheLayout):
    """The zeroer must zero every byte of the target block in every layer and nothing
    outside it — per head-group region under LHBNC, and never past the target block's
    tile under block-major layouts (no out-of-bounds writes, no clobbering)."""
    device = torch.device("cuda")
    num_blocks, num_layers = 4, 2
    spec = FullAttentionSpec(
        block_size=4, num_kv_heads=2, head_size=8, dtype=torch.float32
    )
    raw = torch.empty(
        num_blocks * num_layers * spec.page_size_bytes,
        dtype=torch.int8,
        device=device,
    ).fill_(1)
    views = dense_kv_cache_views(raw, spec, num_blocks, num_layers, layout)
    groups = [
        AttentionGroup(
            backend=None,
            layer_names=[f"layer.{i}" for i in range(num_layers)],
            kv_cache_spec=spec,
            kv_cache_group_id=0,
        )
    ]
    ctx = {f"layer.{i}": SimpleNamespace(kv_cache=views[i]) for i in range(num_layers)}
    zeroer = KVBlockZeroer(
        device,
        attn_groups_iter=iter(groups),
        kernel_block_sizes=[spec.block_size],
        static_forward_context=ctx,
        num_blocks=num_blocks,
    )
    zeroer.zero_block_ids([2])
    torch.accelerator.synchronize()

    for view in views:
        assert (view[2] == 0).all(), layout
        for b in (0, 1, 3):
            assert (view[b].view(torch.int8) == 1).all(), layout
    zero_bytes = int((raw == 0).sum().item())
    assert zero_bytes == num_layers * spec.page_size_bytes, layout


def _tail_zeroer(spec, caches, kernel_block_size, num_blocks, **dcp):
    return KVBlockZeroer(
        next(iter(caches.values())).device,
        attn_groups_iter=[AttentionGroup(None, list(caches), spec, 0)],
        kernel_block_sizes=[kernel_block_size],
        static_forward_context={
            name: SimpleNamespace(kv_cache=cache) for name, cache in caches.items()
        },
        num_blocks=num_blocks,
        **dcp,
    )


@pytest.mark.parametrize("dcp_size", [1, 2, 8])
@pytest.mark.parametrize("interleave", [1, 4, 16])
def test_tail_local_valid_slots_follow_token_ownership(dcp_size, interleave):
    """Every prefix of a block keeps exactly the slots of this rank's tokens,
    where token ``t`` of a block lives on rank ``t // interleave % dcp``."""
    block_size = 16
    zeroer = KVBlockZeroer.__new__(KVBlockZeroer)
    zeroer.group_dcp_sizes = {0: dcp_size, 1: 1}
    zeroer.cp_kv_cache_interleave_size = interleave
    for rank in range(dcp_size):
        zeroer.dcp_rank = rank
        for num_valid in range(block_size * dcp_size + 1):
            owned = sum(t // interleave % dcp_size == rank for t in range(num_valid))
            assert zeroer.local_valid_slots(KVBlockTail(0, 0, num_valid)) == owned
            # A group whose blocks are replicated across DCP ranks.
            assert zeroer.local_valid_slots(KVBlockTail(1, 0, num_valid)) == num_valid


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("layout", list(KVCacheLayout))
@pytest.mark.parametrize("kernel_block_size", [8, 4])
@pytest.mark.parametrize("num_valid", [3, 6])
def test_zero_tails_keep_exactly_the_valid_slots(layout, kernel_block_size, num_valid):
    """A tail zeroes the slots past ``num_valid`` of its block in every layer,
    in every layout and across kernel pages, and nothing else."""
    device = torch.device("cuda")
    num_blocks, num_layers, block_size = 4, 2, 8
    spec = FullAttentionSpec(
        block_size=block_size, num_kv_heads=2, head_size=8, dtype=torch.float32
    )
    raw = torch.ones(
        num_blocks * num_layers * spec.page_size_bytes, dtype=torch.int8, device=device
    )
    expected = raw.clone()
    try:
        views = dense_kv_cache_views(
            raw, spec, num_blocks, num_layers, layout, kernel_block_size
        )
    except ValueError:
        pytest.skip(f"{layout.name} cannot split blocks into kernel pages")
    ratio = block_size // kernel_block_size
    for view in dense_kv_cache_views(
        expected, spec, num_blocks, num_layers, layout, kernel_block_size
    ):
        pages = view.unflatten(0, (num_blocks, ratio))
        for token in range(num_valid, block_size):
            page, slot = divmod(token, kernel_block_size)
            pages[2, page, :, slot] = 0
    zeroer = _tail_zeroer(
        spec,
        {f"layer.{i}": view for i, view in enumerate(views)},
        kernel_block_size,
        num_blocks,
    )
    assert all(zeroer.seg_tail_layouts)

    zeroer.zero_block_ids([], [KVBlockTail(0, 2, num_valid)])
    torch.accelerator.synchronize()

    assert torch.equal(raw, expected), layout


def _mla_cache(num_blocks, pages_per_block, page, slot_bytes):
    return torch.full(
        (num_blocks * pages_per_block, 1, page, slot_bytes),
        7,
        dtype=torch.uint8,
        device="cuda",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dcp_rank", range(8))
def test_zero_tails_dcp8_mla(dcp_rank):
    """DCP8 with 1536-slot rank-local blocks of 24 64-slot FP8 MLA pages: a
    275,089-token load ends in block 22 with 4,753 tokens, i.e. 595 slots on
    rank 0 and 594 on the others. Exactly the rest of that block is zeroed."""
    block_size, page, slot_bytes, num_blocks = 1536, 64, 576, 3
    spec = MLAAttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=slot_bytes, dtype=torch.uint8
    )
    cache = _mla_cache(num_blocks, block_size // page, page, slot_bytes)
    zeroer = _tail_zeroer(
        spec, {"attn": cache}, page, num_blocks, dcp_world_size=8, dcp_rank=dcp_rank
    )

    zeroer.zero_block_ids([], [KVBlockTail(0, 1, 275_089 % (block_size * 8))])
    torch.accelerator.synchronize()

    valid = 595 if dcp_rank == 0 else 594
    slots = cache.unflatten(0, (num_blocks, block_size // page)).transpose(2, 3)
    slots = slots.flatten(1, 2)
    assert (slots[1, :valid] == 7).all()
    assert (slots[1, valid:] == 0).all()
    assert (slots[0] == 7).all() and (slots[2] == 7).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_zero_new_blocks_and_tails_in_one_launch():
    """New blocks are zeroed whole and tails past their valid slots in the same
    call; a tail of a group without segments touches nothing."""
    block_size, page, slot_bytes, num_blocks = 256, 64, 576, 4
    spec = MLAAttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=slot_bytes, dtype=torch.uint8
    )
    cache = _mla_cache(num_blocks, block_size // page, page, slot_bytes)
    zeroer = _tail_zeroer(spec, {"attn": cache}, page, num_blocks)

    zeroer.zero_block_ids([3], [KVBlockTail(0, 1, 100), KVBlockTail(1, 2, 5)])
    torch.accelerator.synchronize()

    slots = cache.unflatten(0, (num_blocks, block_size // page)).flatten(1, 3)
    assert (slots[1, :100] == 7).all() and (slots[1, 100:] == 0).all()
    assert (slots[3] == 0).all()
    assert (slots[0] == 7).all() and (slots[2] == 7).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_tails_skip_layouts_whose_slots_are_not_word_aligned():
    """bf16 with the token axis innermost puts two slots in one int32 word, so
    tails leave the cache alone while whole-block zeroing still works."""
    device = torch.device("cuda")
    num_blocks, block_size, head_size = 3, 8, 4
    spec = FullAttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=head_size, dtype=torch.bfloat16
    )
    cache = torch.ones(
        (num_blocks, 1, head_size, block_size), dtype=torch.bfloat16, device=device
    ).transpose(2, 3)
    zeroer = _tail_zeroer(spec, {"attn": cache}, block_size, num_blocks)
    assert zeroer.seg_tail_layouts == [{}]

    zeroer.zero_block_ids([], [KVBlockTail(0, 1, 3)])
    torch.accelerator.synchronize()
    assert (cache[1:] == 1).all()

    zeroer.zero_block_ids([1])
    torch.accelerator.synchronize()
    assert (cache[1] == 0).all()
    assert (cache[0] == 1).all() and (cache[2] == 1).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("group_id", [0, 1])
def test_zero_tails_of_groups_sharing_a_tensor(group_id):
    """Hybrid groups share KV cache tensors and a block belongs to one group at
    a time, so each group's tails must reach the shared segments."""
    block_size, page, slot_bytes, num_blocks = 128, 64, 576, 3
    spec = MLAAttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=slot_bytes, dtype=torch.uint8
    )
    cache = _mla_cache(num_blocks, block_size // page, page, slot_bytes)
    zeroer = KVBlockZeroer(
        cache.device,
        attn_groups_iter=[
            AttentionGroup(None, ["layer.0"], spec, 0),
            AttentionGroup(None, ["layer.1"], spec, 1),
        ],
        kernel_block_sizes=[page, page],
        static_forward_context={
            "layer.0": SimpleNamespace(kv_cache=cache),
            "layer.1": SimpleNamespace(kv_cache=cache),
        },
        num_blocks=num_blocks,
    )

    zeroer.zero_block_ids([], [KVBlockTail(group_id, 1, 70)])
    torch.accelerator.synchronize()

    slots = cache.unflatten(0, (num_blocks, block_size // page)).flatten(1, 3)
    assert (slots[1, :70] == 7).all() and (slots[1, 70:] == 0).all()
    assert (slots[0] == 7).all() and (slots[2] == 7).all()


def test_tails_skip_indexer_caches():
    """Indexer pages store all values, then all scales, so their token slots
    are not what the [B, N, C] view says."""
    block_size, head_size = 64, 132
    spec = MLAAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=head_size,
        dtype=torch.uint8,
        cache_role=SparseCacheRole.INDEXER,
    )
    cache = torch.zeros((2, block_size, head_size), dtype=torch.uint8)
    zeroer = _tail_zeroer(spec, {"indexer": cache}, block_size, 2)
    assert zeroer.seg_tail_layouts == [{}]
