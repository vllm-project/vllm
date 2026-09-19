# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy

import pytest
import torch

from vllm.config import ModelConfig, VllmConfig
from vllm.utils.extensible_tensor import (
    granule_aligned_blocks,
    granule_block_alignment,
)
from vllm.utils.vmm_driver import vmm_unavailable_reason
from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    KVCacheTensor,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.extensible_kv_cache import (
    ExtensibleKVCache,
    measure_kv_cache_bytes,
)
from vllm.v1.worker.utils import allocate_kv_cache

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


@pytest.fixture
def vmm():
    # Probed lazily so collection does not initialize CUDA in the pytest parent.
    if (reason := vmm_unavailable_reason()) is not None:
        pytest.skip(f"VMM unavailable: {reason}")


NUM_BLOCKS = 8
NUM_LAYERS = 3


def _make_config(layout: KVCacheLayout, num_blocks: int = NUM_BLOCKS):
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=64, dtype=torch.float16
    )
    vllm_config = VllmConfig(model_config=ModelConfig(max_model_len=16))
    vllm_config.cache_config.kv_cache_layout = layout.name
    layers = [f"layer.{i}" for i in range(NUM_LAYERS)]
    config = get_kv_cache_config_from_groups(
        vllm_config,
        [KVCacheGroupSpec(layers, spec)],
        available_memory=NUM_LAYERS * spec.page_size_bytes * num_blocks,
    )
    assert config.num_blocks == num_blocks
    return config, spec


@requires_cuda
@pytest.mark.parametrize(
    ("layout", "expected_segments"),
    [(KVCacheLayout.LBNHC, NUM_LAYERS), (KVCacheLayout.BLNHC, 1)],
)
def test_segments_follow_layout(vmm, layout, expected_segments):
    """Layer-outermost layouts commit one prefix per layer; block-outermost
    layouts commit a single prefix of the whole allocation."""
    config, spec = _make_config(layout)
    kv_cache = ExtensibleKVCache(config, torch.device("cuda"))
    try:
        assert kv_cache.buffer.num_segments == expected_segments
        assert kv_cache.bytes_per_block == NUM_LAYERS * spec.page_size_bytes
        assert kv_cache.num_committed_blocks == 0
    finally:
        kv_cache.free()


@requires_cuda
@pytest.mark.parametrize("layout", [KVCacheLayout.LBNHC, KVCacheLayout.BLNHC])
def test_views_stay_valid_across_commits(vmm, layout):
    """Layer views built over the reservation address committed blocks at
    fixed offsets: data written before a commit survives it and newly
    committed blocks read as zero."""
    config, spec = _make_config(layout)
    device = torch.device("cuda")
    kv_cache = ExtensibleKVCache(config, device)
    try:
        kv_caches = allocate_kv_cache(
            config, device, layout, allocate=kv_cache.allocate
        )
        assert len(kv_caches) == NUM_LAYERS
        views = list(kv_caches.values())
        for view in views:
            assert view.shape[0] == NUM_BLOCKS
            assert view.untyped_storage().nbytes() == kv_cache.size

        kv_cache.commit(2)
        assert kv_cache.num_committed_blocks == 2
        for i, view in enumerate(views):
            view[:2].fill_(float(i + 1))
        torch.accelerator.synchronize()

        kv_cache.commit(NUM_BLOCKS)
        assert kv_cache.num_committed_blocks == NUM_BLOCKS
        assert kv_cache.physical_bytes >= kv_cache.size
        for i, view in enumerate(views):
            assert torch.all(view[:2] == float(i + 1))
            assert torch.count_nonzero(view[2:]) == 0

        # Grow-only and clamped to capacity.
        kv_cache.commit(NUM_BLOCKS + 5)
        assert kv_cache.num_committed_blocks == NUM_BLOCKS
    finally:
        kv_cache.free()


def test_measure_kv_cache_bytes():
    gib = 1 << 30
    # 10 GiB were free at startup; 3 GiB of non-KV memory is now resident and
    # 1 GiB is committed to the KV cache, leaving 6 GiB free. Budget is 9 GiB.
    common = dict(
        init_free_memory=10 * gib,
        free_memory=6 * gib,
        committed_bytes=1 * gib,
        margin_floor_bytes=0,
        margin_fraction=0.0,
    )
    # Budget-bound: 9 - 3 = 6 GiB for the KV cache.
    assert measure_kv_cache_bytes(requested_memory=9 * gib, **common) == 6 * gib
    # Headroom-bound: at most what is free plus what is already committed.
    assert measure_kv_cache_bytes(requested_memory=20 * gib, **common) == 7 * gib
    # The measured transient peak is kept free in full, on either bound.
    assert (
        measure_kv_cache_bytes(
            requested_memory=20 * gib, transient_peak_bytes=gib, **common
        )
        == 6 * gib
    )
    assert (
        measure_kv_cache_bytes(
            requested_memory=9 * gib, transient_peak_bytes=gib, **common
        )
        == 5 * gib
    )
    # The margin is a share of that peak with a floor, taken off the headroom
    # (floor or fraction, whichever is larger), never off an explicit budget
    # that already leaves it free.
    assert (
        measure_kv_cache_bytes(
            requested_memory=20 * gib,
            transient_peak_bytes=gib,
            **{**common, "margin_floor_bytes": gib},
        )
        == 5 * gib
    )
    assert (
        measure_kv_cache_bytes(
            requested_memory=20 * gib,
            transient_peak_bytes=2 * gib,
            **{**common, "margin_fraction": 0.5},
        )
        == 4 * gib
    )
    assert (
        measure_kv_cache_bytes(
            requested_memory=9 * gib, **{**common, "margin_floor_bytes": gib}
        )
        == 6 * gib
    )
    # A small explicit budget (0.5 GiB) survives a floor larger than itself.
    assert (
        measure_kv_cache_bytes(
            requested_memory=3 * gib + gib // 2, **{**common, "margin_floor_bytes": gib}
        )
        == gib // 2
    )
    # The result never goes negative.
    assert measure_kv_cache_bytes(requested_memory=2 * gib, **common) == 0


def test_granule_aligned_blocks():
    """Blocks per stride must span whole granules; with several strides the
    count is a multiple of every stride's requirement."""
    granule = 2 << 20
    # 64 KiB blocks: 32 per granule.
    assert granule_block_alignment([64 << 10], granule) == 32
    assert granule_aligned_blocks(1000, [64 << 10], granule) == 992
    # 96 KiB blocks: 3 blocks cover 288 KiB; the granule is reached at 64.
    assert granule_block_alignment([96 << 10], granule) == 64
    # Mixed strides: lcm(32, 64).
    assert granule_aligned_blocks(1000, [64 << 10, 96 << 10], granule) == 960
    # A stride that is a granule multiple needs no alignment.
    assert granule_aligned_blocks(1000, [4 << 20], granule) == 1000
    assert granule_aligned_blocks(31, [64 << 10], granule) == 0


def _make_mixed_config(num_blocks: int = NUM_BLOCKS):
    """One group whose layers differ in head size, so their block strides differ."""
    specs = {
        f"layer.{i}": FullAttentionSpec(
            block_size=16,
            num_kv_heads=2,
            head_size=64 * (1 + i % 2),
            dtype=torch.float16,
        )
        for i in range(NUM_LAYERS)
    }
    uniform = UniformTypeKVCacheSpecs.from_specs(specs)
    assert uniform is not None
    vllm_config = VllmConfig(model_config=ModelConfig(max_model_len=16))
    vllm_config.cache_config.kv_cache_layout = "LBNHC"
    config = get_kv_cache_config_from_groups(
        vllm_config,
        [KVCacheGroupSpec(list(specs), uniform)],
        available_memory=sum(s.page_size_bytes for s in specs.values()) * num_blocks,
    )
    assert config.num_blocks == num_blocks
    assert len({t.block_stride for t in config.kv_cache_tensors}) == 2
    return config, specs


@requires_cuda
def test_mixed_block_strides_commit_per_layer(vmm):
    """Layers with different page sizes get segments of their own stride; a
    commit backs the same block count in each and the views stay valid."""
    config, specs = _make_mixed_config()
    device = torch.device("cuda")
    kv_cache = ExtensibleKVCache(config, device)
    try:
        assert len(kv_cache.segment_strides) == NUM_LAYERS
        assert sorted(kv_cache.segment_strides) == sorted(
            s.page_size_bytes for s in specs.values()
        )
        views = allocate_kv_cache(
            config, device, KVCacheLayout.LBNHC, allocate=kv_cache.allocate
        )
        kv_cache.commit(3)
        for i, view in enumerate(views.values()):
            view[:3].fill_(float(i + 1))
        torch.accelerator.synchronize()
        assert kv_cache.physical_bytes >= 3 * kv_cache.bytes_per_block
        for i, view in enumerate(views.values()):
            assert torch.all(view[:3] == float(i + 1))

        final = copy.copy(config)
        final.num_blocks = 3
        committed = kv_cache.committed_views(final, KVCacheLayout.LBNHC)
        assert committed is not None
        for name, view in committed.items():
            assert view.shape[0] == 3
            assert view.data_ptr() == views[name].data_ptr()
            assert view.untyped_storage().nbytes() == 3 * specs[name].page_size_bytes
    finally:
        kv_cache.free()


@requires_cuda
def test_release_and_recommit_for_sleep(vmm):
    """Sleep drops the pages; wake maps fresh zeroed pages for the same block
    count under the same addresses, so existing views keep working."""
    config, _ = _make_config(KVCacheLayout.LBNHC)
    device = torch.device("cuda")
    kv_cache = ExtensibleKVCache(config, device)
    try:
        views = list(
            allocate_kv_cache(
                config, device, KVCacheLayout.LBNHC, allocate=kv_cache.allocate
            ).values()
        )
        kv_cache.commit(NUM_BLOCKS)
        views[0].fill_(3.0)
        torch.accelerator.synchronize()

        kv_cache.release_physical()
        assert kv_cache.physical_bytes == 0
        assert kv_cache.num_committed_blocks == NUM_BLOCKS

        kv_cache.recommit()
        assert kv_cache.physical_bytes >= kv_cache.size
        assert torch.count_nonzero(views[0]) == 0
    finally:
        kv_cache.free()


@requires_cuda
@pytest.mark.parametrize("layout", [KVCacheLayout.LBNHC, KVCacheLayout.BLNHC])
def test_shrinking_commit_remaps_to_the_smaller_prefix(vmm, layout):
    """When fewer blocks fit than warmup committed, the final commit remaps
    only that prefix (contents are warmup garbage), so the physical footprint,
    the views and the connector-facing placements all describe the same,
    smaller cache."""
    config, _ = _make_config(layout)
    device = torch.device("cuda")
    kv_cache = ExtensibleKVCache(config, device)
    try:
        capacity_views = allocate_kv_cache(
            config, device, layout, allocate=kv_cache.allocate
        )
        kv_cache.commit(6)
        for view in capacity_views.values():
            view[:6].fill_(1.0)
        torch.accelerator.synchronize()
        # Without `shrink` a smaller request is a no-op.
        kv_cache.commit(4)
        assert kv_cache.num_committed_blocks == 6

        kv_cache.commit(4, shrink=True)
        assert kv_cache.num_committed_blocks == 4
        granule = kv_cache.buffer.granularity
        expected_physical = sum(
            -(-4 * stride // granule) * granule for stride in kv_cache.segment_strides
        )
        assert kv_cache.physical_bytes <= expected_physical
        final = copy.copy(config)
        final.num_blocks = 4
        views = kv_cache.committed_views(final, layout)
        assert views is not None
        for name, view in views.items():
            assert view.shape[0] == 4
            assert view.data_ptr() == capacity_views[name].data_ptr()
            assert torch.count_nonzero(view) == 0
        tensors = kv_cache.committed_kv_cache_tensors(final, layout, 4)
        assert all(t.size == 4 * kv_cache.bytes_per_block for t in tensors)
    finally:
        kv_cache.free()


@requires_cuda
@pytest.mark.parametrize("layout", [KVCacheLayout.LBNHC, KVCacheLayout.BLNHC])
def test_committed_views_cover_exactly_the_committed_bytes(vmm, layout):
    """After the final commit, per-layer views are rebuilt over storages that
    span only the committed blocks, at the same addresses as the capacity
    views, so connectors deriving extents from storage size see backed memory."""
    config, spec = _make_config(layout)
    device = torch.device("cuda")
    kv_cache = ExtensibleKVCache(config, device)
    try:
        capacity_views = allocate_kv_cache(
            config, device, layout, allocate=kv_cache.allocate
        )
        committed = 5
        kv_cache.commit(committed)
        for i, view in enumerate(capacity_views.values()):
            view[:committed].fill_(float(i + 1))
        torch.accelerator.synchronize()

        final_config = copy.copy(config)
        final_config.num_blocks = committed
        views = kv_cache.committed_views(final_config, layout)
        assert views is not None and views.keys() == capacity_views.keys()
        for i, (name, view) in enumerate(views.items()):
            assert view.shape[0] == committed
            assert view.data_ptr() == capacity_views[name].data_ptr()
            assert torch.all(view == float(i + 1))
            block_stride = view.stride(0) * view.element_size()
            if layout.is_block_outermost:
                assert view.untyped_storage().nbytes() == committed * block_stride
            else:
                assert (
                    view.untyped_storage().nbytes() == committed * spec.page_size_bytes
                )
    finally:
        kv_cache.free()


def _segments_backed_by_one_chunk(kv_cache: ExtensibleKVCache) -> list[bool]:
    """Whether each segment's committed bytes lie within one driver allocation."""
    return kv_cache.buffer.segments_backed_by_one_chunk(
        [kv_cache.num_committed_blocks * s for s in kv_cache.segment_strides]
    )


def _granule(layout: KVCacheLayout) -> int:
    probe = ExtensibleKVCache(_make_config(layout)[0], torch.device("cuda"))
    try:
        return probe.buffer.granularity
    finally:
        probe.free()


@requires_cuda
def test_defragmenting_commit_backs_each_segment_with_one_allocation(vmm):
    """Incremental commits map one driver allocation each; a defragmenting
    commit of a granule-aligned count remaps every segment as a single
    allocation (RDMA cannot span several) and still ends up zero-filled."""
    granule = _granule(KVCacheLayout.LBNHC)
    stride = _make_config(KVCacheLayout.LBNHC)[1].page_size_bytes
    alignment = granule_block_alignment([stride], granule)
    # An unaligned capacity puts segment offsets inside granules.
    capacity = 2 * alignment + alignment // 2
    config, _ = _make_config(KVCacheLayout.LBNHC, num_blocks=capacity)
    kv_cache = ExtensibleKVCache(config, torch.device("cuda"))
    try:
        assert kv_cache.block_alignment == alignment
        kv_cache.commit(1)
        kv_cache.commit(alignment + 1)
        num_segments = kv_cache.buffer.num_segments
        assert kv_cache.buffer.num_physical_chunks == 2 * num_segments

        kv_cache.commit(capacity, defragment=True)
        assert kv_cache.num_committed_blocks == capacity
        assert torch.count_nonzero(kv_cache.buffer.full_view()) == 0
        # Segments after the first start mid-granule and share chunks.
        assert not all(_segments_backed_by_one_chunk(kv_cache))
    finally:
        kv_cache.free()

    config, _ = _make_config(KVCacheLayout.LBNHC, num_blocks=2 * alignment)
    kv_cache = ExtensibleKVCache(config, torch.device("cuda"))
    try:
        kv_cache.commit(alignment + 1)
        kv_cache.commit(
            kv_cache.blocks_within(kv_cache.size, aligned=True), defragment=True
        )
        assert kv_cache.num_committed_blocks == 2 * alignment
        buffer = kv_cache.buffer
        assert buffer.num_physical_chunks == buffer.num_segments
        assert all(_segments_backed_by_one_chunk(kv_cache))
    finally:
        kv_cache.free()


@requires_cuda
@pytest.mark.parametrize("layout", [KVCacheLayout.LBNHC, KVCacheLayout.BLNHC])
def test_aligned_capacity_config_allocates_and_defragments(vmm, layout):
    """The engine-side alignment yields a config the cache can be built from,
    with views over it, and whose full defragmenting commit backs each segment
    with one allocation."""
    from vllm.v1.core.kv_cache_utils import (
        align_extensible_kv_cache_capacity,
        generate_scheduler_kv_cache_config,
    )

    granule = _granule(layout)
    config, spec = _make_config(layout, num_blocks=1000)
    vllm_config = VllmConfig(model_config=ModelConfig(max_model_len=16))
    scheduler_config = generate_scheduler_kv_cache_config([config])
    align_extensible_kv_cache_capacity(vllm_config, [config], scheduler_config, granule)
    assert config.num_blocks < 1000
    device = torch.device("cuda")
    kv_cache = ExtensibleKVCache(config, device)
    try:
        views = allocate_kv_cache(config, device, layout, allocate=kv_cache.allocate)
        assert all(v.shape[0] == config.num_blocks for v in views.values())
        kv_cache.commit(3)
        kv_cache.commit(config.num_blocks, defragment=True)
        assert all(_segments_backed_by_one_chunk(kv_cache))
        for view in views.values():
            view.fill_(1.0)
        torch.accelerator.synchronize()
    finally:
        kv_cache.free()


@requires_cuda
def test_blocks_within_accounts_for_granule_rounding(vmm):
    """The count that fits a byte budget is the most whose rounded-up
    per-segment footprint fits, not the budget over the nominal block size;
    aligned, it is the largest aligned count under the nominal footprint."""
    config, spec = _make_config(KVCacheLayout.LBNHC, num_blocks=4096)
    kv_cache = ExtensibleKVCache(config, torch.device("cuda"))
    try:
        granule = kv_cache.buffer.granularity
        alignment = kv_cache.block_alignment
        num_segments = kv_cache.buffer.num_segments
        # One granule per segment holds `alignment` blocks exactly; one more
        # block per segment needs a whole extra granule each.
        budget = num_segments * granule
        assert kv_cache.blocks_within(budget) == alignment
        assert kv_cache.blocks_within(budget + 1) == alignment
        assert kv_cache.physical_bytes_for(alignment + 1) == 2 * budget
        assert kv_cache.blocks_within(2 * budget - 1) == alignment
        assert kv_cache.blocks_within(2 * budget) == 2 * alignment
        nominal = (2 * alignment + 5) * kv_cache.bytes_per_block
        assert kv_cache.blocks_within(nominal, aligned=True) == 2 * alignment
        assert kv_cache.blocks_within(kv_cache.size * 2) == config.num_blocks
    finally:
        kv_cache.free()


@requires_cuda
@pytest.mark.parametrize("layout", [KVCacheLayout.LBNHC, KVCacheLayout.BLNHC])
def test_committed_kv_cache_tensors_match_views(vmm, layout):
    """The connector-facing placements locate every block of every layer at
    the same bytes as the committed views, within storages of their size."""
    config, _ = _make_config(layout)
    device = torch.device("cuda")
    kv_cache = ExtensibleKVCache(config, device)
    try:
        allocate_kv_cache(config, device, layout, allocate=kv_cache.allocate)
        committed = 5
        kv_cache.commit(committed)
        final = copy.copy(config)
        final.num_blocks = committed
        views = kv_cache.committed_views(final, layout)
        tensors = kv_cache.committed_kv_cache_tensors(final, layout, committed)
        assert views is not None
        assert {name for t in tensors for name in t.layers} == set(views)
        for tensor in tensors:
            assert isinstance(tensor, KVCacheTensor)
            # Consumers read any tensor's size as the whole allocation's.
            assert tensor.size == committed * kv_cache.bytes_per_block
            for layer_idx, name in enumerate(tensor.layers):
                view = views[name]
                storage = view.untyped_storage()
                layer_start = tensor.offset + layer_idx * tensor.layer_stride
                page_bytes = view[0].numel() * view.element_size()
                last_page_end = (
                    layer_start + (committed - 1) * tensor.block_stride + page_bytes
                )
                assert storage.nbytes() >= last_page_end
                assert storage.data_ptr() + layer_start == view.data_ptr()
                assert view.stride(0) * view.element_size() == tensor.block_stride
    finally:
        kv_cache.free()
