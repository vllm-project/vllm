# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
from types import SimpleNamespace

import pytest
import torch

from vllm.config import ModelConfig, VllmConfig
from vllm.utils.vmm_driver import vmm_unavailable_reason
from vllm.v1.core.kv_cache_planning import get_kv_cache_config_from_groups
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    KVCacheTensor,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker import extensible_kv_cache
from vllm.v1.worker.extensible_kv_cache import (
    ExtensibleKVCache,
    measure_kv_cache_bytes,
    pad_kv_cache_segments,
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
    kv_cache = ExtensibleKVCache(config, torch.device("cuda"), layout)
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
    kv_cache = ExtensibleKVCache(config, device, layout)
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


def test_measured_sizing_caps_torch_memory(monkeypatch):
    """Torch's caching allocator may grow into what stays free once the cache
    is extended, less multimodal IPC memory and the non-torch reserve,
    rather than caching fragmented segments until other allocations fail."""
    gib = 1 << 30
    limits: list[int | None] = []
    kv_cache = SimpleNamespace(
        reserved_headroom_bytes=gib,
        physical_bytes=gib,
        bytes_per_block=gib,
        blocks_within=lambda num_bytes: num_bytes // gib,
        physical_bytes_for=lambda num_blocks: num_blocks * gib,
        set_torch_memory_limit=limits.append,
    )
    accelerator = extensible_kv_cache.torch.accelerator
    monkeypatch.setattr(accelerator, "synchronize", lambda *args: None)
    monkeypatch.setattr(accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(accelerator, "get_memory_info", lambda *args: (6 * gib, 0))
    monkeypatch.setattr(accelerator, "memory_reserved", lambda *args: 2 * gib)
    monkeypatch.setattr(
        extensible_kv_cache, "measure_kv_cache_bytes", lambda **kwargs: 5 * gib
    )
    # One GiB of the 5 GiB budget is held for multimodal IPC.
    monkeypatch.setattr(
        extensible_kv_cache,
        "reserve_mm_ipc_gpu_memory",
        lambda available, *args: available - gib,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(multimodal_config=None),
        parallel_config=SimpleNamespace(),
        kv_transfer_config=None,
    )

    num_blocks = extensible_kv_cache.measure_kv_cache_blocks(
        kv_cache,  # type: ignore[arg-type]
        vllm_config,  # type: ignore[arg-type]
        torch.device("cpu"),
        init_free_memory=10 * gib,
        requested_memory=10 * gib,
    )

    assert num_blocks == 4
    # 6 GiB free, of which extending to 4 blocks takes 3 and IPC keeps 1.
    left_free = 2 * gib - extensible_kv_cache.KV_CACHE_NON_TORCH_RESERVE_BYTES
    assert limits == [2 * gib + left_free]


def test_freeing_the_cache_lifts_the_torch_memory_cap(monkeypatch):
    """The process may start another engine, which must not inherit the cap."""
    fractions: list[float] = []
    monkeypatch.setattr(
        torch.accelerator, "get_memory_info", lambda *args: (0, 100 << 30)
    )
    monkeypatch.setattr(
        torch.cuda,
        "set_per_process_memory_fraction",
        lambda fraction, device: fractions.append(fraction),
    )
    kv_cache = object.__new__(ExtensibleKVCache)
    kv_cache.buffer = SimpleNamespace(device=torch.device("cpu"), free=lambda: None)
    kv_cache.num_committed_blocks = 2
    kv_cache.set_torch_memory_limit(25 << 30)
    kv_cache.free()
    assert fractions == [0.25, 1.0]
    assert kv_cache.torch_memory_limit is None


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


def test_pad_kv_cache_segments():
    """Layer-outermost regions are placed on granules: offsets, layer strides
    and the size round up while block count and strides stay. Block-outermost
    layouts have one segment at offset 0 and are left as they are."""
    granule = 2 << 20
    page = _make_config(KVCacheLayout.LBNHC)[1].page_size_bytes
    # One block past a granule per layer.
    num_blocks = granule // page + 1
    config, _ = _make_config(KVCacheLayout.LBNHC, num_blocks=num_blocks)
    padded = pad_kv_cache_segments(config, KVCacheLayout.LBNHC, granule)
    assert padded.num_blocks == num_blocks
    (tensor,) = padded.kv_cache_tensors
    assert tensor.block_stride == page
    assert tensor.layer_stride == 2 * granule
    assert tensor.offset == 0
    assert tensor.size == NUM_LAYERS * 2 * granule
    assert config.kv_cache_tensors[0].layer_stride == num_blocks * page

    # Layers 0 and 2 share a stride, so one tensor places both ahead of layer 1.
    config, _ = _make_mixed_config()
    padded = pad_kv_cache_segments(config, KVCacheLayout.LBNHC, granule)
    assert [t.layers for t in padded.kv_cache_tensors] == [
        ["layer.0", "layer.2"],
        ["layer.1"],
    ]
    assert [t.offset for t in padded.kv_cache_tensors] == [0, 2 * granule]
    assert all(t.layer_stride == granule for t in padded.kv_cache_tensors)
    assert {t.size for t in padded.kv_cache_tensors} == {NUM_LAYERS * granule}

    config, _ = _make_config(KVCacheLayout.BLNHC)
    assert pad_kv_cache_segments(config, KVCacheLayout.BLNHC, granule) is config


@requires_cuda
def test_mixed_block_strides_commit_per_layer(vmm):
    """Layers with different page sizes get segments of their own stride; a
    commit backs the same block count in each and the views stay valid."""
    config, specs = _make_mixed_config()
    device = torch.device("cuda")
    kv_cache = ExtensibleKVCache(config, device, KVCacheLayout.LBNHC)
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
    kv_cache = ExtensibleKVCache(config, device, KVCacheLayout.LBNHC)
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
    kv_cache = ExtensibleKVCache(config, device, layout)
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
    kv_cache = ExtensibleKVCache(config, device, layout)
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
    probe = ExtensibleKVCache(_make_config(layout)[0], torch.device("cuda"), layout)
    try:
        return probe.buffer.granularity
    finally:
        probe.free()


@requires_cuda
@pytest.mark.parametrize("padded", [False, True])
def test_defragmenting_commit_backs_each_segment_with_one_allocation(vmm, padded):
    """Incremental commits map one driver allocation each. Unpadded, segments
    after the first start mid-granule and share allocations even after a
    defragmenting commit; padded, a defragmenting commit of any block count
    remaps each segment as one allocation (RDMA cannot span several)."""
    layout = KVCacheLayout.LBNHC
    granule = _granule(layout)
    page = _make_config(layout)[1].page_size_bytes
    blocks_per_granule = granule // page
    capacity = 2 * blocks_per_granule + blocks_per_granule // 2
    config, _ = _make_config(layout, num_blocks=capacity)
    if padded:
        config = pad_kv_cache_segments(config, layout, granule)
    kv_cache = ExtensibleKVCache(config, torch.device("cuda"), layout)
    try:
        kv_cache.commit(1)
        kv_cache.commit(blocks_per_granule + 1)
        buffer = kv_cache.buffer
        assert buffer.num_segments == NUM_LAYERS
        assert buffer.num_physical_chunks == 2 * NUM_LAYERS

        kv_cache.commit(capacity - 3, defragment=True)
        assert kv_cache.num_committed_blocks == capacity - 3
        for segment in range(buffer.num_segments):
            assert torch.count_nonzero(buffer.segment_view(segment)) == 0
        assert all(_segments_backed_by_one_chunk(kv_cache)) == padded
        if padded:
            assert buffer.num_physical_chunks == NUM_LAYERS
    finally:
        kv_cache.free()


@requires_cuda
@pytest.mark.parametrize("case", ["LBNHC", "BLNHC", "mixed"])
def test_padded_config_allocates_and_defragments(vmm, case):
    """A padded config yields a cache with views over its reservation; a
    defragmenting commit of an arbitrary count backs each segment with one
    allocation, and the committed views and connector placements address the
    same bytes as the capacity views."""
    layout = KVCacheLayout.LBNHC if case == "mixed" else KVCacheLayout[case]
    granule = _granule(layout)
    capacity = 1000
    if case == "mixed":
        config, _ = _make_mixed_config(num_blocks=capacity)
    else:
        config, _ = _make_config(layout, num_blocks=capacity)
    config = pad_kv_cache_segments(config, layout, granule)
    device = torch.device("cuda")
    kv_cache = ExtensibleKVCache(config, device, layout)
    try:
        assert all(offset % granule == 0 for offset in kv_cache.segment_offsets)
        views = allocate_kv_cache(config, device, layout, allocate=kv_cache.allocate)
        assert all(v.shape[0] == capacity for v in views.values())
        kv_cache.commit(3)
        num_blocks = capacity - 3
        kv_cache.commit(num_blocks, defragment=True)
        assert all(_segments_backed_by_one_chunk(kv_cache))
        for i, view in enumerate(views.values()):
            view[:num_blocks].fill_(float(i + 1))
        torch.accelerator.synchronize()

        final = copy.copy(config)
        final.num_blocks = num_blocks
        committed = kv_cache.committed_views(final, layout)
        assert committed is not None
        for i, (name, view) in enumerate(views.items()):
            assert committed[name].data_ptr() == view.data_ptr()
            assert torch.all(committed[name] == float(i + 1))
        for tensor in kv_cache.committed_kv_cache_tensors(final, layout, num_blocks):
            for layer_idx, name in enumerate(tensor.layers):
                storage = committed[name].untyped_storage()
                layer_start = tensor.offset + layer_idx * tensor.layer_stride
                assert storage.data_ptr() + layer_start == committed[name].data_ptr()
    finally:
        kv_cache.free()


@requires_cuda
def test_blocks_within_accounts_for_granule_rounding(vmm):
    """The count that fits a byte budget is the most whose rounded-up
    per-segment footprint fits, not the budget over the nominal block size."""
    config, spec = _make_config(KVCacheLayout.LBNHC, num_blocks=4096)
    kv_cache = ExtensibleKVCache(config, torch.device("cuda"), KVCacheLayout.LBNHC)
    try:
        granule = kv_cache.buffer.granularity
        blocks_per_granule = granule // spec.page_size_bytes
        num_segments = kv_cache.buffer.num_segments
        # One granule per segment holds `blocks_per_granule` blocks exactly;
        # one more block per segment needs a whole extra granule each.
        budget = num_segments * granule
        assert kv_cache.blocks_within(budget) == blocks_per_granule
        assert kv_cache.blocks_within(budget + 1) == blocks_per_granule
        assert kv_cache.physical_bytes_for(blocks_per_granule + 1) == 2 * budget
        assert kv_cache.blocks_within(2 * budget - 1) == blocks_per_granule
        assert kv_cache.blocks_within(2 * budget) == 2 * blocks_per_granule
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
    kv_cache = ExtensibleKVCache(config, device, layout)
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
