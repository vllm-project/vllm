# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy

import pytest
import torch

from vllm.config import ModelConfig, VllmConfig
from vllm.utils.vmm_driver import vmm_unavailable_reason
from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.extensible_kv_cache import (
    ExtensibleKVCache,
    measure_kv_cache_blocks,
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


def test_measure_kv_cache_blocks():
    gib = 1 << 30
    # 10 GiB were free at startup; 3 GiB of non-KV memory is now resident and
    # 1 GiB is committed to the KV cache, leaving 6 GiB free. Budget is 9 GiB.
    common = dict(
        init_free_memory=10 * gib,
        free_memory=6 * gib,
        committed_bytes=1 * gib,
        bytes_per_block=gib // 4,
        margin_floor_bytes=0,
        margin_fraction=0.0,
    )
    # Budget-bound: 9 - 3 = 6 GiB for the KV cache.
    assert measure_kv_cache_blocks(requested_memory=9 * gib, **common) == 24
    # Headroom-bound: at most what is free plus what is already committed.
    assert measure_kv_cache_blocks(requested_memory=20 * gib, **common) == 28
    # The margin comes off the headroom (floor or fraction, whichever is
    # larger), not off an explicit budget that already leaves it free.
    assert (
        measure_kv_cache_blocks(
            requested_memory=20 * gib, **{**common, "margin_floor_bytes": gib}
        )
        == 24
    )
    assert (
        measure_kv_cache_blocks(
            requested_memory=20 * gib, **{**common, "margin_fraction": 0.5}
        )
        == 14
    )
    assert (
        measure_kv_cache_blocks(
            requested_memory=9 * gib, **{**common, "margin_floor_bytes": gib}
        )
        == 24
    )
    # A small explicit budget (0.5 GiB) survives a floor larger than itself.
    assert (
        measure_kv_cache_blocks(
            requested_memory=3 * gib + gib // 2, **{**common, "margin_floor_bytes": gib}
        )
        == 2
    )
    # Extra margin always applies, and the result never goes negative.
    assert (
        measure_kv_cache_blocks(
            requested_memory=9 * gib, extra_margin_bytes=gib, **common
        )
        == 20
    )
    assert measure_kv_cache_blocks(requested_memory=2 * gib, **common) == 0


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


@requires_cuda
def test_defragmenting_commit_leaves_one_allocation_per_segment(vmm):
    """Incremental commits map one driver allocation each; a defragmenting
    commit remaps every segment as a single allocation (RDMA cannot span
    several) and still ends up zero-filled and fully committed."""
    config, _ = _make_config(KVCacheLayout.LBNHC, num_blocks=1024)
    kv_cache = ExtensibleKVCache(config, torch.device("cuda"))
    try:
        granule_blocks = kv_cache.buffer.granularity // kv_cache.block_stride
        kv_cache.commit(1)
        kv_cache.commit(2 * granule_blocks + 1)
        num_segments = kv_cache.buffer.num_segments
        assert kv_cache.buffer.num_physical_chunks == 2 * num_segments

        kv_cache.commit(config.num_blocks, defragment=True)
        assert kv_cache.num_committed_blocks == config.num_blocks
        assert kv_cache.buffer.num_physical_chunks == num_segments
        assert torch.count_nonzero(kv_cache.buffer.full_view()) == 0
    finally:
        kv_cache.free()
