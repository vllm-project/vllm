# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.config import ModelConfig, VllmConfig
from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.extensible_kv_cache import (
    ExtensibleKVCache,
    measure_kv_cache_blocks,
)
from vllm.v1.worker.utils import allocate_kv_cache

NUM_BLOCKS = 8
NUM_LAYERS = 3


def _make_config(layout: KVCacheLayout):
    spec = FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=64, dtype=torch.float16
    )
    vllm_config = VllmConfig(model_config=ModelConfig(max_model_len=16))
    vllm_config.cache_config.kv_cache_layout = layout.name
    layers = [f"layer.{i}" for i in range(NUM_LAYERS)]
    config = get_kv_cache_config_from_groups(
        vllm_config,
        [KVCacheGroupSpec(layers, spec)],
        available_memory=NUM_LAYERS * spec.page_size_bytes * NUM_BLOCKS,
    )
    assert config.num_blocks == NUM_BLOCKS
    return config, spec


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("layout", "expected_segments"),
    [(KVCacheLayout.LBNHC, NUM_LAYERS), (KVCacheLayout.BLNHC, 1)],
)
def test_segments_follow_layout(layout, expected_segments):
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("layout", [KVCacheLayout.LBNHC, KVCacheLayout.BLNHC])
def test_views_stay_valid_across_commits(layout):
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
        margin_bytes=0,
    )
    # Budget-bound: 9 - 3 = 6 GiB for the KV cache.
    assert measure_kv_cache_blocks(requested_memory=9 * gib, **common) == 24
    # Free-bound: at most what is free plus what is already committed.
    assert measure_kv_cache_blocks(requested_memory=20 * gib, **common) == 28
    # The margin comes off the top and the result never goes negative.
    assert (
        measure_kv_cache_blocks(
            requested_memory=9 * gib, **{**common, "margin_bytes": gib}
        )
        == 20
    )
    assert measure_kv_cache_blocks(requested_memory=2 * gib, **common) == 0
