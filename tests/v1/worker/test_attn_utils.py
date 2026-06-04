# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Padded-page handling in reshape_kv_cache.

Guards that a page_size_padded spec strides the block dimension by the
padded page while keeping per-block content compact, so padding bytes at
the end of each page are never addressed by the logical view.
"""

import pytest
import torch

from vllm.v1.attention.backends.utils import set_kv_cache_layout
from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheLayout,
    KVCacheTensor,
    KVQuantMode,
    MLAAttentionSpec,
    reshape_kv_cache,
)
from vllm.v1.worker.gpu.attn_utils import _allocate_and_reshape_kv_cache
from vllm.v1.worker.utils import copy_kv_cache_blocks_inplace


def test_reshape_padded_kv_cache_strides_by_padded_page():
    num_blocks = 3
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
        page_size_padded=384,
    )
    assert spec.real_page_size_bytes == 256

    raw = torch.zeros(spec.page_size_bytes * num_blocks, dtype=torch.int8)
    (kv_cache,) = reshape_kv_cache(raw, spec, num_blocks, 1, KVCacheLayout.LBHNC)

    elem_size = 4  # float32
    # Content dim packs K and V: 2 * head_size.
    assert kv_cache.shape == (num_blocks, 1, 16, 2 * spec.head_size)
    assert kv_cache.dtype == spec.dtype
    assert kv_cache.stride(0) == spec.page_size_padded // elem_size
    assert kv_cache[1].storage_offset() == spec.page_size_padded // elem_size
    # Within one block the (unpadded) content stays compact.
    assert kv_cache[0].is_contiguous()


def test_reshape_padded_quantized_kv_cache_budgets_scale_bytes():
    num_blocks = 3
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.int8,
        kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
        page_size_padded=384,
    )
    # Per-token-head scales are budgeted into the page but live past the
    # real content, so the logical view must stride by the padded page.
    assert spec.real_page_size_bytes == 128
    assert spec.page_size_bytes == 384

    raw = torch.zeros(spec.page_size_bytes * num_blocks, dtype=torch.int8)
    (kv_cache,) = reshape_kv_cache(raw, spec, num_blocks, 1, KVCacheLayout.LBHNC)

    assert kv_cache.shape == (num_blocks, 1, 16, 2 * spec.head_size)
    assert kv_cache.stride(0) == spec.page_size_padded
    assert kv_cache[1].storage_offset() == spec.page_size_padded


@pytest.mark.parametrize(
    ("kernel_block_sizes", "expected_num_blocks", "expected_num_states"),
    [
        (None, 4, 64),
        ([256], 4, 64),
        ([64], 16, 16),
    ],
)
def test_allocate_compressed_mla_cache(
    kernel_block_sizes: list[int] | None,
    expected_num_blocks: int,
    expected_num_states: int,
):
    spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        tokens_per_state=4,
    )
    num_pages = 4
    config = KVCacheConfig(
        num_blocks=num_pages,
        kv_cache_tensors=[
            KVCacheTensor(
                size=num_pages * spec.page_size_bytes,
                shared_by=[["layer.0"]],
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(["layer.0"], spec)],
    )

    caches = _allocate_and_reshape_kv_cache(
        config,
        torch.device("cpu"),
        layout=KVCacheLayout.LBHNC,
        kernel_block_sizes=kernel_block_sizes,
    )

    assert caches["layer.0"].shape == (
        expected_num_blocks,
        1,
        expected_num_states,
        128,
    )


@pytest.mark.parametrize("layout", list(KVCacheLayout))
def test_copy_kv_cache_blocks_shared_storage(layout: KVCacheLayout):
    num_blocks = 4
    num_layers = 2
    spec = FullAttentionSpec(
        block_size=2,
        num_kv_heads=2,
        head_size=2,
        dtype=torch.float32,
    )
    raw = torch.zeros(
        num_blocks * num_layers * spec.page_size_bytes,
        dtype=torch.int8,
    )
    caches = reshape_kv_cache(raw, spec, num_blocks, num_layers, layout)

    for layer_idx, cache in enumerate(caches):
        for block_idx in range(num_blocks):
            cache[block_idx].fill_(10 * layer_idx + block_idx)

    expected = [[cache[i].clone() for i in range(num_blocks)] for cache in caches]
    copies = [KVCacheBlockCopy(src_block_id=0, dst_block_id=2)]

    set_kv_cache_layout(layout.name)
    try:
        copy_kv_cache_blocks_inplace(caches, num_blocks, copies)
    finally:
        set_kv_cache_layout(None)

    for layer_idx, cache in enumerate(caches):
        torch.testing.assert_close(cache[2], expected[layer_idx][0])
        torch.testing.assert_close(cache[1], expected[layer_idx][1])


@pytest.mark.parametrize("layout", [KVCacheLayout.BLHNC, KVCacheLayout.BHLNC])
def test_copy_kv_cache_blocks_separate_head_groups(layout: KVCacheLayout):
    num_blocks = 4
    num_layers = 2
    spec = FullAttentionSpec(
        block_size=2,
        num_kv_heads=2,
        head_size=2,
        dtype=torch.float32,
        separate_kv_head_groups=True,
    )
    raw = torch.zeros(
        num_blocks * num_layers * spec.page_size_bytes,
        dtype=torch.int8,
    )
    caches = reshape_kv_cache(raw, spec, num_blocks, num_layers, layout)

    for layer_idx, cache in enumerate(caches):
        for block_idx in range(num_blocks):
            for head_idx in range(cache.shape[1]):
                cache[block_idx, head_idx].fill_(
                    100 * layer_idx + 10 * head_idx + block_idx
                )

    expected = [[cache[i].clone() for i in range(num_blocks)] for cache in caches]
    set_kv_cache_layout(layout.name)
    try:
        copy_kv_cache_blocks_inplace(
            caches,
            num_blocks,
            [KVCacheBlockCopy(src_block_id=0, dst_block_id=2)],
        )
    finally:
        set_kv_cache_layout(None)

    for layer_idx, cache in enumerate(caches):
        torch.testing.assert_close(cache[2], expected[layer_idx][0])
        torch.testing.assert_close(cache[1], expected[layer_idx][1])


@pytest.mark.parametrize("layout", [KVCacheLayout.LBHNC, KVCacheLayout.BLHNC])
def test_copy_kv_cache_blocks_with_virtual_block_splitting(layout: KVCacheLayout):
    num_blocks = 4
    num_layers = 2
    physical_per_logical = 2
    spec = FullAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
    )
    raw = torch.zeros(
        num_blocks * num_layers * spec.page_size_bytes,
        dtype=torch.int8,
    )
    caches = reshape_kv_cache(
        raw,
        spec,
        num_blocks * physical_per_logical,
        num_layers,
        layout,
        block_size=spec.block_size // physical_per_logical,
    )

    for layer_idx, cache in enumerate(caches):
        for block_idx in range(cache.shape[0]):
            cache[block_idx].fill_(100 * layer_idx + block_idx)
    expected = [[cache[i].clone() for i in range(cache.shape[0])] for cache in caches]

    set_kv_cache_layout(layout.name)
    try:
        copy_kv_cache_blocks_inplace(
            caches,
            num_blocks,
            [KVCacheBlockCopy(src_block_id=0, dst_block_id=2)],
        )
    finally:
        set_kv_cache_layout(None)

    dst_start = 2 * physical_per_logical
    for layer_idx, cache in enumerate(caches):
        for physical_idx in range(physical_per_logical):
            torch.testing.assert_close(
                cache[dst_start + physical_idx], expected[layer_idx][physical_idx]
            )
