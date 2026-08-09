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


def test_reshape_separate_kv_head_groups_matches_aiter_block_interior():
    """K/V head groups under LBHNC must be token-major and block-contiguous.

    The AITER fused QK-norm+RoPE+cache kernel indexes each side as
    ``block_id * stride(0) + token * (H*hs) + head * hs + elem`` and validates
    the interior with a contiguous-from-dim-1 check (aiter rope_common.h). This
    guards the geometry that contract depends on, since no CI runner exercises
    the ROCm path.
    """
    num_blocks, num_layers, block_size, num_kv_heads, head_size = 3, 2, 4, 2, 8
    common = dict(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.float32,
    )
    packed = FullAttentionSpec(**common)
    planes = FullAttentionSpec(**common, separate_kv_head_groups=True)

    # Pure repermutation: same bytes per page, K/V regrouped out of the content.
    assert planes.page_size_bytes == packed.page_size_bytes
    assert planes.num_heads == 2
    assert planes.state_content_size_bytes == num_kv_heads * head_size * 4

    raw = torch.zeros(
        num_blocks * num_layers * planes.page_size_bytes, dtype=torch.int8
    )
    views = reshape_kv_cache(raw, planes, num_blocks, num_layers, KVCacheLayout.LBHNC)

    def is_contiguous_from_dim1(t: torch.Tensor) -> bool:
        """Mirror of aiter's csrc/kernels/rope/rope_common.h predicate."""
        expected = 1
        for dim in range(t.dim() - 1, 0, -1):
            if t.size(dim) != 1 and t.stride(dim) != expected:
                return False
            expected *= t.size(dim)
        return True

    for kv_cache in views:
        assert kv_cache.shape == (
            num_blocks,
            2,
            block_size,
            num_kv_heads * head_size,
        )
        key_cache, value_cache = kv_cache.unbind(1)
        shape = (num_blocks, block_size, num_kv_heads, head_size)
        key_cache, value_cache = key_cache.view(shape), value_cache.view(shape)
        for side in (key_cache, value_cache):
            assert is_contiguous_from_dim1(side)
            assert side.stride(1) == num_kv_heads * head_size  # slot_size
            assert side.stride(2) == head_size
            # Block stride spans both planes and is read at runtime by the kernel.
            assert side.stride(0) == 2 * block_size * num_kv_heads * head_size

    # The plane views together address every byte of the allocation exactly once.
    for kv_cache in views:
        key_cache, value_cache = kv_cache.unbind(1)
        key_cache.fill_(1.0)
        value_cache.fill_(2.0)
    assert (raw.view(torch.float32) != 0).all()


def test_reshape_quantized_kv_cache_content_includes_inline_scales():
    num_blocks = 3
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.int8,
        kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
        page_size_padded=384,
    )
    # Per-token-head scales live inline in the content dim as
    # [K | K_scale | V | V_scale] per (head, slot), matching
    # TritonAttentionImpl._ensure_scale_caches. The view must address
    # every budgeted byte, including the scales; only alignment padding
    # past unpadded_page_size_bytes stays unaddressed.
    assert spec.real_page_size_bytes == 128
    scale_bytes = 2 * spec.block_size * spec.num_kv_heads * 4
    assert spec.unpadded_page_size_bytes == 128 + scale_bytes
    assert spec.page_size_bytes == 384

    raw = torch.zeros(spec.page_size_bytes * num_blocks, dtype=torch.int8)
    (kv_cache,) = reshape_kv_cache(raw, spec, num_blocks, 1, KVCacheLayout.LBHNC)

    scale_pad = 4  # sizeof(float32) / sizeof(int8)
    assert kv_cache.shape == (num_blocks, 1, 16, 2 * (spec.head_size + scale_pad))
    assert kv_cache.stride(0) == spec.page_size_padded
    assert kv_cache[1].storage_offset() == spec.page_size_padded
    assert (
        kv_cache[0].numel() * kv_cache.element_size() == spec.unpadded_page_size_bytes
    )


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


def test_copy_kv_cache_blocks_separate_head_groups():
    # LHBNC is the layout that puts each head group in its own plane, so a
    # block's bytes are scattered across L*H regions.
    layout = KVCacheLayout.LHBNC
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
