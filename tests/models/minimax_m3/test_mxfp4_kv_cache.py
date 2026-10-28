# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 MXFP4 K/V and index cache (ROCm gfx950): page geometry, KV cache
specs and dtype dispatch. No GPU needed."""

import pytest
import torch

from vllm.config.cache import CacheConfig
from vllm.models.minimax_m3.amd.ops.mxfp4_pages import (
    PAGE_BYTES,
    PAGE_TOKENS,
    mxfp4_page_views,
)
from vllm.models.minimax_m3.amd.sparse_attention_mxfp4 import (
    MiniMaxM3SparseRocmMxfp4Backend,
)
from vllm.models.minimax_m3.common.sparse_attention import (
    select_main_backend_and_impl_cls,
)
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVQuantMode,
    MLAAttentionSpec,
    SparseCacheRole,
    get_kv_quant_mode,
)


def test_page_is_data_then_scale():
    assert PAGE_BYTES == PAGE_TOKENS * 68 == 8704
    pages = torch.arange(3 * PAGE_BYTES, dtype=torch.int64).remainder(251)
    pages = pages.to(torch.uint8).view(3, PAGE_BYTES)
    data, scale = mxfp4_page_views(pages)
    assert data.shape == (3, 128, 64) and scale.shape == (3, 128, 4)
    # Token t's data is bytes [64 t, 64 t + 64), its scales bytes
    # [8192 + 4 t, 8192 + 4 t + 4) of the page.
    assert torch.equal(data[2, 5], pages[2, 5 * 64 : 6 * 64])
    assert torch.equal(scale[2, 5], pages[2, 8192 + 20 : 8192 + 24])


def test_page_views_keep_head_and_page_strides():
    """A K/V pool [pages, 2 heads, 8704] is viewed per side without a copy."""
    pool = torch.zeros(4, 2, PAGE_BYTES, dtype=torch.uint8)
    data, scale = mxfp4_page_views(pool[:, 1:])
    assert data.shape == (4, 1, 128, 64)
    assert data.stride() == (2 * PAGE_BYTES, PAGE_BYTES, 64, 1)
    assert data.data_ptr() == pool[:, 1:].data_ptr()
    assert scale.data_ptr() == pool[:, 1:].data_ptr() + 128 * 64
    with pytest.raises(ValueError):
        mxfp4_page_views(pool[:, :, :-1])


@pytest.mark.parametrize("num_kv_heads", [1, 4])
def test_main_spec_page_holds_k_and_v_pages_per_head(num_kv_heads):
    spec = FullAttentionSpec(
        block_size=128,
        num_kv_heads=num_kv_heads,
        head_size=128,
        head_size_v=128,
        dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.MXFP4,
    )
    spec = MiniMaxM3SparseRocmMxfp4Backend.customize_spec(spec)
    assert spec.num_heads == 2 * num_kv_heads
    assert spec.page_size_bytes == 2 * num_kv_heads * PAGE_BYTES
    # Idempotent: a spec that already carries its packing is left alone.
    assert MiniMaxM3SparseRocmMxfp4Backend.customize_spec(spec) == spec


def test_main_spec_of_other_modes_is_untouched():
    spec = FullAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float8_e4m3fn,
        kv_quant_mode=KVQuantMode.FP8_PER_TENSOR,
    )
    assert MiniMaxM3SparseRocmMxfp4Backend.customize_spec(spec) is spec


def test_index_spec_page_is_one_head_of_68_byte_tokens():
    spec = MLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=128 // 2 + 128 // 32,
        dtype=torch.uint8,
        cache_role=SparseCacheRole.INDEXER,
    )
    assert spec.page_size_bytes == PAGE_BYTES


def test_kv_cache_dtype_name():
    assert CacheConfig(cache_dtype="mxfp4").cache_dtype == "mxfp4"
    assert STR_DTYPE_TO_TORCH_DTYPE["mxfp4"] == torch.uint8
    mode = get_kv_quant_mode("mxfp4")
    assert mode == KVQuantMode.MXFP4 and mode.is_mxfp4
    # A duplicate enum value would make MXFP4 an alias of another mode.
    assert KVQuantMode(mode.value).name == "MXFP4"
    assert not mode.is_nvfp4 and not get_kv_quant_mode("nvfp4").is_mxfp4
    assert get_kv_quant_mode("auto") == KVQuantMode.NONE


def test_mxfp4_kv_selects_the_rocm_attend():
    from vllm.models.minimax_m3.amd.sparse_attention_mxfp4 import (
        MiniMaxM3SparseRocmMxfp4Impl,
    )
    from vllm.platforms import current_platform

    if current_platform.is_rocm():
        assert select_main_backend_and_impl_cls(
            topk_blocks=16, kv_cache_dtype="mxfp4", num_kv_heads=1
        ) == (MiniMaxM3SparseRocmMxfp4Backend, MiniMaxM3SparseRocmMxfp4Impl)
    else:
        with pytest.raises(ValueError, match="only on ROCm gfx950"):
            select_main_backend_and_impl_cls(
                topk_blocks=16, kv_cache_dtype="mxfp4", num_kv_heads=1
            )
