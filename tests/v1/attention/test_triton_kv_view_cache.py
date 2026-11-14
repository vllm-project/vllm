# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression test for vLLM #64572.

TritonAttentionImpl re-derived K/V cache views (transpose(1, 2).split +
optional FP8 retype) on every attention call, even though the views depend
only on the kv_cache tensor's storage/shape/strides, which are fixed once
the cache is allocated. This test verifies the _get_kv_views caching:
views are byte-identical to the derivation they replace, reused on the
same tensor, and rebuilt on tensor identity change.
"""

import pytest
import torch

from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl


def _make_kv_cache(
    num_blocks=4,
    num_kv_heads=2,
    block_size=16,
    head_size=128,
    dtype=torch.bfloat16,
):
    """Create a minimal KV cache tensor in the expected layout."""
    return torch.zeros(
        num_blocks, num_kv_heads, block_size, 2 * head_size,
        dtype=dtype, device="cpu",
    )


def _make_impl(head_size=128, kv_cache_dtype="auto"):
    """Build a TritonAttentionImpl with only the attributes _get_kv_views needs."""
    impl = object.__new__(TritonAttentionImpl)
    impl.head_size = head_size
    impl.fp8_dtype = torch.float8_e4m3fn
    impl._kv_view_cache = None
    impl._kv_quant_mode = None
    impl._is_per_token_head_quant = False
    impl.kv_cache_dtype = kv_cache_dtype
    impl._set_kv_ptr_range = False
    return impl


def test_views_match_reference():
    """Cached views must be byte-identical to the derivation they replace."""
    kv_cache = _make_kv_cache()
    impl = _make_impl()

    key_cache, value_cache = impl._get_kv_views(kv_cache, impl.head_size)

    # Reference derivation
    kv_t = kv_cache.transpose(1, 2)
    ref_key, ref_value = kv_t.split(impl.head_size, dim=-1)

    assert key_cache.shape == ref_key.shape
    assert value_cache.shape == ref_value.shape
    assert key_cache.stride() == ref_key.stride()
    assert value_cache.stride() == ref_value.stride()
    assert key_cache.data_ptr() == ref_key.data_ptr()
    assert value_cache.data_ptr() == ref_value.data_ptr()


def test_same_tensor_reuses_cached_views():
    """Calling _get_kv_views with the same tensor must reuse the cache."""
    kv_cache = _make_kv_cache()
    impl = _make_impl()

    k1, v1 = impl._get_kv_views(kv_cache, impl.head_size)
    k2, v2 = impl._get_kv_views(kv_cache, impl.head_size)

    assert k1 is k2
    assert v1 is v2
    assert impl._kv_view_cache is not None
    assert impl._kv_view_cache[0] is kv_cache


def test_new_tensor_rebuilds_views():
    """Replacing the kv_cache tensor must rebuild the cached views."""
    kv1 = _make_kv_cache()
    kv2 = _make_kv_cache()
    impl = _make_impl()

    k1, v1 = impl._get_kv_views(kv1, impl.head_size)
    k2, v2 = impl._get_kv_views(kv2, impl.head_size)

    assert k1 is not k2
    assert v1 is not v2
    assert impl._kv_view_cache[0] is kv2
    assert k2.data_ptr() != k1.data_ptr()


def test_fp8_retype_cached():
    """FP8 retype views must be cached and match reference."""
    kv_cache = _make_kv_cache(dtype=torch.float8_e4m3fn)
    impl = _make_impl()

    key_cache, value_cache = impl._get_kv_views(
        kv_cache, impl.head_size, fp8_retype=True
    )

    # Reference: split then view as fp8
    kv_t = kv_cache.transpose(1, 2)
    ref_key, ref_value = kv_t.split(impl.head_size, dim=-1)
    ref_key_fp8 = ref_key.view(impl.fp8_dtype)
    ref_value_fp8 = ref_value.view(impl.fp8_dtype)

    assert key_cache.dtype == impl.fp8_dtype
    assert value_cache.dtype == impl.fp8_dtype
    assert key_cache.data_ptr() == ref_key_fp8.data_ptr()
    assert value_cache.data_ptr() == ref_value_fp8.data_ptr()

    # Second call reuses cache
    k2, v2 = impl._get_kv_views(
        kv_cache, impl.head_size, fp8_retype=True
    )
    assert k2 is key_cache
    assert v2 is value_cache


def test_different_fp8_retype_cache_miss():
    """Different fp8_retype flag must cause a cache rebuild."""
    kv_cache = _make_kv_cache(head_size=128)
    impl = _make_impl(head_size=128)

    # First call without fp8 retype
    k1, v1 = impl._get_kv_views(kv_cache, impl.head_size, fp8_retype=False)
    assert k1.dtype == kv_cache.dtype

    # Second call with fp8 retype -> rebuild
    kv_cache_fp8 = _make_kv_cache(head_size=128, dtype=torch.float8_e4m3fn)
    k2, v2 = impl._get_kv_views(kv_cache_fp8, impl.head_size, fp8_retype=True)
    assert k2.dtype == impl.fp8_dtype
    assert k1 is not k2
