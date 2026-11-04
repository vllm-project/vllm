# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only regression tests for the FlexAttention K/V view cache (#64572).

Run `pytest tests/v1/attention/test_flex_kv_view_cache.py`.
"""

import pytest
import torch

from vllm.v1.attention.backends.flex_attention import FlexAttentionImpl

pytestmark = pytest.mark.cpu_test

HEAD_SIZE = 128
NUM_KV_HEADS = 4
NUM_BLOCKS = 10
BLOCK_SIZE = 16


def _make_impl() -> FlexAttentionImpl:
    return FlexAttentionImpl(
        num_heads=8,
        head_size=HEAD_SIZE,
        scale=1.0,
        num_kv_heads=NUM_KV_HEADS,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )


def _make_kv_cache() -> torch.Tensor:
    # Flex uses the LBNHC layout: the per-layer cache is stored so that
    # transpose(1, 2) yields a contiguous [num_blocks, block_size, heads,
    # 2 * head_size], which is what makes the forward-path flatten a zero-copy
    # view. Build it the same way: a contiguous base seen transposed, exactly
    # the tensor the model runner hands the impl.
    base = torch.randn(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, 2 * HEAD_SIZE)
    return base.transpose(1, 2)


def test_views_are_cached_per_tensor_identity():
    impl = _make_impl()
    kv = _make_kv_cache()
    k1, v1 = impl._get_kv_views(kv, flatten=False)
    k2, v2 = impl._get_kv_views(kv, flatten=False)
    # Same tensor in -> identical view objects back (no re-derivation).
    assert k1 is k2 and v1 is v2
    kf1, vf1 = impl._get_kv_views(kv, flatten=True)
    kf2, vf2 = impl._get_kv_views(kv, flatten=True)
    assert kf1 is kf2 and vf1 is vf2


def test_views_match_direct_derivation():
    impl = _make_impl()
    kv = _make_kv_cache()
    k_split, v_split = impl._get_kv_views(kv, flatten=False)
    dk, dv = kv.transpose(1, 2).split(HEAD_SIZE, dim=-1)
    assert torch.equal(k_split, dk) and torch.equal(v_split, dv)
    k_flat, v_flat = impl._get_kv_views(kv, flatten=True)
    assert k_flat.shape == (NUM_BLOCKS * BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    assert torch.equal(k_flat, dk.reshape(-1, NUM_KV_HEADS, HEAD_SIZE))


def test_cache_rebuilds_when_tensor_replaced():
    impl = _make_impl()
    kv_profiling = _make_kv_cache()
    k_old, _ = impl._get_kv_views(kv_profiling, flatten=False)
    # Real cache replaces the profiling one -> identity differs -> rebuild.
    kv_real = _make_kv_cache()
    k_new, _ = impl._get_kv_views(kv_real, flatten=False)
    assert k_new is not k_old
    assert impl._kv_view_cache is not None
    assert impl._kv_view_cache[0] is kv_real


def test_teardown_clears_cache():
    impl = _make_impl()
    kv = _make_kv_cache()
    impl._get_kv_views(kv, flatten=False)
    assert impl._kv_view_cache is not None
    # Simulate _cleanup_profiling_kv_cache / clear_layer_kv_caches.
    impl._kv_view_cache = None
    k, _ = impl._get_kv_views(kv, flatten=False)
    assert impl._kv_view_cache is not None


def test_cached_view_aliases_storage():
    impl = _make_impl()
    kv = _make_kv_cache()
    k_split, _ = impl._get_kv_views(kv, flatten=False)
    # An in-place write through the cached view must land in kv (shared storage),
    # so the attention kernels still read/write the real cache.
    k_split.zero_()
    dk, _ = kv.transpose(1, 2).split(HEAD_SIZE, dim=-1)
    assert torch.all(dk == 0)
