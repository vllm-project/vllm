# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TritonAttentionImpl K/V view caching (same class of issue as #59534; #64572).

``TritonAttentionImpl`` re-derived the K/V cache views
(``transpose(1, 2).split(head_size)``) on every attention call, even though
the views depend only on the ``kv_cache`` tensor's storage/shape/strides,
which are fixed once the cache is allocated. The views are now cached on the
impl (``_get_kv_views``) and reused while the same ``kv_cache`` tensor is
passed in.

This optimization must not change the views it returns, so the tests compare
the cached views against the exact derivation they replace. They also assert
the cache is actually reused (same tensor -> identical view objects) and is
rebuilt on identity change (the profiling path binds a minimal KV cache,
frees it, then allocates the real one).
"""

import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl

HEAD_SIZE = 128
BLOCK_SIZE = 16
NUM_BLOCKS = 8


def _make_impl(monkeypatch, num_kv_heads: int) -> TritonAttentionImpl:
    # Keep current_platform off the CUDA/ROCm capability-check and
    # ptr-range-setting paths so this test runs without a GPU.
    monkeypatch.setattr(current_platform, "is_cuda", lambda: False)
    monkeypatch.setattr(current_platform, "is_rocm", lambda: False)
    return TritonAttentionImpl(
        num_heads=num_kv_heads,
        head_size=HEAD_SIZE,
        scale=1.0 / (HEAD_SIZE**0.5),
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )


def _make_kv_cache(num_kv_heads: int) -> torch.Tensor:
    # Logical (B, H, N, 2*hs) order; forward does transpose(1, 2).split(hs).
    return torch.randn(
        NUM_BLOCKS,
        num_kv_heads,
        BLOCK_SIZE,
        2 * HEAD_SIZE,
        dtype=torch.bfloat16,
    )


def _reference_views(kv_cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # The derivation _get_kv_views replaces.
    return kv_cache.transpose(1, 2).split(HEAD_SIZE, dim=-1)


class TestTritonAttentionKVViewCache:
    def test_views_match_reference(self, monkeypatch):
        impl = _make_impl(monkeypatch, num_kv_heads=4)
        kv_cache = _make_kv_cache(num_kv_heads=4)
        k_ref, v_ref = _reference_views(kv_cache)

        k, v = impl._get_kv_views(kv_cache)

        assert k.shape == k_ref.shape and v.shape == v_ref.shape
        assert k.stride() == k_ref.stride() and v.stride() == v_ref.stride()
        assert torch.equal(k, k_ref) and torch.equal(v, v_ref)

    def test_same_tensor_reuses_cached_views(self, monkeypatch):
        impl = _make_impl(monkeypatch, num_kv_heads=4)
        kv_cache = _make_kv_cache(num_kv_heads=4)

        k1, v1 = impl._get_kv_views(kv_cache)
        k2, v2 = impl._get_kv_views(kv_cache)

        assert k2 is k1 and v2 is v1

    def test_new_tensor_rebuilds_views(self, monkeypatch):
        impl = _make_impl(monkeypatch, num_kv_heads=4)
        first = _make_kv_cache(num_kv_heads=4)
        second = _make_kv_cache(num_kv_heads=4)

        k_first, _ = impl._get_kv_views(first)
        k_second, _ = impl._get_kv_views(second)

        assert k_second is not k_first
        k_ref, _ = _reference_views(second)
        assert torch.equal(k_second, k_ref)

    def test_cleared_cache_rebuilds_views(self, monkeypatch):
        """Mirrors the profiling cleanup path in gpu_model_runner.py, which
        resets ``_kv_view_cache`` to free the profiling KV cache tensor."""
        impl = _make_impl(monkeypatch, num_kv_heads=4)
        kv_cache = _make_kv_cache(num_kv_heads=4)

        k1, _ = impl._get_kv_views(kv_cache)
        impl._kv_view_cache = None
        k2, _ = impl._get_kv_views(kv_cache)

        assert k2 is not k1
        assert torch.equal(k2, k1)
