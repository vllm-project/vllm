# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashAttentionImpl K/V view caching (issue #59534).

Since #44455 ``FlashAttentionImpl`` re-derived the K/V cache views
(``transpose(1, 2).split(head_size)`` plus stride canonicalization) on every
attention call, even though the views depend only on the ``kv_cache`` tensor's
storage/shape/strides, which are fixed once the cache is allocated. The views
are now cached on the impl and reused while the same ``kv_cache`` tensor is
passed in.

This optimization must not change the views it returns, so the tests compare
the cached views against the exact derivation they replace. They also assert
the cache is actually reused (same tensor -> identical view objects) and is
rebuilt on identity change (the profiling path binds a minimal KV cache, frees
it, then allocates the real one).

The impl constructor probes the platform for the FlashAttention version, so the
tests require CUDA with FlashAttention available.
"""

import pytest
import torch

from vllm.config import set_current_vllm_config
from vllm.platforms import current_platform


def _fa_available() -> bool:
    if not current_platform.is_cuda():
        return False
    from vllm.v1.attention.backends.fa_utils import is_fa_version_supported

    return is_fa_version_supported(3) or is_fa_version_supported(2)


requires_fa = pytest.mark.skipif(
    not _fa_available(), reason="FlashAttentionImpl requires CUDA with FlashAttention"
)

if _fa_available():
    from tests.v1.attention.utils import create_vllm_config
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

HEAD_SIZE = 128
BLOCK_SIZE = 16
NUM_BLOCKS = 8


def _make_impl(num_kv_heads: int) -> "FlashAttentionImpl":
    vllm_config = create_vllm_config(block_size=BLOCK_SIZE)
    with set_current_vllm_config(vllm_config):
        return FlashAttentionImpl(
            num_heads=num_kv_heads,
            head_size=HEAD_SIZE,
            scale=1.0 / (HEAD_SIZE**0.5),
            num_kv_heads=num_kv_heads,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
        )


def _make_kv_cache(num_kv_heads: int) -> torch.Tensor:
    # FA packed layout: (B, H, N, 2*D); forward does transpose(1, 2).split(D).
    return torch.randn(
        NUM_BLOCKS,
        num_kv_heads,
        BLOCK_SIZE,
        2 * HEAD_SIZE,
        dtype=torch.bfloat16,
        device="cuda",
    )


def _reference_views(kv_cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # The derivation _get_kv_views replaces.
    return kv_cache.transpose(1, 2).split(HEAD_SIZE, dim=-1)


@requires_fa
class TestFlashAttentionKVViewCache:
    def test_views_match_reference(self):
        """Cached views are identical to the derivation they replace."""
        impl = _make_impl(num_kv_heads=4)
        kv_cache = _make_kv_cache(num_kv_heads=4)
        k_ref, v_ref = _reference_views(kv_cache)

        k, v = impl._get_kv_views(kv_cache, canonicalize=False)

        # num_kv_heads > 1 has canonical strides, so raw == canonicalized here.
        assert k.shape == k_ref.shape and v.shape == v_ref.shape
        assert k.stride() == k_ref.stride() and v.stride() == v_ref.stride()
        assert torch.equal(k, k_ref) and torch.equal(v, v_ref)

    def test_same_tensor_reuses_cached_views(self):
        """Passing the same kv_cache returns the identical view objects."""
        impl = _make_impl(num_kv_heads=4)
        kv_cache = _make_kv_cache(num_kv_heads=4)

        k1, v1 = impl._get_kv_views(kv_cache, canonicalize=True)
        k2, v2 = impl._get_kv_views(kv_cache, canonicalize=True)
        assert k2 is k1 and v2 is v1

        # The raw (uncanonicalized) variant is served from the same cache entry.
        k_raw1, v_raw1 = impl._get_kv_views(kv_cache, canonicalize=False)
        k_raw2, v_raw2 = impl._get_kv_views(kv_cache, canonicalize=False)
        assert k_raw2 is k_raw1 and v_raw2 is v_raw1

    def test_new_tensor_rebuilds_views(self):
        """A different kv_cache tensor rebuilds the views (profiling path)."""
        impl = _make_impl(num_kv_heads=4)
        first = _make_kv_cache(num_kv_heads=4)
        second = _make_kv_cache(num_kv_heads=4)

        k_first, _ = impl._get_kv_views(first, canonicalize=False)
        k_second, _ = impl._get_kv_views(second, canonicalize=False)

        assert k_second is not k_first
        k_ref, _ = _reference_views(second)
        assert torch.equal(k_second, k_ref)

    def test_clear_layer_kv_caches_drops_view_cache(self):
        """Runner teardown must drop the cached views, or the detached
        kv_cache tensor stays alive through them.

        Both runners release layer KV caches on profiling teardown and
        shutdown: Model Runner V2 (and the cudagraph utils) via
        clear_layer_kv_caches, V1 via the same pattern inlined in
        _cleanup_profiling_kv_cache.
        """
        from types import SimpleNamespace

        from vllm.v1.worker.utils import clear_layer_kv_caches

        impl = _make_impl(num_kv_heads=4)
        kv_cache = _make_kv_cache(num_kv_heads=4)
        impl._get_kv_views(kv_cache, canonicalize=True)
        assert impl._kv_view_cache is not None

        layer = SimpleNamespace(kv_cache=kv_cache, impl=impl)
        clear_layer_kv_caches([layer])

        assert impl._kv_view_cache is None
        assert layer.kv_cache.numel() == 0

    def test_single_kv_head_raw_and_canonicalized_variants(self):
        """With one KV head the two variants share storage and values; the
        canonicalized view carries canonical size-1 strides.

        (The degenerate-stride *repair* itself is covered by the
        canonicalize_singleton_dim_strides unit tests; a freshly allocated
        contiguous cache is already canonical, so here the two variants agree.)
        """
        impl = _make_impl(num_kv_heads=1)
        kv_cache = _make_kv_cache(num_kv_heads=1)

        # After transpose(1, 2) the size-1 KV-head dim is -2; its canonical
        # stride is head_size.
        k_raw, v_raw = impl._get_kv_views(kv_cache, canonicalize=False)
        k_canon, v_canon = impl._get_kv_views(kv_cache, canonicalize=True)

        assert k_canon.stride(-2) == HEAD_SIZE and v_canon.stride(-2) == HEAD_SIZE
        # Same storage, zero-copy.
        assert k_canon.data_ptr() == k_raw.data_ptr()
        assert v_canon.data_ptr() == v_raw.data_ptr()
        # Canonicalization never changes values.
        assert torch.equal(k_canon, k_raw) and torch.equal(v_canon, v_raw)
