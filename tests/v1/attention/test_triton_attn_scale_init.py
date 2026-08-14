# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inline per-token-head KV scale regions must be initialized exactly once
per storage.

KV-sharing layers alias their target layer's cache tensor and build their
scale views lazily on first forward, which can happen after the target layer
has already written real per-token scales in the same pass; re-initializing
on view creation wipes those scales and corrupts the first batch
(#50702, #50749).
"""

import pytest
import torch

from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl


def _make_impl() -> TritonAttentionImpl:
    impl = object.__new__(TritonAttentionImpl)
    impl._k_scale_cache = None
    impl._v_scale_cache = None
    return impl


@pytest.fixture(autouse=True)
def _reset_registry():
    registry = getattr(TritonAttentionImpl, "_scale_initialized_storages", None)
    saved = dict(registry) if registry is not None else None
    if registry is not None:
        registry.clear()
    yield
    if registry is not None:
        registry.clear()
        registry.update(saved)


def _kv_cache(num_blocks=4, nkv=1, block_size=16, head_size=256):
    padded = head_size + 4
    return torch.zeros(num_blocks, nkv, block_size, 2 * padded, dtype=torch.int8)


def test_first_impl_initializes_scales_to_one():
    impl = _make_impl()
    impl._ensure_scale_caches(_kv_cache())
    assert torch.all(impl._k_scale_cache == 1.0)
    assert torch.all(impl._v_scale_cache == 1.0)


def test_aliasing_impl_preserves_written_scales():
    kv = _kv_cache()
    writer = _make_impl()
    writer._ensure_scale_caches(kv)
    writer._k_scale_cache[1, 2, 0] = 0.123
    writer._v_scale_cache[3, 5, 0] = 0.456

    alias = _make_impl()
    alias._ensure_scale_caches(kv)
    assert alias._k_scale_cache[1, 2, 0].item() == pytest.approx(0.123)
    assert alias._v_scale_cache[3, 5, 0].item() == pytest.approx(0.456)


def test_new_storage_is_initialized():
    first = _make_impl()
    kv1 = _kv_cache()
    first._ensure_scale_caches(kv1)

    second = _make_impl()
    kv2 = _kv_cache()
    second._ensure_scale_caches(kv2)
    assert torch.all(second._k_scale_cache == 1.0)
    assert torch.all(second._v_scale_cache == 1.0)


def test_dead_registry_entry_is_reinitialized():
    """A registry entry whose storage died (freed-and-reallocated cache at a
    reused address) must not suppress initialization of the new cache."""
    import weakref

    doomed = torch.zeros(8, dtype=torch.int8)
    dead_ref = weakref.ref(doomed.untyped_storage())
    del doomed

    kv2 = _kv_cache()
    kv2[0, 0, 0, 256:260] = -5  # dirty the first K-scale byte region
    TritonAttentionImpl._scale_initialized_storages[kv2.data_ptr()] = dead_ref
    assert dead_ref() is None

    impl = _make_impl()
    impl._ensure_scale_caches(kv2)
    assert torch.all(impl._k_scale_cache == 1.0)
    assert torch.all(impl._v_scale_cache == 1.0)


def test_packed_layers_with_distinct_offsets_both_initialize():
    """Two layers packed into one storage at different offsets must each
    initialize their own scale region (registry keys on view identity)."""
    padded = 260
    backing = torch.zeros(2 * 4 * 1 * 16 * 2 * padded, dtype=torch.int8)
    kv_a = backing[: backing.numel() // 2].view(4, 1, 16, 2 * padded)
    kv_b = backing[backing.numel() // 2 :].view(4, 1, 16, 2 * padded)

    a = _make_impl()
    a._ensure_scale_caches(kv_a)
    b = _make_impl()
    b._ensure_scale_caches(kv_b)
    assert torch.all(a._k_scale_cache == 1.0)
    assert torch.all(b._k_scale_cache == 1.0)
    assert torch.all(b._v_scale_cache == 1.0)
