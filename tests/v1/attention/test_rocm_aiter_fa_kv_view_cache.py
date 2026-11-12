# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for the ROCM_AITER_FA K/V cache view cache."""

import weakref
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends import rocm_aiter_fa
from vllm.v1.attention.backends.rocm_aiter_fa import AiterFlashAttentionImpl
from vllm.v1.worker.utils import clear_layer_kv_caches

pytestmark = pytest.mark.cpu_test

HEAD_SIZE = 128
NUM_KV_HEADS = 4
NUM_BLOCKS = 8
BLOCK_SIZE = 16


def _make_impl(
    monkeypatch: pytest.MonkeyPatch,
    shuffle: bool = False,
    kv_cache_dtype: str = "auto",
) -> AiterFlashAttentionImpl:
    monkeypatch.setattr(
        rocm_aiter_fa.rocm_aiter_ops, "is_shuffle_kv_cache_enabled", lambda: shuffle
    )
    return AiterFlashAttentionImpl(
        num_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        scale=HEAD_SIZE**-0.5,
        num_kv_heads=NUM_KV_HEADS,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype=kv_cache_dtype,
    )


def _make_kv_cache(
    shuffle: bool = False, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    if shuffle:
        shape = (NUM_BLOCKS, 2, BLOCK_SIZE, NUM_KV_HEADS * HEAD_SIZE)
    else:
        shape = (NUM_BLOCKS, NUM_KV_HEADS, BLOCK_SIZE, 2 * HEAD_SIZE)
    return torch.zeros(shape, dtype=dtype)


@pytest.mark.parametrize("shuffle", [False, True])
def test_views_alias_cache_and_are_reused(monkeypatch, shuffle):
    impl = _make_impl(monkeypatch, shuffle)
    kv_cache = _make_kv_cache(shuffle)
    if shuffle:
        expected = kv_cache.unflatten(-1, (NUM_KV_HEADS, HEAD_SIZE)).unbind(1)
    else:
        expected = kv_cache.transpose(1, 2).split(HEAD_SIZE, dim=-1)

    views = impl._split_kv_cache(kv_cache)
    for view, ref in zip(views, expected, strict=True):
        assert (view.data_ptr(), view.shape, view.stride()) == (
            ref.data_ptr(),
            ref.shape,
            ref.stride(),
        )
    k, v = impl._split_kv_cache(kv_cache)
    assert k is views[0] and v is views[1]


@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize(
    ("kv_cache_dtype", "storage_dtype"),
    [("auto", torch.bfloat16), ("fp8", torch.uint8)],
)
def test_fp8_views_are_cached_next_to_raw_views(
    monkeypatch, shuffle, kv_cache_dtype, storage_dtype
):
    impl = _make_impl(monkeypatch, shuffle, kv_cache_dtype)
    kv_cache = _make_kv_cache(shuffle, storage_dtype)
    fp8 = kv_cache_dtype == "fp8"

    raw = impl._split_kv_cache(kv_cache)
    typed = impl._split_kv_cache(kv_cache, fp8=True)
    for raw_view, typed_view in zip(raw, typed, strict=True):
        assert raw_view.dtype == storage_dtype
        assert typed_view.dtype == (
            current_platform.fp8_dtype() if fp8 else storage_dtype
        )
        assert (typed_view.data_ptr(), typed_view.shape, typed_view.stride()) == (
            raw_view.data_ptr(),
            raw_view.shape,
            raw_view.stride(),
        )
    k, v = impl._split_kv_cache(kv_cache, fp8=True)
    assert k is typed[0] and v is typed[1]


def test_new_tensor_or_storage_rebuilds_views(monkeypatch):
    impl = _make_impl(monkeypatch)
    first, second = _make_kv_cache(), _make_kv_cache()
    k_first, _ = impl._split_kv_cache(first)
    k_second, _ = impl._split_kv_cache(second)
    assert k_second is not k_first
    assert k_second.data_ptr() == second.data_ptr()

    other = _make_kv_cache()
    second.set_(other.untyped_storage(), 0, second.shape, second.stride())
    k_moved, _ = impl._split_kv_cache(second)
    assert k_moved.data_ptr() == other.data_ptr()

    fewer = second[: NUM_BLOCKS // 2]
    k_fewer, _ = impl._split_kv_cache(fewer)
    assert k_fewer.shape[0] == NUM_BLOCKS // 2


def test_clear_layer_kv_caches_drops_view_cache(monkeypatch):
    impl = _make_impl(monkeypatch, kv_cache_dtype="fp8")
    kv_cache = _make_kv_cache(dtype=torch.uint8)
    impl._split_kv_cache(kv_cache, fp8=True)
    layer = SimpleNamespace(kv_cache=kv_cache, impl=impl)
    kv_cache_ref = weakref.ref(kv_cache)
    del kv_cache

    clear_layer_kv_caches([layer])

    assert impl._kv_view_cache is None
    assert kv_cache_ref() is None
