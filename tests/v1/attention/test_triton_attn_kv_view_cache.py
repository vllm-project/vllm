# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl
from vllm.v1.worker.utils import clear_layer_kv_caches

HEAD_SIZE = 128
BLOCK_SIZE = 16
NUM_BLOCKS = 8


def _make_impl(
    num_kv_heads: int = 4, kv_cache_dtype: str = "auto"
) -> TritonAttentionImpl:
    impl = TritonAttentionImpl.__new__(TritonAttentionImpl)
    impl.head_size = HEAD_SIZE
    impl.num_kv_heads = num_kv_heads
    impl.kv_cache_dtype = kv_cache_dtype
    impl.fp8_dtype = torch.float8_e4m3fn
    impl._set_kv_ptr_range = False
    return impl


def _make_kv_cache(
    num_kv_heads: int = 4, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    return torch.zeros(
        NUM_BLOCKS,
        num_kv_heads,
        BLOCK_SIZE,
        2 * HEAD_SIZE,
        dtype=dtype,
    )


@pytest.mark.cpu_test
def test_views_match_reference() -> None:
    impl = _make_impl(num_kv_heads=4)
    kv_cache = _make_kv_cache(num_kv_heads=4)
    k_ref, v_ref = kv_cache.transpose(1, 2).split(HEAD_SIZE, dim=-1)

    k, v = impl._get_kv_views(kv_cache)

    assert k.shape == k_ref.shape and v.shape == v_ref.shape
    assert k.stride() == k_ref.stride() and v.stride() == v_ref.stride()
    assert torch.equal(k, k_ref) and torch.equal(v, v_ref)


@pytest.mark.cpu_test
def test_same_tensor_reuses_cached_views() -> None:
    impl = _make_impl(num_kv_heads=4)
    kv_cache = _make_kv_cache(num_kv_heads=4)

    k1, v1 = impl._get_kv_views(kv_cache)
    k2, v2 = impl._get_kv_views(kv_cache)
    assert k2 is k1 and v2 is v1


@pytest.mark.cpu_test
def test_new_tensor_rebuilds_views() -> None:
    impl = _make_impl(num_kv_heads=4)
    first = _make_kv_cache(num_kv_heads=4)
    second = _make_kv_cache(num_kv_heads=4)

    k_first, _ = impl._get_kv_views(first)
    k_second, _ = impl._get_kv_views(second)

    assert k_second is not k_first
    assert k_second.data_ptr() == second.data_ptr()


@pytest.mark.cpu_test
def test_fp8_views_cached_with_fp8_dtype() -> None:
    impl = _make_impl(num_kv_heads=4, kv_cache_dtype="fp8")
    kv_cache = _make_kv_cache(num_kv_heads=4, dtype=torch.uint8)

    k1, v1 = impl._get_kv_views(kv_cache)
    k2, v2 = impl._get_kv_views(kv_cache)

    assert k1.dtype == torch.float8_e4m3fn and v1.dtype == torch.float8_e4m3fn
    assert k2 is k1 and v2 is v1


@pytest.mark.cpu_test
def test_clear_layer_kv_caches_drops_view_cache() -> None:
    impl = _make_impl(num_kv_heads=4)
    kv_cache = _make_kv_cache(num_kv_heads=4)
    impl._get_kv_views(kv_cache)
    assert impl._kv_view_cache is not None

    layer = SimpleNamespace(kv_cache=kv_cache, impl=impl)
    clear_layer_kv_caches([layer])

    assert impl._kv_view_cache is None
    assert layer.kv_cache.numel() == 0
