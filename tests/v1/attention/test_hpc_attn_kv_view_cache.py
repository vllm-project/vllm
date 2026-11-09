# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only regression tests for the HpcAttentionImpl K/V view cache."""

import importlib.machinery
import sys
import types

import pytest
import torch

from vllm.v1.attention.backends.hpc_attn import HpcAttentionImpl
from vllm.v1.worker.utils import clear_layer_kv_caches

pytestmark = pytest.mark.cpu_test

HEAD_SIZE = 128
NUM_HEADS = 16
NUM_KV_HEADS = 4
NUM_BLOCKS = 2
BLOCK_SIZE = 4


@pytest.fixture(autouse=True)
def _fake_hpc(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("hpc")
    module.__spec__ = importlib.machinery.ModuleSpec("hpc", loader=None)
    module.QuantType = types.SimpleNamespace(
        QPERTOKEN_PERHEAD_KPERTENSOR_VPERTENSOR=object()
    )
    monkeypatch.setitem(sys.modules, "hpc", module)


def _make_impl(kv_cache_dtype: str = "auto") -> HpcAttentionImpl:
    return HpcAttentionImpl(
        num_heads=NUM_HEADS,
        head_size=HEAD_SIZE,
        scale=1.0,
        num_kv_heads=NUM_KV_HEADS,
        kv_cache_dtype=kv_cache_dtype,
    )


def _make_kv_cache(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    logical = (NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, 2 * HEAD_SIZE)
    if dtype == torch.uint8:
        base = torch.randint(0, 256, logical, dtype=dtype)
    else:
        base = torch.randn(logical, dtype=dtype)
    return base.transpose(1, 2)


def test_same_tensor_reuses_views() -> None:
    impl = _make_impl()
    kv_cache = _make_kv_cache()
    key1, value1 = impl._get_kv_views(kv_cache)
    key2, value2 = impl._get_kv_views(kv_cache)
    assert key1 is key2
    assert value1 is value2


def test_views_match_split() -> None:
    impl = _make_impl()
    kv_cache = _make_kv_cache()
    key, value = impl._get_kv_views(kv_cache)
    key_ref, value_ref = kv_cache.transpose(1, 2).split(HEAD_SIZE, dim=-1)
    assert key.shape == key_ref.shape
    assert value.shape == value_ref.shape
    assert torch.equal(key, key_ref)
    assert torch.equal(value, value_ref)


def test_new_tensor_rebuilds_views() -> None:
    impl = _make_impl()
    first = _make_kv_cache()
    second = _make_kv_cache()
    key_first, _ = impl._get_kv_views(first)
    key_second, _ = impl._get_kv_views(second)
    assert key_second is not key_first
    assert impl._kv_view_cache is not None
    assert impl._kv_view_cache[0] is second


def test_fp8_views_are_retyped_and_cached() -> None:
    impl = _make_impl(kv_cache_dtype="fp8_e4m3")
    kv_cache = _make_kv_cache(dtype=torch.uint8)
    key1, value1 = impl._get_kv_views(kv_cache)
    key2, value2 = impl._get_kv_views(kv_cache)
    assert key1.dtype == torch.float8_e4m3fn
    assert value1.dtype == torch.float8_e4m3fn
    assert key1 is key2
    assert value1 is value2


def test_clear_layer_kv_caches_resets_view_cache() -> None:
    impl = _make_impl()
    kv_cache = _make_kv_cache()
    impl._get_kv_views(kv_cache)
    assert impl._kv_view_cache is not None
    layer = types.SimpleNamespace(kv_cache=kv_cache, impl=impl)
    clear_layer_kv_caches([layer])
    assert impl._kv_view_cache is None


def test_cached_view_shares_storage() -> None:
    impl = _make_impl()
    kv_cache = _make_kv_cache()
    key, value = impl._get_kv_views(kv_cache)
    value_before = value.clone()
    key.zero_()
    key_ref, _ = kv_cache.transpose(1, 2).split(HEAD_SIZE, dim=-1)
    assert torch.all(key_ref == 0)
    assert torch.equal(value, value_before)
