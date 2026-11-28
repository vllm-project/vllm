# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regressions for DiffKV's packed-write and split-read cache views."""

import weakref

import pytest
import torch

from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.attention import Attention
from vllm.utils.torch_utils import canonicalize_singleton_dim_strides
from vllm.v1.attention.backends.flash_attn_diffkv import (
    FlashAttentionDiffKVBackend,
    FlashAttentionDiffKVImpl,
)
from vllm.v1.worker.utils import clear_layer_kv_caches

pytestmark = pytest.mark.cpu_test

HEAD_SIZE = 192
HEAD_SIZE_V = 128


def make_impl(num_kv_heads: int = 1) -> FlashAttentionDiffKVImpl:
    return FlashAttentionDiffKVImpl(
        num_heads=4,
        head_size=HEAD_SIZE,
        scale=HEAD_SIZE**-0.5,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )


def make_cache(num_kv_heads: int = 1, dtype: torch.dtype = torch.bfloat16):
    shape = (2, 4, num_kv_heads, HEAD_SIZE + HEAD_SIZE_V)
    data = torch.arange(2 * 4 * num_kv_heads * (HEAD_SIZE + HEAD_SIZE_V)) % 251
    return data.to(dtype).reshape(shape).transpose(1, 2)


@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.uint8])
def test_reuses_views_with_original_layout(num_kv_heads, dtype):
    impl = make_impl(num_kv_heads)
    kv_cache = make_cache(num_kv_heads, dtype)
    views = impl._get_diffkv_views(kv_cache)
    reused = impl._get_diffkv_views(kv_cache)
    packed = kv_cache.transpose(1, 2)
    key, value = packed.split(HEAD_SIZE, dim=-1)
    expected = (
        packed,
        canonicalize_singleton_dim_strides(key),
        canonicalize_singleton_dim_strides(value),
    )
    for actual, cached, reference in zip(views, reused, expected):
        assert actual is cached
        assert torch.equal(actual, reference)
        assert actual.stride() == reference.stride()
        assert actual.storage_offset() == reference.storage_offset()
        assert (
            actual.untyped_storage().data_ptr() == kv_cache.untyped_storage().data_ptr()
        )
    assert views[1].shape[-1] == HEAD_SIZE
    assert views[2].shape[-1] == HEAD_SIZE_V


def test_cached_views_observe_and_update_live_storage():
    impl = make_impl()
    kv_cache = make_cache()
    packed, key, value = impl._get_diffkv_views(kv_cache)
    old_value = value.clone()
    packed[..., :HEAD_SIZE].fill_(7)
    assert torch.all(impl._get_diffkv_views(kv_cache)[1] == 7)
    assert torch.equal(value, old_value)
    value.fill_(9)
    assert torch.all(kv_cache[..., HEAD_SIZE:] == 9)
    assert torch.all(key == 7)


def test_replacement_rebuilds_views_and_releases_old_cache():
    impl = make_impl()
    profiling_cache = make_cache()
    profiling_ref = weakref.ref(profiling_cache)
    old_views = impl._get_diffkv_views(profiling_cache)
    real_cache = make_cache() + 1
    new_views = impl._get_diffkv_views(real_cache)
    assert all(new is not old for new, old in zip(new_views, old_views))
    assert torch.equal(new_views[0], real_cache.transpose(1, 2))
    del profiling_cache, old_views
    assert profiling_ref() is None


def test_teardown_drops_cached_views_and_rebuilds_on_rebind():
    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    with set_current_vllm_config(config):
        layer = Attention(
            num_heads=4,
            head_size=HEAD_SIZE,
            head_size_v=HEAD_SIZE_V,
            scale=HEAD_SIZE**-0.5,
            num_kv_heads=1,
            prefix="layers.0.attn",
            attn_backend=FlashAttentionDiffKVBackend,
        )
    impl = layer.impl
    assert isinstance(impl, FlashAttentionDiffKVImpl)
    kv_cache = make_cache()
    old_ref = weakref.ref(kv_cache)
    layer.bind_kv_cache(kv_cache)
    impl._get_diffkv_views(layer.kv_cache)
    del kv_cache
    assert old_ref() is not None
    clear_layer_kv_caches([layer])
    assert layer.kv_cache.numel() == 0
    assert impl._diffkv_view_cache is None
    assert old_ref() is None
    real_cache = make_cache() + 1
    layer.bind_kv_cache(real_cache)
    assert torch.equal(
        impl._get_diffkv_views(layer.kv_cache)[0], real_cache.transpose(1, 2)
    )
