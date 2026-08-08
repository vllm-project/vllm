# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch

from vllm.v1.attention.backends.cpu_attn import _split_cpu_kv_cache
from vllm.v1.attention.backends.utils import set_kv_cache_layout
from vllm.v1.kv_cache_interface import KVCacheLayout


def _make_cache_with_layout(layout: KVCacheLayout) -> torch.Tensor:
    logical_shape = (2, 3, 2, 4, 10)
    physical_shape = tuple(logical_shape[i] for i in layout.stride_order)
    physical = torch.arange(math.prod(logical_shape)).view(physical_shape)
    inverse_order = tuple(layout.stride_order.index(i) for i in range(5))
    return physical.permute(*inverse_order)[0]


@pytest.mark.parametrize(
    "layout", [KVCacheLayout.LBHNC, KVCacheLayout.BLHNC, KVCacheLayout.BHLNC]
)
def test_split_cpu_kv_cache_supports_hnd_layouts(layout: KVCacheLayout):
    set_kv_cache_layout(layout.name)
    try:
        kv_cache = _make_cache_with_layout(layout)
        key_cache, value_cache = _split_cpu_kv_cache(kv_cache)
    finally:
        set_kv_cache_layout(None)

    assert key_cache.shape == value_cache.shape == (3, 2, 4, 5)
    assert key_cache.stride() == value_cache.stride()
    assert key_cache.stride(-2) == 5
    assert value_cache.storage_offset() - key_cache.storage_offset() == 20


def test_split_cpu_kv_cache_rejects_nhd_layout():
    set_kv_cache_layout(KVCacheLayout.LBNHC.name)
    try:
        kv_cache = _make_cache_with_layout(KVCacheLayout.LBNHC)
        with pytest.raises(ValueError, match="does not support KV cache layout LBNHC"):
            _split_cpu_kv_cache(kv_cache)
    finally:
        set_kv_cache_layout(None)


def test_split_cpu_kv_cache_rejects_incompatible_strides():
    set_kv_cache_layout(KVCacheLayout.LBHNC.name)
    try:
        kv_cache = torch.empty(3, 4, 2, 10).transpose(1, 2)
        with pytest.raises(ValueError, match="contiguous token and content"):
            _split_cpu_kv_cache(kv_cache)
    finally:
        set_kv_cache_layout(None)
