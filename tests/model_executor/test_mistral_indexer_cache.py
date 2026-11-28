# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The migrated model's indexer must implement the current cache contract."""

import inspect

import torch

from vllm.config import CacheConfig, VllmConfig, set_current_vllm_config
from vllm.models.mistral.nvidia.indexer import MistralIndexerCache
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import MLAAttentionSpec, SparseCacheRole


def test_mistral_indexer_implements_cache_config_contract():
    assert not inspect.isabstract(MistralIndexerCache)
    config = VllmConfig()
    with set_current_vllm_config(config):
        cache = MistralIndexerCache(
            head_dim=132,
            dtype=torch.uint8,
            prefix="test.indexer.k_cache",
            cache_config=CacheConfig(block_size=64),
        )
        layer = cache.get_kv_cache_config(config)
    assert layer.manager_cls is FullAttentionManager
    assert isinstance(layer.spec, MLAAttentionSpec)
    assert layer.spec.cache_role is SparseCacheRole.INDEXER
    assert layer.spec.block_size == 64
    assert layer.spec.head_size == 132
    assert layer.spec.num_kv_heads == layer.spec.max_tp_shards == 1
    assert layer.spec.dtype is torch.uint8
    # The new config API must retain the reader's existing three-dimensional view.
    backing = torch.empty((2, 1, 64, 132), dtype=torch.uint8)
    cache.bind_kv_cache(backing)
    assert cache.kv_cache.shape == (2, 64, 132)
    assert cache.kv_cache.data_ptr() == backing.data_ptr()
