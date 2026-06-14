# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch


def test_mla_backend_rejects_cross_layer_kv_cache():
    """MLA backends return identity permutation (layers dim first)
    to signal cross-layer KV cache is unsupported."""
    from vllm.model_executor.layers.attention.mla_attention import (
        MLACommonBackend,
    )

    stride_order = MLACommonBackend.get_kv_cache_stride_order(
        include_num_layers_dimension=True
    )
    assert stride_order == (0, 1, 2, 3)
    assert stride_order[0] == 0  # layers dim first => no cross-layer
    assert MLACommonBackend.get_kv_cache_stride_order(
        include_num_layers_dimension=False
    ) == (0, 1, 2)


def test_deepseek_v32_indexer_rejects_cross_layer_kv_cache():
    """DeepseekV32Indexer returns identity permutation (layers dim first)
    to signal cross-layer KV cache is unsupported."""
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerBackend,
    )

    stride_order = DeepseekV32IndexerBackend.get_kv_cache_stride_order(
        include_num_layers_dimension=True
    )
    assert stride_order == (0, 1, 2, 3)
    assert stride_order[0] == 0  # layers dim first => no cross-layer
    assert DeepseekV32IndexerBackend.get_kv_cache_stride_order(
        include_num_layers_dimension=False
    ) == (0, 1, 2)


def test_rocm_attn_accepts_connector_friendly_layouts(monkeypatch):
    """ROCM_ATTN exposes block-first logical cache views for KV connectors."""
    rocm_attn = pytest.importorskip("vllm.v1.attention.backends.rocm_attn")
    from vllm.v1.attention.backends.utils import set_kv_cache_layout

    backend = rocm_attn.RocmAttentionBackend
    assert backend.supports_sink()
    assert backend.supports_kv_connector()
    assert backend.get_required_kv_cache_layout() is None
    assert backend.get_kv_cache_shape(8, 16, 4, 64) == (8, 2, 16, 4, 64)

    try:
        set_kv_cache_layout("NHD")
        assert backend.get_kv_cache_stride_order() == (0, 1, 2, 3, 4)
        assert backend.get_kv_cache_stride_order(include_num_layers_dimension=True) == (
            1,
            0,
            2,
            3,
            4,
            5,
        )

        set_kv_cache_layout("HND")
        assert backend.get_kv_cache_stride_order() == (0, 1, 3, 2, 4)
        assert backend.get_kv_cache_stride_order(include_num_layers_dimension=True) == (
            1,
            4,
            0,
            2,
            3,
            5,
        )
    finally:
        set_kv_cache_layout(None)


def test_rocm_attn_split_connector_layouts_keep_native_inner_blocks():
    """Connector block-first storage still presents native packed ROCm pages."""
    rocm_attn = pytest.importorskip("vllm.v1.attention.backends.rocm_attn")
    has_native_layout = rocm_attn.has_native_kv_cache_layout
    split_kv_cache = rocm_attn.RocmAttentionImpl._split_kv_cache

    def logical_cache_from_order(
        shape: tuple[int, ...], order: tuple[int, ...]
    ) -> torch.Tensor:
        physical_shape = tuple(shape[i] for i in order)
        inv_order = [order.index(i) for i in range(len(order))]
        return torch.empty(physical_shape, dtype=torch.float16).permute(*inv_order)

    num_blocks = 4
    block_size = 16
    num_kv_heads = 3
    head_size = 64
    logical_shape = (num_blocks, 2, block_size, num_kv_heads, head_size)
    layered_shape = (2, *logical_shape)

    cases = [
        logical_cache_from_order(logical_shape, (0, 1, 2, 3, 4)),
        logical_cache_from_order(logical_shape, (0, 1, 3, 2, 4)),
        logical_cache_from_order(layered_shape, (1, 4, 0, 2, 3, 5))[1],
    ]

    x = 16 // torch.empty((), dtype=torch.float16).element_size()
    for kv_cache in cases:
        key_cache, value_cache = split_kv_cache(None, kv_cache)
        assert key_cache.shape == (
            num_blocks,
            num_kv_heads,
            head_size // x,
            block_size,
            x,
        )
        assert value_cache.shape == (
            num_blocks,
            num_kv_heads,
            head_size,
            block_size,
        )
        assert has_native_layout(key_cache, value_cache)
        assert key_cache.stride()[2:] == (block_size * x, x, 1)
        assert value_cache.stride()[2:] == (block_size, 1)
        assert value_cache.storage_offset() == (
            kv_cache.storage_offset() + kv_cache.stride(1)
        )
