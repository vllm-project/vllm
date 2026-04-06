# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


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
