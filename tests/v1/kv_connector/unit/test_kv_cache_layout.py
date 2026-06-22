# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.platforms import current_platform


def test_mla_common_backend_rejects_cross_layer_kv_cache():
    """MLACommonBackend defaults to the identity permutation (layers dim
    first) so MLA backends whose decode kernels are not verified to honor
    the cache's block-dim stride stay opted out of cross-layer KV cache."""
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


@pytest.mark.parametrize(
    "backend_path",
    # See: https://github.com/vllm-project/vllm/issues/46411
    [
        "vllm.v1.attention.backends.mla.triton_mla.TritonMLABackend",
    ]
    if current_platform.is_rocm()
    else [
        "vllm.v1.attention.backends.mla.triton_mla.TritonMLABackend",
        "vllm.v1.attention.backends.mla.cutlass_mla.CutlassMLABackend",
        "vllm.v1.attention.backends.mla.flashattn_mla.FlashAttnMLABackend",
        "vllm.v1.attention.backends.mla.flashmla.FlashMLABackend",
        "vllm.v1.attention.backends.mla.flashinfer_mla.FlashInferMLABackend",
    ],
)
def test_verified_mla_backends_support_cross_layer_kv_cache(backend_path):
    """Backends whose decode kernels honor the cache's block-dim stride opt
    in to the cross-layer layout with a non-identity permutation placing
    num_blocks first in physical layout."""
    module_path, name = backend_path.rsplit(".", 1)
    backend = getattr(
        pytest.importorskip(module_path, reason="backend deps unavailable"), name
    )

    stride_order = backend.get_kv_cache_stride_order(include_num_layers_dimension=True)
    assert stride_order == (1, 0, 2, 3)
    assert stride_order[0] != 0  # num_blocks first => cross-layer supported
    assert backend.get_kv_cache_stride_order(include_num_layers_dimension=False) == (
        0,
        1,
        2,
    )


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
