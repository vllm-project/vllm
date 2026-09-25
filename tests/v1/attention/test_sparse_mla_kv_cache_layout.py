# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse MLA allocation and warmup must agree on physical addressing."""

from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.config import CacheConfig
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseTRTLLMBackend,
)
from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseBackend
from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend
from vllm.v1.attention.backends.mla.sparse_utils import (
    _CONVERT_REQ_INDEX_TO_GLOBAL_INDEX_KERNEL,
    flat_kv_row_view,
)
from vllm.v1.attention.backends.utils import get_supported_kv_cache_layouts
from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, UniformTypeKVCacheSpecs
from vllm.v1.kv_cache_layout import KVCacheLayout
from vllm.v1.worker.utils import allocate_kv_cache

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not current_platform.is_cuda(), reason="CUDA only"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "layout", [KVCacheLayout.BLHNC, KVCacheLayout.BLNHC, KVCacheLayout.LBNHC]
)
@pytest.mark.parametrize(
    "backend,cache_dtype,sm100,stride_alignment",
    [
        (FlashInferMLASparseTRTLLMBackend, "auto", True, 1152),
        (FlashInferMLASparseTRTLLMBackend, "fp8", True, 576),
        (FlashMLASparseBackend, "auto", True, 1152),
        (FlashMLASparseBackend, "fp8_ds_mla", False, None),
        (FlashMLASparseBackend, "fp8_ds_mla", True, 656),
        (FlashMLASparseBackend, "nvfp4_ds_mla", True, 352),
    ],
)
def test_allocation_and_warmup_follow_addressing_mode(
    monkeypatch, layout, backend, cache_dtype, sm100, stride_alignment, device
):
    monkeypatch.setattr(
        current_platform, "is_device_capability_family", lambda family: sm100
    )
    config = SimpleNamespace(
        cache_config=CacheConfig(block_size=64),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        attention_config=SimpleNamespace(hisparse_config=None),
        kernel_config=SimpleNamespace(enable_jit_warmup=True),
    )
    config.cache_config.kv_cache_layout = layout.name
    layer = SimpleNamespace(
        _vllm_config=config,
        attn_backend=backend,
        kv_cache_dtype=cache_dtype,
        head_size=576,
        sliding_window=None,
        indexer=None,
        non_causal_multi_token_decode=False,
    )
    layer._uses_flat_kv_cache = MethodType(MLAAttention._uses_flat_kv_cache, layer)
    spec = MLAAttention.get_kv_cache_spec(layer, config)
    indexer = SimpleNamespace(
        cache_config=config.cache_config, head_dim=132, dtype=torch.uint8
    )
    index_spec = DeepseekV32IndexerCache.get_kv_cache_spec(indexer, config)
    assert spec.block_stride_alignment == stride_alignment
    assert index_spec.block_stride_alignment is None
    assert layout in get_supported_kv_cache_layouts(
        [backend, DeepseekV32IndexerBackend]
    )
    specs = {"mla": spec, "indexer": index_spec}
    group = KVCacheGroupSpec(
        list(specs), UniformTypeKVCacheSpecs(block_size=64, kv_cache_specs=specs)
    )
    cache = get_kv_cache_config_from_groups(config, [group], 2**20)
    views = allocate_kv_cache(cache, torch.device(device), layout)
    register = MagicMock()
    monkeypatch.setattr(
        _CONVERT_REQ_INDEX_TO_GLOBAL_INDEX_KERNEL, "register_warmup", register
    )
    MLAAttention.bind_kv_cache(layer, views["mla"])
    DeepseekV32IndexerCache.bind_kv_cache(indexer, views["indexer"])
    assert indexer.kv_cache.stride(0) == views["indexer"].stride(0)
    if stride_alignment is not None:
        assert (
            views["mla"].stride(0) * views["mla"].element_size() % stride_alignment == 0
        )
    if not layer._uses_flat_kv_cache():
        register.assert_not_called()
    else:
        rows, stride = flat_kv_row_view(layer.kv_cache, 64)
        register.assert_called_once_with(config, block_stride_rows=stride)
        layer.kv_cache[1, 0].fill_(3)
        torch.testing.assert_close(rows[stride], layer.kv_cache[1, 0])
    if stride_alignment is None:
        assert cache.kv_cache_tensors[0].size == cache.num_blocks * sum(
            item.page_size_bytes for item in specs.values()
        )
