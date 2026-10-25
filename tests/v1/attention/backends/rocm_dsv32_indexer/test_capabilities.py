# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capability contract of the DSA indexer backends on ROCm.

``DeepseekV32IndexerBackend`` (DeepSeek V3.2, GLM-5.2/5.3) and its GLM-5.3-
Flash subclasses (``Glm5NextIndexerBackend`` for the kpool indexer cache,
``KpoolTailBackend`` for the storage-only tail cache) only hold the indexer's
fp8 K cache; the declarations below decide the cache's block size, layout
and CUDA-graph level, so each is pinned and checked both ways.
"""

import pytest
import torch

from tests.v1.attention.backends.rocm_dsv32_indexer.idx_harness import (
    make_idx_case,
)
from tests.v1.attention.model_shapes import DEEPSEEK_V3_2
from vllm.v1.attention.backend import AttentionCGSupport, MultipleOf
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend,
    DeepseekV32IndexerMetadataBuilder,
    Glm5NextIndexerBackend,
    KpoolTailBackend,
)
from vllm.v1.kv_cache_interface import KVCacheLayout


def test_dsv32_indexer_declarations():
    backend = DeepseekV32IndexerBackend
    assert backend.get_name() == "DEEPSEEK_V32_INDEXER"
    assert backend.get_supported_kernel_block_sizes() == [1, MultipleOf(16)]
    assert backend.get_supported_head_sizes() == [32, 64, 128]
    # ROCm has neither the varlen nor the flattened device-query-lens kernel.
    assert not backend.supports_device_cpu_query_lens_mismatch()
    assert backend.get_builder_cls() is DeepseekV32IndexerMetadataBuilder


@pytest.mark.parametrize("block_size", [1, 16, 32, 64, 128])
def test_dsv32_indexer_accepts_block_sizes(block_size):
    assert DeepseekV32IndexerBackend.supports_block_size(block_size)


def test_kpool_tail_declarations():
    assert KpoolTailBackend.get_name() == "KPOOL_TAIL"
    assert KpoolTailBackend.supported_kv_cache_layouts() == (KVCacheLayout.BLHNC,)
    assert KpoolTailBackend.get_supported_kernel_block_sizes() == [MultipleOf(1)]
    assert issubclass(Glm5NextIndexerBackend, DeepseekV32IndexerBackend)


@pytest.mark.parametrize(
    "spec_tokens, support",
    [
        (0, AttentionCGSupport.UNIFORM_BATCH),
        (1, AttentionCGSupport.UNIFORM_BATCH),  # native next_n 2
        (3, AttentionCGSupport.ALWAYS),  # flattened per token
    ],
)
def test_cudagraph_support_follows_the_decode_path(env, spec_tokens, support):
    case = make_idx_case(DEEPSEEK_V3_2, [64], spec_tokens=spec_tokens)
    got = DeepseekV32IndexerMetadataBuilder.get_cudagraph_support(
        case.config, case.builder.kv_cache_spec
    )
    assert got == support
    assert case.builder.use_flattening == (spec_tokens > 1)


def test_indexer_cache_layout_matches_its_writer(env):
    """Block size 1 stores rows plainly; larger blocks use the shuffled
    16x16 tile layout the ROCm writer and gather agree on."""
    for block_size, shuffled in [(1, False), (16, True), (64, True)]:
        case = make_idx_case(DEEPSEEK_V3_2, [64], block_size=block_size)
        assert case.k_cache.uses_shuffled_layout == shuffled
        assert case.k_cache.kv_cache.dtype == torch.uint8
        assert case.k_cache.kv_cache.shape[-1] == 128 + 4
