# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerBackend,
    DeepseekV32IndexerMetadataBuilder,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec

ROW = 132  # 128 fp8 bytes + 4 scale bytes per indexer state


@pytest.mark.parametrize(("page_states", "stride_pages"), [(32, 7), (64, 5)])
def test_kernel_page_table_uses_physical_stride(page_states, stride_pages):
    pages_per_block = 4
    builder = object.__new__(DeepseekV32IndexerMetadataBuilder)
    builder.kv_cache_spec = MLAAttentionSpec(
        block_size=pages_per_block * page_states * 4,
        num_kv_heads=1,
        head_size=128,
        head_size_v=0,
        dtype=torch.uint8,
        state_content_bytes=ROW,
        tokens_per_state=4,
        kernel_page_size=page_states * 4,
    )
    builder._kernel_page_size = builder.kv_cache_spec.kernel_page_size
    builder.block_stride_bytes = stride_pages * page_states * ROW
    builder.arange_buffer = torch.arange(16, dtype=torch.int32)

    page_size, page_table = builder._expand_kernel_page_table(
        torch.tensor([[0, 2]], dtype=torch.int32)
    )
    assert page_size == page_states
    assert page_table.tolist() == [
        list(range(pages_per_block))
        + list(range(2 * stride_pages, 2 * stride_pages + pages_per_block))
    ]


@pytest.mark.parametrize(
    ("block_size", "expected_kernel_page", "expected_alignment"),
    [
        (640, 128, 32 * ROW),
        (1024, 256, 64 * ROW),
        (1152, 128, 32 * ROW),
        (256, 256, None),
    ],
)
def test_kpool_spec_declares_repage_alignment(
    default_vllm_config, block_size, expected_kernel_page, expected_alignment
):
    from vllm.models.glm5next.common.attention import Glm5NextIndexerCache

    default_vllm_config.cache_config.block_size = block_size
    cache = Glm5NextIndexerCache(
        head_dim=ROW,
        dtype=torch.uint8,
        prefix=f"indexer_{block_size}",
        cache_config=default_vllm_config.cache_config,
        index_kpool=4,
    )
    spec = cache.get_kv_cache_spec(default_vllm_config)
    assert spec.kernel_page_size == expected_kernel_page
    assert spec.block_stride_alignment == expected_alignment
    assert DeepseekV32IndexerBackend.get_supported_kernel_block_sizes(spec) == [
        block_size
    ]
