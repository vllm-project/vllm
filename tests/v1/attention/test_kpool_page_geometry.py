# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerMetadataBuilder
from vllm.v1.kv_cache_interface import MLAAttentionSpec

ROW = 132  # 128 fp8 bytes + 4 scale bytes per indexer state


def test_packed_page_table_uses_physical_stride():
    page_states, stride_pages = 64, 5
    builder = object.__new__(DeepseekV32IndexerMetadataBuilder)
    builder.kv_cache_spec = MLAAttentionSpec(
        block_size=1024,
        num_kv_heads=1,
        head_size=128,
        head_size_v=0,
        dtype=torch.uint8,
        state_content_bytes=ROW,
        tokens_per_state=4,
    )
    builder.block_stride_bytes = stride_pages * page_states * ROW
    builder.arange_buffer = torch.arange(16, dtype=torch.int32)

    page_size, page_table = builder._indexer_page_table(
        torch.tensor([[0, 2]], dtype=torch.int32)
    )
    assert page_size == page_states
    assert page_table.tolist() == [[0, 1, 2, 3, 10, 11, 12, 13]]


@pytest.mark.parametrize(
    ("block_size", "expected_page"),
    [(1024, 64), (1152, 32), (256, None)],
)
def test_kpool_spec_declares_repage_alignment(
    default_vllm_config, block_size, expected_page
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
    expected = None if expected_page is None else expected_page * ROW
    assert spec.block_stride_alignment == expected
