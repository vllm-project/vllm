# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.utils import cdiv, get_dtype_size
from vllm.v1.kv_cache_interface import FullAttentionSpec


@pytest.fixture
def base_attention_params():
    return {
        'block_size': 16,
        'num_kv_heads': 4,
        'head_size': 64,
        'dtype': torch.float16
    }


@pytest.fixture
def full_attention_spec(base_attention_params):
    return FullAttentionSpec(**base_attention_params, use_mla=False)


@pytest.fixture
def mla_attention_spec(base_attention_params):
    return FullAttentionSpec(**base_attention_params, use_mla=True)


class TestFullAttentionSpec:

    def test_type_id(self, full_attention_spec):
        expected_type_id = (f"full_attention_{full_attention_spec.block_size}_"
                            f"{full_attention_spec.page_size_bytes}")
        assert full_attention_spec.type_id == expected_type_id

    def test_page_size_bytes_standard(self, full_attention_spec,
                                      base_attention_params):
        expected_page_size = (2 * base_attention_params['block_size'] *
                              base_attention_params['num_kv_heads'] *
                              base_attention_params['head_size'] *
                              get_dtype_size(base_attention_params['dtype']))
        assert full_attention_spec.page_size_bytes == expected_page_size

    @pytest.mark.parametrize(
        "num_tokens,expected_pages",
        [
            (16, 1),  # Exactly one page
            (17, 2),  # Just over one page -> needs two pages
            (32, 2),  # Exactly two pages
            (53, 4),  # Non-multiple of block_size
        ])
    def test_bytes_for_tokens(self, full_attention_spec, num_tokens,
                              expected_pages):
        expected_bytes = expected_pages * full_attention_spec.page_size_bytes
        assert full_attention_spec.bytes_for_tokens(
            num_tokens) == expected_bytes

    def test_mla_page_size(self, mla_attention_spec, base_attention_params):
        expected_page_size = (base_attention_params['block_size'] *
                              base_attention_params['num_kv_heads'] *
                              base_attention_params['head_size'] *
                              get_dtype_size(base_attention_params['dtype']))
        assert mla_attention_spec.page_size_bytes == expected_page_size

    @pytest.mark.parametrize("block_size", [8, 16, 32, 64])
    def test_different_block_sizes(self, base_attention_params, block_size):
        params = base_attention_params.copy()
        params['block_size'] = block_size
        spec = FullAttentionSpec(**params, use_mla=False)

        expected_page_size = 2 * block_size * params['num_kv_heads'] * params[
            'head_size'] * get_dtype_size(params['dtype'])
        assert spec.page_size_bytes == expected_page_size

        num_tokens = block_size * 3 + 5
        expected_bytes = cdiv(num_tokens, block_size) * spec.page_size_bytes
        assert spec.bytes_for_tokens(num_tokens) == expected_bytes
