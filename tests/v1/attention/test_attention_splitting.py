# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.v1.attention.test_attention_backends import BATCH_SPECS
from tests.v1.attention.utils import create_common_attn_metadata
from vllm.v1.attention.backends.utils import (UBatchSlice,
                                              _make_metadata_with_slice,
                                              slice_query_start_locs,
                                              split_attn_metadata)


@pytest.fixture
def sample_query_start_loc():
    """Sample query_start_loc tensor for testing"""
    return torch.tensor([0, 5, 12, 20, 35, 50])


def test_basic_slice_middle(sample_query_start_loc):
    """Test slicing from middle of tensor"""
    req_slice = slice(1, 3)  # slice from index 1 to 3
    result = slice_query_start_locs(sample_query_start_loc, req_slice)

    expected = torch.tensor([0, 7, 15])
    assert torch.equal(result, expected)


def test_slice_from_beginning(sample_query_start_loc):
    """Test slicing from the beginning of tensor"""
    req_slice = slice(0, 2)  # slice from index 0 to 2
    result = slice_query_start_locs(sample_query_start_loc, req_slice)

    expected = torch.tensor([0, 5, 12])
    assert torch.equal(result, expected)


def test_slice_to_end(sample_query_start_loc):
    """Test slicing to the end of tensor"""
    req_slice = slice(3, 5)  # slice from index 3 to 5 (last index)
    result = slice_query_start_locs(sample_query_start_loc, req_slice)

    expected = torch.tensor([0, 15, 30])
    assert torch.equal(result, expected)


def test_single_element_slice(sample_query_start_loc):
    """Test slice that results in single element"""
    req_slice = slice(2, 3)  # slice from index 2 to 3
    result = slice_query_start_locs(sample_query_start_loc, req_slice)

    expected = torch.tensor([0, 8])
    assert torch.equal(result, expected)


def test_full_tensor_slice(sample_query_start_loc):
    """Test slicing the entire tensor"""
    req_slice = slice(0, 5)  # slice entire tensor
    result = slice_query_start_locs(sample_query_start_loc, req_slice)

    expected = torch.tensor([0, 5, 12, 20, 35, 50])
    assert torch.equal(result, expected)


def test_slice_bounds_edge_cases(sample_query_start_loc):
    # Test slice that goes exactly to the last element
    req_slice = slice(4, 5)  # Last index
    result = slice_query_start_locs(sample_query_start_loc, req_slice)

    expected = torch.tensor([0, 15])
    assert torch.equal(result, expected)


@pytest.fixture
def small_decode_metadata():
    """Create metadata for small decode batch"""
    batch_spec = BATCH_SPECS["small_decode"]
    device = torch.device("cpu")
    return create_common_attn_metadata(batch_spec,
                                       block_size=16,
                                       device=device)


@pytest.fixture
def large_decode_metadata():
    """Create metadata for small decode batch"""
    batch_spec = BATCH_SPECS["large_decode"]
    device = torch.device("cpu")
    return create_common_attn_metadata(batch_spec,
                                       block_size=16,
                                       device=device)


@pytest.fixture
def mixed_small_metadata():
    """Create metadata for mixed small batch"""
    batch_spec = BATCH_SPECS["mixed_small"]
    device = torch.device("cpu")
    return create_common_attn_metadata(batch_spec,
                                       block_size=16,
                                       device=device)


# Tests for _make_metadata_with_slice
def test_make_metadata_with_slice_decode_batch(small_decode_metadata):
    """Test slicing decode batch metadata"""
    # Split first request only
    ubatch_slice = UBatchSlice(slice(0, 1), slice(0, 1))

    result = _make_metadata_with_slice(ubatch_slice, small_decode_metadata)

    # Check sliced results
    assert result.num_reqs == 1  # slice(0, 1) gives 1 requests
    assert result.num_actual_tokens == 1  # slice(0, 1) gives 1 token
    assert result.max_query_len == 1
    assert torch.equal(result.query_start_loc, torch.tensor([0, 1]))
    assert torch.equal(result.seq_lens, torch.tensor([32]))


def test_make_metadata_with_slice_mixed_batch(mixed_small_metadata):
    """Test slicing mixed batch metadata"""
    ubatch_slice = UBatchSlice(slice(1, 3),
                               slice(1, 7))  # Requests 1-3, tokens 1-7

    result = _make_metadata_with_slice(ubatch_slice, mixed_small_metadata)

    assert result.num_reqs == 2  # slice(1, 3) gives 2 requests
    assert result.num_actual_tokens == 6  # slice(1, 7) gives 6 tokens
    assert result.max_query_len == 5
    assert torch.equal(result.query_start_loc, torch.tensor([0, 1, 6]))
    assert torch.equal(result.seq_lens, torch.tensor([40, 48]))


def test_split_attn_metadata_decode_batch(large_decode_metadata):
    """Test splitting decode batch into two equal parts"""
    num_tokens = large_decode_metadata.num_reqs
    mid_point = num_tokens // 2
    ubatch_slices = [
        UBatchSlice(slice(0, mid_point), slice(0, mid_point)),
        UBatchSlice(slice(mid_point, num_tokens), slice(mid_point,
                                                        num_tokens)),
    ]

    results = split_attn_metadata(ubatch_slices, large_decode_metadata)

    assert len(results) == 2

    # Check first split
    assert results[0].num_reqs == mid_point
    assert results[0].num_actual_tokens == mid_point
    assert torch.equal(results[0].seq_lens, torch.tensor([2048] * mid_point))

    # Check second split
    assert results[1].num_reqs == mid_point
    assert results[1].num_actual_tokens == mid_point
    assert torch.equal(results[1].seq_lens, torch.tensor([2048] * mid_point))
