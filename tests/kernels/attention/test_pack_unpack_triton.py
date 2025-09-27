# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch.testing import assert_close

from vllm.attention.ops.common import pack_seq_triton, unpack_seq_triton


def test_pack_decode_query_basic_fp8():
    """Test basic functionality of pack_seq_triton with fp8 and 3D tensors."""
    device = "cuda"
    dtype = torch.float8_e4m3fn
    
    # Test cases with 3D tensors (N, H, D)
    test_cases = [
        (6, 8, 4, 2, [3, 3]),      # (6, 8, 4) -> (2, 3, 8, 4)
        (10, 4, 8, 3, [2, 4, 4]),  # (10, 4, 8) -> (3, 4, 4, 8)
        (20, 16, 32, 4, [5, 5, 5, 5]),  # (20, 16, 32) -> (4, 5, 16, 32)
    ]
    
    for N, H, D, B, lengths_list in test_cases:
        # Create input tensor with small values for fp8
        x = torch.randn(N, H, D, dtype=torch.float32, device=device) * 0.1
        x = x.to(dtype=dtype)
        lengths = torch.tensor(lengths_list, device=device)
        
        # Pack the data
        packed = pack_seq_triton(x, lengths)
        
        # Check output shape and properties
        expected_shape = (B, max(lengths_list), H, D)
        assert packed.shape == expected_shape
        assert packed.dtype == dtype
        assert packed.device == x.device
        
        # Check that valid data is preserved (within fp8 precision)
        for b in range(B):
            start_idx = sum(lengths_list[:b])
            seq_len = lengths_list[b]
            
            expected_data = x[start_idx:start_idx + seq_len].to(torch.float32)
            actual_data = packed[b, :seq_len].to(torch.float32)
            
            assert_close(actual_data, expected_data, rtol=1e-1, atol=1e-2)


def test_pack_decode_query_custom_padding_fp8():
    """Test pack_seq_triton with custom padding values for fp8."""
    device = "cuda"
    dtype = torch.float8_e4m3fn
    N, H, D, B = 20, 8, 16, 2
    lengths = torch.tensor([10, 10], device=device)
    
    x = torch.randn(N, H, D, dtype=torch.float32, device=device) * 0.1
    x = x.to(dtype=dtype)
    
    # Test with different padding values
    for pad_value in [-100.0, -10.0, 0.0, 10.0, 100.0]:
        result = pack_seq_triton(x, lengths, pad_value=pad_value)
        
        # Check valid data
        for b in range(B):
            start_idx = b * 10
            expected_data = x[start_idx:start_idx + 10].to(torch.float32)
            actual_data = result[b, :10].to(torch.float32)
            assert_close(actual_data, expected_data, rtol=1e-1, atol=1e-2)
        
        # Check padding (fp8 has limited range, so check for large values)
        padded_data = result[:, 10:].to(torch.float32)
        if pad_value < 0:
            assert torch.all(padded_data < -50)  # Large negative values
        elif pad_value > 0:
            assert torch.all(padded_data > 50)   # Large positive values
        else:
            assert torch.allclose(padded_data, torch.zeros_like(padded_data), atol=1e-2)


def test_pack_decode_query_default_negative_inf_padding_fp8():
    """Test that pack_seq_triton uses -inf padding by default for fp8."""
    device = "cuda"
    dtype = torch.float8_e4m3fn
    N, H, D, B = 20, 8, 16, 2
    lengths = torch.tensor([10, 10], device=device)
    
    x = torch.randn(N, H, D, dtype=torch.float32, device=device) * 0.1
    x = x.to(dtype=dtype)
    result = pack_seq_triton(x, lengths)
    
    # Check that padding is large negative values (fp8 representation of -inf)
    padded_data = result[:, 10:].to(torch.float32)
    assert torch.all(padded_data < -100)  # fp8 -inf is represented as large negative number


def test_pack_decode_query_edge_cases_fp8():
    """Test pack_seq_triton with edge cases for fp8."""
    device = "cuda"
    dtype = torch.float8_e4m3fn
    
    # Test with single batch element
    x = torch.randn(10, 8, 16, dtype=torch.float32, device=device) * 0.1
    x = x.to(dtype=dtype)
    lengths = torch.tensor([10], device=device)
    result = pack_seq_triton(x, lengths)
    assert result.shape == (1, 10, 8, 16)
    
    # Test with very short sequences
    x = torch.randn(20, 4, 8, dtype=torch.float32, device=device) * 0.1
    x = x.to(dtype=dtype)
    lengths = torch.tensor([1, 1, 1], device=device)
    result = pack_seq_triton(x, lengths)
    assert result.shape == (3, 1, 4, 8)
    
    # Test with different sequence lengths
    x = torch.randn(15, 8, 16, dtype=torch.float32, device=device) * 0.1
    x = x.to(dtype=dtype)
    lengths = torch.tensor([5, 7, 3], device=device)
    result = pack_seq_triton(x, lengths)
    assert result.shape == (3, 7, 8, 16)


def test_pack_decode_query_different_block_sizes_fp8():
    """Test pack_seq_triton with different block sizes for fp8."""
    device = "cuda"
    dtype = torch.float8_e4m3fn
    N, H, D, B = 100, 16, 32, 4
    lengths = torch.tensor([25, 25, 25, 25], device=device)
    
    x = torch.randn(N, H, D, dtype=torch.float32, device=device) * 0.1
    x = x.to(dtype=dtype)
    
    # Test different block sizes
    for block_t, block_d in [(32, 32), (64, 64), (128, 128)]:
        result = pack_seq_triton(x, lengths, block_t=block_t, block_d=block_d)
        
        assert result.shape == (B, 25, H, D)
        
        # Check that valid data is preserved (within fp8 precision)
        for b in range(B):
            start_idx = b * 25
            expected_data = x[start_idx:start_idx + 25].to(torch.float32)
            actual_data = result[b, :25].to(torch.float32)
            assert_close(actual_data, expected_data, rtol=1e-1, atol=1e-2)


def test_pack_decode_query_shape_consistency():
    """Test that pack_seq_triton maintains shape consistency."""
    device = "cuda"
    dtype = torch.float8_e4m3fn
    N, H, D, B = 20, 8, 16, 2
    lengths = torch.tensor([10, 10], device=device)
    
    x = torch.randn(N, H, D, dtype=torch.float32, device=device) * 0.1
    x = x.to(dtype=dtype)
    
    result = pack_seq_triton(x, lengths)
    
    # Check shape consistency
    assert result.shape[0] == B  # Batch dimension
    assert result.shape[1] == lengths.max().item()  # Max sequence length
    assert result.shape[2:] == x.shape[1:]  # Feature dimensions preserved


def test_pack_unpack_roundtrip_fp8():
    """Test that pack -> unpack gives us back the original data for fp8."""
    device = "cuda"
    dtype = torch.float8_e4m3fn
    
    # Test cases with 3D tensors
    test_cases = [
        (6, 8, 4, 2, [3, 3]),
        (10, 4, 8, 3, [2, 4, 4]),
        (20, 16, 32, 4, [5, 5, 5, 5]),
        (15, 8, 16, 3, [7, 5, 3]),
    ]
    
    for N, H, D, B, lengths_list in test_cases:
        # Create input tensor with small values for fp8
        x = torch.randn(N, H, D, dtype=torch.float32, device=device) * 0.1
        x = x.to(dtype=dtype)
        lengths = torch.tensor(lengths_list, device=device)
        
        # Pack the data
        packed = pack_seq_triton(x, lengths)
        
        # Unpack the data
        unpacked = unpack_seq_triton(packed, lengths)
        
        # Check that we get back the original data (within fp8 precision)
        assert unpacked.shape == x.shape
        x_f32 = x.to(torch.float32)
        unpacked_f32 = unpacked.to(torch.float32)
        assert_close(x_f32, unpacked_f32, rtol=1e-1, atol=1e-2)
        
        # Test with query_start_loc
        query_start_loc = torch.cat([torch.zeros(1, device=device, dtype=lengths.dtype),
                                   lengths.cumsum(0)[:-1]])
        unpacked_with_loc = unpack_seq_triton(packed, lengths, query_start_loc)
        assert_close(x_f32, unpacked_with_loc.to(torch.float32), rtol=1e-1, atol=1e-2)


def test_unpack_seq_triton_edge_cases_fp8():
    """Test unpack function with edge cases for fp8."""
    device = "cuda"
    dtype = torch.float8_e4m3fn
    
    # Test with single batch element
    x = torch.randn(10, 8, 16, dtype=torch.float32, device=device) * 0.1
    x = x.to(dtype=dtype)
    lengths = torch.tensor([10], device=device)
    packed = pack_seq_triton(x, lengths)
    unpacked = unpack_seq_triton(packed, lengths)
    assert unpacked.shape == x.shape
    assert_close(x.to(torch.float32), unpacked.to(torch.float32), rtol=1e-1, atol=1e-2)
    
    # Test with very short sequences
    x = torch.randn(20, 4, 8, dtype=torch.float32, device=device) * 0.1
    x = x.to(dtype=dtype)
    lengths = torch.tensor([1, 1, 1], device=device)
    packed = pack_seq_triton(x, lengths)
    unpacked = unpack_seq_triton(packed, lengths)
    # Only compare the first 3 elements that were actually packed
    assert_close(x[:3].to(torch.float32), unpacked.to(torch.float32), rtol=1e-1, atol=1e-2)
    
    # Test with query_start_loc
    x = torch.randn(15, 8, 16, dtype=torch.float32, device=device) * 0.1
    x = x.to(dtype=dtype)
    lengths = torch.tensor([5, 7, 3], device=device)
    query_start_loc = torch.tensor([0, 5, 12], device=device)
    packed = pack_seq_triton(x, lengths)
    unpacked = unpack_seq_triton(packed, lengths, query_start_loc)
    assert unpacked.shape == x.shape
    assert_close(x.to(torch.float32), unpacked.to(torch.float32), rtol=1e-1, atol=1e-2)


def test_masked_topk_basic():
    """Test basic functionality of masked_topk function."""
    device = "cuda"
    
    # Test case 1: Simple example
    seq_lens = torch.tensor([2, 1], device=device)  # 2 batches: lengths 2,1
    starting_pos = torch.tensor([3, 7], device=device)  # starting positions
    N = seq_lens.sum().item()  # 3 total positions
    vocab_size, k = 20, 2
    
    scores = torch.randn(N, vocab_size, device=device)
    
    indices, top_scores = masked_topk(scores, seq_lens, starting_pos, k)
    
    # Check output shapes
    assert indices.shape == (N, k)
    assert top_scores.shape == (N, k)
    
    # Verify masking constraints
    # Positions 0,1 (batch 0): should only use indices < 3
    assert torch.all(indices[0] < 3)
    assert torch.all(indices[1] < 3)
    # Position 2 (batch 1): should only use indices < 7
    assert torch.all(indices[2] < 7)


def test_masked_topk_complex():
    """Test masked_topk with more complex sequences."""
    device = "cuda"
    
    # Test case: 4 batches with different lengths
    seq_lens = torch.tensor([3, 1, 1, 1], device=device)  # lengths: 3,1,1,1
    starting_pos = torch.tensor([4, 12, 33, 50], device=device)  # starting positions
    N = seq_lens.sum().item()  # 6 total positions
    vocab_size, k = 100, 3
    
    scores = torch.randn(N, vocab_size, device=device)
    
    indices, top_scores = masked_topk(scores, seq_lens, starting_pos, k)
    
    # Check output shapes
    assert indices.shape == (N, k)
    assert top_scores.shape == (N, k)
    
    # Verify masking constraints for each batch
    pos_idx = 0
    for b in range(len(seq_lens)):
        seq_len = seq_lens[b].item()
        start_pos = starting_pos[b].item()
        
        # Check all positions in this batch
        for i in range(seq_len):
            assert torch.all(indices[pos_idx] < start_pos), f"Position {pos_idx} should only use indices < {start_pos}"
            pos_idx += 1


def test_masked_topk_edge_cases():
    """Test masked_topk with edge cases."""
    device = "cuda"
    
    # Test case 1: Single batch
    seq_lens = torch.tensor([5], device=device)
    starting_pos = torch.tensor([10], device=device)
    scores = torch.randn(5, 50, device=device)
    
    indices, top_scores = masked_topk(scores, seq_lens, starting_pos, k=3)
    assert indices.shape == (5, 3)
    assert torch.all(indices < 10)  # All positions should use indices < 10
    
    # Test case 2: Very small starting positions
    seq_lens = torch.tensor([2, 1], device=device)
    starting_pos = torch.tensor([1, 2], device=device)
    scores = torch.randn(3, 20, device=device)
    
    indices, top_scores = masked_topk(scores, seq_lens, starting_pos, k=1)
    assert indices.shape == (3, 1)
    assert torch.all(indices[0] < 1)  # First position can only use index 0
    assert torch.all(indices[1] < 1)  # Second position can only use index 0
    assert torch.all(indices[2] < 2)  # Third position can use indices 0,1
    
    # Test case 3: Large starting positions
    seq_lens = torch.tensor([2], device=device)
    starting_pos = torch.tensor([95], device=device)
    scores = torch.randn(2, 100, device=device)
    
    indices, top_scores = masked_topk(scores, seq_lens, starting_pos, k=5)
    assert indices.shape == (2, 5)
    assert torch.all(indices < 95)


def test_masked_topk_different_k_values():
    """Test masked_topk with different k values."""
    device = "cuda"
    
    seq_lens = torch.tensor([2, 1], device=device)
    starting_pos = torch.tensor([5, 10], device=device)
    scores = torch.randn(3, 20, device=device)
    
    # Test different k values
    for k in [1, 3, 5, 10]:
        indices, top_scores = masked_topk(scores, seq_lens, starting_pos, k)
        
        assert indices.shape == (3, k)
        assert top_scores.shape == (3, k)
        
        # Verify masking constraints
        assert torch.all(indices[0] < 5)
        assert torch.all(indices[1] < 5)
        assert torch.all(indices[2] < 10)


def test_masked_topk_fp8():
    """Test masked_topk with fp8 dtype."""
    device = "cuda"
    dtype = torch.float8_e4m3fn
    
    seq_lens = torch.tensor([2, 1], device=device)
    starting_pos = torch.tensor([5, 10], device=device)
    
    # Create fp8 scores
    scores_f32 = torch.randn(3, 20, device=device) * 0.1
    scores = scores_f32.to(dtype)
    
    indices, top_scores = masked_topk(scores, seq_lens, starting_pos, k=3)
    
    # Check output shapes
    assert indices.shape == (3, 3)
    assert top_scores.shape == (3, 3)
    assert top_scores.dtype == dtype
    
    # Verify masking constraints
    assert torch.all(indices[0] < 5)
    assert torch.all(indices[1] < 5)
    assert torch.all(indices[2] < 10)
    
    # Check that top scores are reasonable (not all -inf)
    assert not torch.all(torch.isinf(top_scores.to(torch.float32)))


def test_masked_topk_consistency():
    """Test that masked_topk produces consistent results."""
    device = "cuda"
    
    seq_lens = torch.tensor([2, 1], device=device)
    starting_pos = torch.tensor([5, 10], device=device)
    
    # Use deterministic scores for consistency testing
    torch.manual_seed(42)
    scores = torch.randn(3, 20, device=device)
    
    # Run multiple times
    results = []
    for _ in range(3):
        indices, top_scores = masked_topk(scores, seq_lens, starting_pos, k=3)
        results.append((indices.clone(), top_scores.clone()))
    
    # Check that all runs produce identical results
    for i in range(1, len(results)):
        assert torch.equal(results[0][0], results[i][0]), "Indices should be consistent"
        assert_close(results[0][1], results[i][1], rtol=1e-5, atol=1e-5), "Scores should be consistent"
