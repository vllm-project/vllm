# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Tests for FlexAttention sliding window support.
"""

import pytest
import torch
from unittest.mock import Mock

from vllm.v1.attention.backends.flex_attention import (
    FlexAttentionImpl, 
    FlexAttentionMetadata,
    sliding_window_causal_mask_mod,
    causal_mask_mod
)
from vllm.attention.backends.abstract import AttentionType


class TestFlexAttentionSlidingWindow:
    """Test FlexAttention sliding window functionality."""
    
    def test_sliding_window_mask_function(self):
        """Test the sliding window mask function logic."""
        window_size = 4
        mask_func = sliding_window_causal_mask_mod(window_size)
        
        # Create test tensors
        b = torch.tensor([0])
        h = torch.tensor([0])
        
        # Test cases: (q_idx, kv_idx, expected_result)
        test_cases = [
            (5, 5, True),   # Current token (causal + within window)
            (5, 4, True),   # Previous token within window
            (5, 2, True),   # Token at window boundary (5-2=3 < 4)
            (5, 1, False),  # Token outside window (5-1=4 >= 4)
            (5, 0, False),  # Token far outside window
            (3, 4, False),  # Future token (violates causal)
        ]
        
        for q_idx, kv_idx, expected in test_cases:
            q_tensor = torch.tensor([q_idx])
            kv_tensor = torch.tensor([kv_idx])
            result = mask_func(b, h, q_tensor, kv_tensor)
            assert result.item() == expected, \
                f"Failed for q_idx={q_idx}, kv_idx={kv_idx}. Expected {expected}, got {result.item()}"
    
    def test_causal_mask_function(self):
        """Test the standard causal mask function."""
        # Test cases: (q_idx, kv_idx, expected_result)
        test_cases = [
            (5, 5, True),   # Current token
            (5, 4, True),   # Previous token
            (5, 0, True),   # Much earlier token
            (3, 4, False),  # Future token
        ]
        
        b = torch.tensor([0])
        h = torch.tensor([0])
        
        for q_idx, kv_idx, expected in test_cases:
            q_tensor = torch.tensor([q_idx])
            kv_tensor = torch.tensor([kv_idx])
            result = causal_mask_mod(b, h, q_tensor, kv_tensor)
            assert result.item() == expected, \
                f"Failed for q_idx={q_idx}, kv_idx={kv_idx}. Expected {expected}, got {result.item()}"
    
    def test_flex_attention_impl_with_sliding_window(self):
        """Test FlexAttentionImpl initialization with sliding window."""
        # Test with sliding window
        impl = FlexAttentionImpl(
            num_heads=8,
            head_size=64,
            scale=0.125,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=128,  # Set sliding window
            kv_cache_dtype="auto",
            logits_soft_cap=None,
            attn_type=AttentionType.DECODER,
        )
        
        # Should not raise NotImplementedError anymore
        assert impl.sliding_window == (127, 0)  # (window_size - 1, 0)
    
    def test_flex_attention_impl_without_sliding_window(self):
        """Test FlexAttentionImpl initialization without sliding window."""
        impl = FlexAttentionImpl(
            num_heads=8,
            head_size=64,
            scale=0.125,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=None,  # No sliding window
            kv_cache_dtype="auto",
            logits_soft_cap=None,
            attn_type=AttentionType.DECODER,
        )
        
        assert impl.sliding_window == (-1, -1)  # Default values
    
    def test_flex_attention_metadata_with_sliding_window(self):
        """Test FlexAttentionMetadata with sliding window."""
        # Create mock tensors
        device = torch.device('cpu')
        batch_size = 2
        seq_len = 10
        
        metadata = FlexAttentionMetadata(
            causal=True,
            num_actual_tokens=batch_size,
            max_query_len=1,
            query_start_loc=torch.tensor([0, 1, 2]),
            max_seq_len=seq_len,
            seq_lens=torch.tensor([seq_len, seq_len]),
            block_table=torch.zeros((batch_size, 4), dtype=torch.int32),
            slot_mapping=torch.zeros(batch_size, dtype=torch.int64),
            use_cascade=False,
            common_prefix_len=0,
            cu_prefix_query_lens=None,
            prefix_kv_lens=None,
            suffix_kv_lens=None,
            total_cache_tokens=100,
            block_size=16,
            max_possible_sequence_length=1024,
            num_reqs=batch_size,
            physical_to_logical=torch.zeros((4, 16), dtype=torch.int32),
            decode_offset=torch.zeros(batch_size, dtype=torch.int32),
            num_blocks_per_seq=torch.ones(batch_size, dtype=torch.int32),
            sliding_window=64,  # Set sliding window
        )
        
        # Test that sliding window is properly set
        assert metadata.sliding_window == 64
    
    def test_flex_attention_metadata_without_sliding_window(self):
        """Test FlexAttentionMetadata without sliding window."""
        # Create mock tensors
        device = torch.device('cpu')
        batch_size = 2
        seq_len = 10
        
        metadata = FlexAttentionMetadata(
            causal=True,
            num_actual_tokens=batch_size,
            max_query_len=1,
            query_start_loc=torch.tensor([0, 1, 2]),
            max_seq_len=seq_len,
            seq_lens=torch.tensor([seq_len, seq_len]),
            block_table=torch.zeros((batch_size, 4), dtype=torch.int32),
            slot_mapping=torch.zeros(batch_size, dtype=torch.int64),
            use_cascade=False,
            common_prefix_len=0,
            cu_prefix_query_lens=None,
            prefix_kv_lens=None,
            suffix_kv_lens=None,
            total_cache_tokens=100,
            block_size=16,
            max_possible_sequence_length=1024,
            num_reqs=batch_size,
            physical_to_logical=torch.zeros((4, 16), dtype=torch.int32),
            decode_offset=torch.zeros(batch_size, dtype=torch.int32),
            num_blocks_per_seq=torch.ones(batch_size, dtype=torch.int32),
            sliding_window=None,  # No sliding window
        )
        
        # Test that sliding window is None
        assert metadata.sliding_window is None
    
    def test_sliding_window_edge_cases(self):
        """Test edge cases for sliding window."""
        # Test window size of 1
        mask_func = sliding_window_causal_mask_mod(1)
        b = torch.tensor([0])
        h = torch.tensor([0])
        
        # Only current token should be visible
        assert mask_func(b, h, torch.tensor([5]), torch.tensor([5])).item() == True
        assert mask_func(b, h, torch.tensor([5]), torch.tensor([4])).item() == False
        
        # Test large window size
        mask_func = sliding_window_causal_mask_mod(1000)
        
        # Should behave like standard causal mask for reasonable sequence lengths
        assert mask_func(b, h, torch.tensor([10]), torch.tensor([5])).item() == True
        assert mask_func(b, h, torch.tensor([10]), torch.tensor([15])).item() == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
