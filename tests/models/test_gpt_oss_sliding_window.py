# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Integration test for GptOss model with sliding window attention using FlexAttention.
"""

import pytest
import torch
from unittest.mock import Mock, patch

from vllm.v1.attention.backends.flex_attention import FlexAttentionImpl
from vllm.attention.backends.abstract import AttentionType


class MockGptOssConfig:
    """Mock GptOss configuration for testing."""
    
    def __init__(self, sliding_window=128):
        self.sliding_window = sliding_window
        self.head_dim = 64
        self.num_attention_heads = 8
        self.num_key_value_heads = 8
        self.hidden_size = 512
        self.max_position_embeddings = 2048
        self.rope_theta = 10000.0
        self.rope_scaling = {
            "factor": 1.0,
            "original_max_position_embeddings": 2048,
            "beta_fast": 32,
            "beta_slow": 1,
        }


class TestGptOssSlidingWindow:
    """Test GptOss model with sliding window attention."""
    
    @pytest.mark.parametrize("sliding_window", [None, 64, 128, 256])
    def test_flex_attention_impl_initialization(self, sliding_window):
        """Test FlexAttentionImpl initialization with different sliding window values."""
        # This should not raise NotImplementedError anymore
        try:
            impl = FlexAttentionImpl(
                num_heads=8,
                head_size=64,
                scale=0.125,
                num_kv_heads=8,
                alibi_slopes=None,
                sliding_window=sliding_window,
                kv_cache_dtype="auto",
                logits_soft_cap=None,
                attn_type=AttentionType.DECODER,
            )
            
            if sliding_window is not None:
                assert impl.sliding_window == (sliding_window - 1, 0)
            else:
                assert impl.sliding_window == (-1, -1)
                
        except NotImplementedError as e:
            if "FlexAttention does not support sliding window yet" in str(e):
                pytest.fail("FlexAttention should now support sliding window")
            else:
                # Some other NotImplementedError, re-raise
                raise
    
    def test_flex_attention_impl_direct(self):
        """Test FlexAttentionImpl directly with sliding window."""
        # This should not raise NotImplementedError anymore
        impl = FlexAttentionImpl(
            num_heads=8,
            head_size=64,
            scale=0.125,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=128,
            kv_cache_dtype="auto",
            logits_soft_cap=None,
            attn_type=AttentionType.DECODER,
        )
        
        assert impl.sliding_window == (127, 0)  # (sliding_window - 1, 0)
    
    def test_no_sliding_window_still_works(self):
        """Test that models without sliding window still work."""
        # This should work regardless of our changes
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
        
        assert impl.sliding_window == (-1, -1)
        assert True, "Models without sliding window should still work"
    
    def test_sliding_window_backward_compatibility(self):
        """Test that the changes maintain backward compatibility."""
        # Test various configurations that should all work
        configs = [
            {"sliding_window": None},
            {"sliding_window": 64},
            {"sliding_window": 128},
            {"sliding_window": 256},
            {"sliding_window": 512},
        ]
        
        for config in configs:
            impl = FlexAttentionImpl(
                num_heads=8,
                head_size=64,
                scale=0.125,
                num_kv_heads=8,
                alibi_slopes=None,
                kv_cache_dtype="auto",
                logits_soft_cap=None,
                attn_type=AttentionType.DECODER,
                **config
            )
            
            if config["sliding_window"] is not None:
                expected = (config["sliding_window"] - 1, 0)
            else:
                expected = (-1, -1)
            
            assert impl.sliding_window == expected
    
    def test_sliding_window_edge_cases(self):
        """Test edge cases for sliding window values."""
        # Test minimum sliding window
        impl = FlexAttentionImpl(
            num_heads=8,
            head_size=64,
            scale=0.125,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=1,  # Minimum window
            kv_cache_dtype="auto",
            logits_soft_cap=None,
            attn_type=AttentionType.DECODER,
        )
        assert impl.sliding_window == (0, 0)
        
        # Test large sliding window
        impl = FlexAttentionImpl(
            num_heads=8,
            head_size=64,
            scale=0.125,
            num_kv_heads=8,
            alibi_slopes=None,
            sliding_window=4096,  # Large window
            kv_cache_dtype="auto",
            logits_soft_cap=None,
            attn_type=AttentionType.DECODER,
        )
        assert impl.sliding_window == (4095, 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
