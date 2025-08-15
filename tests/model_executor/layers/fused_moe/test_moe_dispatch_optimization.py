"""
Test suite for MoE token dispatch optimization.

This module tests the optimization where only leader DP ranks dispatch tokens
when TP > 1, reducing cross-rank communication overhead.
"""
import pytest
import torch
from unittest.mock import patch

from vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe import BatchedDeepGemmExperts


class TestMoEDispatchOptimization:
    """Test cases for MoE dispatch optimization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.max_num_tokens = 1024
        self.block_shape = [128, 128]
        self.per_act_token_quant = False
        
    def _create_batched_experts(self, num_dispatchers: int) -> BatchedDeepGemmExperts:
        """Helper to create BatchedDeepGemmExperts instance."""
        return BatchedDeepGemmExperts(
            max_num_tokens=self.max_num_tokens,
            num_dispatchers=num_dispatchers,
            block_shape=self.block_shape,
            per_act_token_quant=self.per_act_token_quant
        )

    def test_single_tp_rank_uses_all_dispatchers(self):
        """Test that single TP rank uses all dispatchers."""
        num_dispatchers = 8
        experts = self._create_batched_experts(num_dispatchers)
        
        # Mock TP size = 1, rank = 0
        with patch('vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe.get_tensor_model_parallel_world_size', return_value=1), \
             patch('vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe.get_tensor_model_parallel_rank', return_value=0):
            
            effective_dispatchers = experts._get_effective_num_dispatchers()
            assert effective_dispatchers == num_dispatchers, \
                f"Expected {num_dispatchers} dispatchers, got {effective_dispatchers}"

    def test_tp_leader_rank_uses_reduced_dispatchers(self):
        """Test that TP leader rank uses reduced number of dispatchers."""
        num_dispatchers = 8
        tp_size = 2
        experts = self._create_batched_experts(num_dispatchers)
        
        # Mock TP size = 2, rank = 0 (leader)
        with patch('vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe.get_tensor_model_parallel_world_size', return_value=tp_size), \
             patch('vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe.get_tensor_model_parallel_rank', return_value=0):
            
            effective_dispatchers = experts._get_effective_num_dispatchers()
            expected_dispatchers = num_dispatchers // tp_size
            assert effective_dispatchers == expected_dispatchers, \
                f"Expected {expected_dispatchers} dispatchers, got {effective_dispatchers}"

    def test_tp_non_leader_rank_uses_minimal_dispatchers(self):
        """Test that TP non-leader rank uses minimal dispatchers."""
        num_dispatchers = 8
        tp_size = 2
        experts = self._create_batched_experts(num_dispatchers)
        
        # Mock TP size = 2, rank = 1 (non-leader)
        with patch('vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe.get_tensor_model_parallel_world_size', return_value=tp_size), \
             patch('vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe.get_tensor_model_parallel_rank', return_value=1):
            
            effective_dispatchers = experts._get_effective_num_dispatchers()
            # Should use at least 1 dispatcher even for non-leader ranks
            assert effective_dispatchers == 1, \
                f"Expected 1 dispatcher for non-leader rank, got {effective_dispatchers}"

    def test_various_tp_sizes(self):
        """Test optimization with various TP sizes."""
        num_dispatchers = 16
        experts = self._create_batched_experts(num_dispatchers)
        
        test_cases = [
            (1, 0, 16),  # TP=1: all dispatchers
            (2, 0, 8),   # TP=2, leader: half dispatchers
            (2, 1, 1),   # TP=2, non-leader: minimal dispatchers
            (4, 0, 4),   # TP=4, leader: quarter dispatchers
            (4, 3, 1),   # TP=4, non-leader: minimal dispatchers
            (8, 0, 2),   # TP=8, leader: eighth dispatchers
            (8, 7, 1),   # TP=8, non-leader: minimal dispatchers
        ]
        
        for tp_size, tp_rank, expected in test_cases:
            with patch('vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe.get_tensor_model_parallel_world_size', return_value=tp_size), \
                 patch('vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe.get_tensor_model_parallel_rank', return_value=tp_rank):
                
                effective_dispatchers = experts._get_effective_num_dispatchers()
                assert effective_dispatchers == expected, \
                    f"TP={tp_size}, rank={tp_rank}: expected {expected}, got {effective_dispatchers}"

    def test_workspace_shapes_with_optimization(self):
        """Test that workspace shapes reflect the dispatch optimization."""
        num_dispatchers = 8
        tp_size = 2
        experts = self._create_batched_experts(num_dispatchers)
        
        # Create dummy tensors for testing
        batch_size = 32
        hidden_size = 1024
        a = torch.randn(batch_size, hidden_size)
        aq = torch.empty_like(a)  # Not used but required
        
        # Test with TP leader rank
        with patch('vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe.get_tensor_model_parallel_world_size', return_value=tp_size), \
             patch('vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe.get_tensor_model_parallel_rank', return_value=0):
            
            shapes = experts.workspace_shapes(
                a=a, aq=aq, M=batch_size, N=hidden_size, K=hidden_size,
                topk=2, global_num_experts=8, local_num_experts=4,
                expert_tokens_metadata=None
            )
            
            workspace13_shape, workspace2_shape, output_shape, dtype = shapes
            
            # Expected effective dispatchers = 8 // 2 = 4
            expected_token_dim = self.max_num_tokens * 4
            
            assert workspace13_shape[1] == expected_token_dim, \
                f"Expected token dimension {expected_token_dim}, got {workspace13_shape[1]}"
            assert workspace2_shape[1] == expected_token_dim, \
                f"Expected token dimension {expected_token_dim}, got {workspace2_shape[1]}"
            assert output_shape[1] == expected_token_dim, \
                f"Expected token dimension {expected_token_dim}, got {output_shape[1]}"

    def test_minimal_dispatchers_guarantee(self):
        """Test that we always have at least 1 dispatcher."""
        # Edge case: very small num_dispatchers with large TP size
        num_dispatchers = 2
        tp_size = 8
        experts = self._create_batched_experts(num_dispatchers)
        
        # Test leader rank
        with patch('vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe.get_tensor_model_parallel_world_size', return_value=tp_size), \
             patch('vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe.get_tensor_model_parallel_rank', return_value=0):
            
            effective_dispatchers = experts._get_effective_num_dispatchers()
            assert effective_dispatchers >= 1, \
                f"Expected at least 1 dispatcher, got {effective_dispatchers}"
        
        # Test non-leader rank
        with patch('vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe.get_tensor_model_parallel_world_size', return_value=tp_size), \
             patch('vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe.get_tensor_model_parallel_rank', return_value=1):
            
            effective_dispatchers = experts._get_effective_num_dispatchers()
            assert effective_dispatchers >= 1, \
                f"Expected at least 1 dispatcher, got {effective_dispatchers}"

    def test_communication_reduction_calculation(self):
        """Test that the optimization achieves expected communication reduction."""
        num_dispatchers = 16
        test_configs = [
            (1, 1.0),   # TP=1: no reduction
            (2, 2.0),   # TP=2: 2x reduction (only leader dispatches)
            (4, 4.0),   # TP=4: 4x reduction
            (8, 8.0),   # TP=8: 8x reduction
        ]
        
        experts = self._create_batched_experts(num_dispatchers)
        
        for tp_size, expected_reduction in test_configs:
            # Calculate original total dispatchers (all ranks dispatch)
            original_total_dispatchers = num_dispatchers
            
            # Calculate optimized total dispatchers (only leaders dispatch)
            with patch('vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe.get_tensor_model_parallel_world_size', return_value=tp_size), \
                 patch('vllm.model_executor.layers.fused_moe.batched_deep_gemm_moe.get_tensor_model_parallel_rank', return_value=0):
                
                leader_dispatchers = experts._get_effective_num_dispatchers()
                # In distributed setting, only leader ranks would dispatch
                optimized_total_dispatchers = leader_dispatchers
                
                if tp_size == 1:
                    # No optimization for single TP
                    actual_reduction = 1.0
                else:
                    # Reduction ratio
                    actual_reduction = original_total_dispatchers / optimized_total_dispatchers
                
                assert abs(actual_reduction - expected_reduction) < 0.1, \
                    f"TP={tp_size}: expected {expected_reduction}x reduction, got {actual_reduction}x"


if __name__ == "__main__":
    pytest.main([__file__])