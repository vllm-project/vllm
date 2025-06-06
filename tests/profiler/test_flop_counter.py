# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.profiler.flop_counter import (DetailedFlopCount, FlopContextManager,
                                       FlopCounter, format_flops)


class TestFlopCounter:
    
    def test_flop_counter_basic(self):
        """Test basic FLOP counter functionality."""
        counter = FlopCounter()
        
        # Test initial state
        assert counter.get_total_flops() == 0
        assert len(counter.get_flop_breakdown()) == 0
        
        with counter:
            a = torch.randn(10, 10)
            b = torch.randn(10, 10)
            _ = torch.mm(a, b)
        total_flops = counter.get_total_flops()
        assert total_flops > 0
        
        breakdown = counter.get_flop_breakdown()
        assert len(breakdown) > 0
        assert any('mm' in op for op in breakdown.keys())
    
    def test_flop_context_manager(self):
        """Test FlopContextManager functionality."""
        with FlopContextManager() as counter:
            a = torch.randn(5, 5)
            b = torch.randn(5, 5)
            _ = torch.mm(a, b)
        
        total_flops = counter.get_total_flops()
        assert total_flops > 0
        
        detailed = counter.get_detailed_counts()
        assert isinstance(detailed, DetailedFlopCount)
        assert detailed.total_flops > 0
    
    def test_matrix_multiplication_flops(self):
        """Test FLOP counting for matrix operations."""
        counter = FlopCounter()
        
        with counter:
            a = torch.randn(4, 6)
            b = torch.randn(6, 8)
            _ = torch.mm(a, b)
        
        breakdown = counter.get_flop_breakdown()
        mm_flops = sum(flops for op, flops in breakdown.items() if 'mm' in op)
        
        assert mm_flops >= 300
        assert mm_flops <= 500
    
    def test_batch_matrix_multiplication_flops(self):
        """Test FLOP counting for batch matrix operations."""
        counter = FlopCounter()
        
        with counter:
            a = torch.randn(3, 4, 5)
            b = torch.randn(3, 5, 6)
            _ = torch.bmm(a, b)
        
        breakdown = counter.get_flop_breakdown()
        bmm_flops = sum(flops for op, flops in breakdown.items() if 'bmm' in op)
        
        assert bmm_flops >= 600
        assert bmm_flops <= 800
    
    def test_softmax_flops(self):
        """Test FLOP counting for softmax operations."""
        counter = FlopCounter()
        
        with counter:
            a = torch.randn(10, 10)
            _ = torch.softmax(a, dim=-1)
        
        breakdown = counter.get_flop_breakdown()
        softmax_flops = sum(flops for op, flops in breakdown.items() 
                           if 'softmax' in op)
        assert softmax_flops > 0
    
    def test_reset_functionality(self):
        """Test counter reset functionality."""
        counter = FlopCounter()
        
        with counter:
            a = torch.randn(5, 5)
            b = torch.randn(5, 5)
            _ = torch.mm(a, b)
        
        assert counter.get_total_flops() > 0
        
        counter.reset()
        assert counter.get_total_flops() == 0
        assert len(counter.get_flop_breakdown()) == 0
    
    def test_detailed_flop_count(self):
        """Test DetailedFlopCount functionality."""
        flop_count = DetailedFlopCount()
        
        flop_count.add_operation("aten::mm", 100, "layer1")
        flop_count.add_operation("aten::softmax", 50, "layer1")
        flop_count.add_operation("aten::mm", 200, "layer2")
        
        assert flop_count.total_flops == 350
        assert len(flop_count.operation_counts) == 2
        assert flop_count.operation_counts["aten::mm"] == 300
        assert flop_count.operation_counts["aten::softmax"] == 50
        assert len(flop_count.layer_counts) == 2
    
    def test_format_flops(self):
        """Test FLOP formatting functionality."""
        assert format_flops(500) == "500 FLOPs"
        assert format_flops(1500) == "1.50 KFLOPs"
        assert format_flops(1500000) == "1.50 MFLOPs"
        assert format_flops(1500000000) == "1.50 GFLOPs"
        assert format_flops(1500000000000) == "1.50 TFLOPs"


class TestFlopCountOperations:
    
    def test_flop_count_addition(self):
        """Test FlopCount addition operations."""
        from vllm.profiler.flop_counter import FlopCount
        
        count1 = FlopCount(mm=100, softmax=50)
        count2 = FlopCount(mm=200, gelu=75)
        
        # Test addition
        result = count1 + count2
        assert result.mm == 300
        assert result.softmax == 50
        assert result.gelu == 75
        assert result.total() == 425
        
        # Test in-place addition
        count1 += count2
        assert count1.mm == 300
        assert count1.softmax == 50
        assert count1.gelu == 75
        assert count1.total() == 425
    
    def test_flop_count_to_dict(self):
        """Test FlopCount dictionary conversion."""
        from vllm.profiler.flop_counter import FlopCount
        
        count = FlopCount(mm=100, softmax=50, gelu=25)
        result_dict = count.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['mm'] == 100
        assert result_dict['softmax'] == 50
        assert result_dict['gelu'] == 25
        assert 'total' not in result_dict


# Integration tests
class TestFlopCounterIntegration:
    
    @pytest.mark.skipif(not torch.cuda.is_available(), 
                       reason="CUDA not available")
    def test_cuda_operations(self):
        """Test FLOP counting with CUDA operations."""
        counter = FlopCounter()
        
        with counter:
            a = torch.randn(10, 10, device='cuda')
            b = torch.randn(10, 10, device='cuda')
            _ = torch.mm(a, b)
        
        assert counter.get_total_flops() > 0
    
    def test_attention_like_operations(self):
        """Test FLOP counting for attention-like operations."""
        counter = FlopCounter()
        
        with counter:
            seq_len, d_model = 128, 512
            batch_size = 2
            
            q = torch.randn(batch_size, seq_len, d_model)
            k = torch.randn(batch_size, seq_len, d_model)
            v = torch.randn(batch_size, seq_len, d_model)
            
            scores = torch.bmm(q, k.transpose(-2, -1))
            attn_weights = torch.softmax(scores, dim=-1)
            _ = torch.bmm(attn_weights, v)
        
        total_flops = counter.get_total_flops()
        assert total_flops > 0
        
        breakdown = counter.get_flop_breakdown()
        assert any('bmm' in op for op in breakdown.keys())
        assert any('softmax' in op for op in breakdown.keys())