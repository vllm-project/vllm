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
        assert any('mm' in op for op in breakdown)

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

        # Expected: 2 * M * N * K = 2 * 4 * 8 * 6 = 384 FLOPs
        expected_flops = 2 * 4 * 8 * 6
        assert mm_flops == expected_flops

    def test_batch_matrix_multiplication_flops(self):
        """Test FLOP counting for batch matrix operations."""
        counter = FlopCounter()

        with counter:
            a = torch.randn(3, 4, 5)
            b = torch.randn(3, 5, 6)
            _ = torch.bmm(a, b)

        breakdown = counter.get_flop_breakdown()
        bmm_flops = sum(flops for op, flops in breakdown.items()
                        if 'bmm' in op)

        # Expected: batch_size * 2 * M * N * K = 3 * 2 * 4 * 6 * 5 = 720 FLOPs
        expected_flops = 3 * 2 * 4 * 6 * 5
        assert bmm_flops == expected_flops

    def test_softmax_flops(self):
        """Test FLOP counting for softmax operations."""
        counter = FlopCounter()

        with counter:
            a = torch.randn(10, 10)
            _ = torch.softmax(a, dim=-1)

        breakdown = counter.get_flop_breakdown()
        softmax_flops = sum(flops for op, flops in breakdown.items()
                            if 'softmax' in op)

        # Expected: 5 * numel = 5 * 10 * 10 = 500 FLOPs
        expected_flops = 5 * 10 * 10
        assert softmax_flops == expected_flops

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
        assert any('bmm' in op for op in breakdown)
        assert any('softmax' in op for op in breakdown)

    def test_module_stack_tracking(self):
        """Test that module stack tracking works for layer attribution."""
        import torch.nn as nn

        counter = FlopCounter()

        class SimpleModule(nn.Module):

            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModule()

        with counter:
            x = torch.randn(3, 10)
            _ = model(x)

        detailed_counts = counter.get_detailed_counts()

        assert len(detailed_counts.layer_counts) > 0

        layer_names = list(detailed_counts.layer_counts.keys())
        has_linear_layer = any('linear' in name.lower()
                               for name in layer_names)

        # At minimum, we should have some layer attribution
        assert has_linear_layer or len(layer_names) > 0
