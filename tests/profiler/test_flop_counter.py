# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.profiler.flop_counter import (DetailedFlopCount, FlopContextManager,
                                        format_flops)


class TestFlopCounter:

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
        with FlopContextManager() as counter:
            a = torch.randn(4, 6)
            b = torch.randn(6, 8)
            _ = torch.mm(a, b)

        breakdown = counter.get_flop_breakdown()
        assert isinstance(breakdown, dict)
        assert "mm_flops" in breakdown
        assert breakdown["mm_flops"] > 0

        total_flops = counter.get_total_flops()
        # Expected: 2 * M * N * K = 2 * 4 * 8 * 6 = 384 FLOPs
        assert total_flops == 384

    def test_batch_matrix_multiplication_flops(self):
        """Test FLOP counting for batch matrix operations."""
        with FlopContextManager() as counter:
            a = torch.randn(3, 4, 5)
            b = torch.randn(3, 5, 6)
            _ = torch.bmm(a, b)

        breakdown = counter.get_flop_breakdown()
        assert isinstance(breakdown, dict)
        assert "mm_flops" in breakdown
        assert breakdown["mm_flops"] > 0

        total_flops = counter.get_total_flops()
        # Expected: batch_size * 2 * M * N * K = 3 * 2 * 4 * 6 * 5 = 720 FLOPs
        assert total_flops == 720

    @pytest.mark.xfail(not torch.cuda.is_available(), )
    def test_softmax_flops(self):
        """Test FLOP counting for softmax operations."""
        with FlopContextManager() as counter:
            a = torch.randn(10, 10)
            _ = torch.softmax(a, dim=-1)

        breakdown = counter.get_flop_breakdown()
        assert isinstance(breakdown, dict)

        total_flops = counter.get_total_flops()
        assert total_flops > 0

    @pytest.mark.xfail(not torch.cuda.is_available(), )
    def test_reset_functionality(self):
        """Test counter reset functionality."""
        with FlopContextManager() as counter:
            a = torch.randn(5, 5)
            b = torch.randn(5, 5)
            _ = torch.mm(a, b)

            assert counter.get_total_flops() > 0

            counter.reset()
            assert counter.get_total_flops() == 0
            breakdown = counter.get_flop_breakdown()
            assert all(flops == 0 for flops in breakdown.values())

    def test_detailed_flop_count(self):
        """Test DetailedFlopCount functionality."""
        flop_count = DetailedFlopCount()

        flop_count.add_operation("aten::mm", 100)
        flop_count.add_operation("aten::softmax", 50)
        flop_count.add_operation("aten::mm", 200)

        assert flop_count.total_flops == 350
        assert len(flop_count.operation_counts) == 2
        assert flop_count.operation_counts["aten::mm"] == 300
        assert flop_count.operation_counts["aten::softmax"] == 50

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

        count1 = FlopCount(total_flops=150,
                           flop_counts={
                               "mm": 100,
                               "softmax": 50
                           })
        count2 = FlopCount(total_flops=275,
                           flop_counts={
                               "mm": 200,
                               "gelu": 75
                           })

        # Test addition
        result = count1 + count2
        assert result.total_flops == 425
        assert result.flop_counts["mm"] == 300
        assert result.flop_counts["softmax"] == 50
        assert result.flop_counts["gelu"] == 75

        # Test in-place addition
        count1 += count2
        assert count1.total_flops == 425
        assert count1.flop_counts["mm"] == 300
        assert count1.flop_counts["softmax"] == 50
        assert count1.flop_counts["gelu"] == 75

    def test_flop_count_to_dict(self):
        """Test FlopCount dictionary conversion."""
        from vllm.profiler.flop_counter import FlopCount

        count = FlopCount(total_flops=175,
                          flop_counts={
                              "mm": 100,
                              "softmax": 50,
                              "gelu": 25
                          })
        result_dict = count.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["total_flops"] == 175
        assert result_dict["mm"] == 100
        assert result_dict["softmax"] == 50
        assert result_dict["gelu"] == 25


# Integration tests
class TestFlopCounterIntegration:

    @pytest.mark.skipif(not torch.cuda.is_available(),
                        reason="CUDA not available")
    def test_cuda_operations(self):
        """Test FLOP counting with CUDA operations."""
        with FlopContextManager() as counter:
            a = torch.randn(10, 10, device="cuda")
            b = torch.randn(10, 10, device="cuda")
            _ = torch.mm(a, b)

        assert counter.get_total_flops() > 0

    def test_attention_like_operations(self):
        """Test FLOP counting for attention-like operations."""
        with FlopContextManager() as counter:
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
        assert isinstance(breakdown, dict)
        assert any(flops > 0 for flops in breakdown.values())

    def test_nested_context_manager(self):
        """Test nested context manager behavior."""
        with FlopContextManager() as outer_counter:
            a = torch.randn(5, 5)
            b = torch.randn(5, 5)
            _ = torch.mm(a, b)

            outer_flops = outer_counter.get_total_flops()

            with FlopContextManager() as inner_counter:
                c = torch.randn(3, 3)
                d = torch.randn(3, 3)
                _ = torch.mm(c, d)

                inner_flops = inner_counter.get_total_flops()
                assert inner_flops > 0

            # Outer counter should still work
            final_outer_flops = outer_counter.get_total_flops()
            assert final_outer_flops >= outer_flops

    def test_multiple_operations(self):
        """Test FLOP counting with multiple different operations."""
        with FlopContextManager() as counter:
            # Matrix multiplication
            a = torch.randn(10, 20)
            b = torch.randn(20, 15)
            c = torch.mm(a, b)

            # Activation function
            d = torch.relu(c)

            # Softmax
            e = torch.softmax(d, dim=-1)

            # Another matrix multiplication
            f = torch.randn(15, 5)
            _ = torch.mm(e, f)

        total_flops = counter.get_total_flops()
        assert total_flops > 0

        breakdown = counter.get_flop_breakdown()
        assert isinstance(breakdown, dict)
        assert any(flops > 0 for flops in breakdown.values())

        detailed = counter.get_detailed_counts()
        assert detailed.total_flops == total_flops
        assert len(detailed.operation_counts) > 0
        assert detailed.mm_flops >= 0
        assert detailed.attention_flops >= 0
        assert detailed.activation_flops >= 0
        assert detailed.normalization_flops >= 0
