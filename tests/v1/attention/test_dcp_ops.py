# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for DCP (Decode Context Parallelism) operations."""

from unittest.mock import patch

import pytest
import torch

from vllm.v1.attention.ops.common import (
    cp_lse_ag_out_ar,
)


class MockGroupCoordinator:
    """Mock GroupCoordinator for testing DCP functions without distributed setup."""

    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank_in_group = rank

    def all_gather(self, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Simulate all-gather by repeating the tensor."""
        return tensor.repeat_interleave(self.world_size, dim=dim)

    def reduce_scatter(self, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Simulate reduce-scatter by taking a slice."""
        size = tensor.size(dim) // self.world_size
        start = self.rank_in_group * size
        return torch.narrow(tensor, dim, start, size)

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """Simulate all-reduce (identity for single process mock)."""
        return tensor


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def mock_groups():
    """Create mock TP and DCP groups and patch the getters."""
    tp_group = MockGroupCoordinator(world_size=2, rank=0)
    dcp_group = MockGroupCoordinator(world_size=2, rank=0)

    with (
        patch("vllm.v1.attention.ops.common.get_tp_group", return_value=tp_group),
        patch("vllm.v1.attention.ops.common.get_dcp_group", return_value=dcp_group),
    ):
        yield tp_group, dcp_group


class TestDCPPrepareQuery:
    """Tests for dcp_prepare_query function."""

    def test_basic_shape(self, device, mock_groups):
        """Test that all-gather produces correct output shape."""
        from vllm.v1.attention.ops.common import dcp_prepare_query

        tp_group, _ = mock_groups
        B, H_local, D = 2, 4, 64

        query = torch.randn(B, H_local, D, device=device)
        result = dcp_prepare_query(query)

        expected_shape = (B, H_local * tp_group.world_size, D)
        assert result.shape == expected_shape, (
            f"Expected {expected_shape}, got {result.shape}"
        )

    def test_single_rank_passthrough(self, device):
        """Test that world_size=1 returns input unchanged."""
        from vllm.v1.attention.ops.common import dcp_prepare_query

        tp_group = MockGroupCoordinator(world_size=1, rank=0)
        dcp_group = MockGroupCoordinator(world_size=1, rank=0)

        with (
            patch("vllm.v1.attention.ops.common.get_tp_group", return_value=tp_group),
            patch("vllm.v1.attention.ops.common.get_dcp_group", return_value=dcp_group),
        ):
            B, H, D = 2, 8, 64
            query = torch.randn(B, H, D, device=device)
            result = dcp_prepare_query(query)

            assert result.shape == query.shape
            torch.testing.assert_close(result, query)


class TestDCPReduceOutput:
    """Tests for dcp_reduce_output function."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_basic_shape(self, device, mock_groups):
        """Test that reduce produces correct output shape."""
        from vllm.v1.attention.ops.common import dcp_reduce_output

        tp_group, _ = mock_groups
        B, H_total, D = 2, 8, 64

        attn_output = torch.randn(B, H_total, D, device=device)
        attn_lse = torch.randn(B, H_total, device=device)

        result = dcp_reduce_output(attn_output, attn_lse)

        expected_H = H_total // tp_group.world_size
        expected_shape = (B, expected_H, D)
        assert result.shape == expected_shape, (
            f"Expected {expected_shape}, got {result.shape}"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_single_rank_both_groups(self, device):
        """Test with both groups having world_size=1."""
        from vllm.v1.attention.ops.common import dcp_reduce_output

        tp_group = MockGroupCoordinator(world_size=1, rank=0)
        dcp_group = MockGroupCoordinator(world_size=1, rank=0)

        with (
            patch("vllm.v1.attention.ops.common.get_tp_group", return_value=tp_group),
            patch("vllm.v1.attention.ops.common.get_dcp_group", return_value=dcp_group),
        ):
            B, H, D = 2, 8, 64
            attn_output = torch.randn(B, H, D, device=device)
            attn_lse = torch.randn(B, H, device=device)

            result = dcp_reduce_output(attn_output, attn_lse)
            assert result.shape == attn_output.shape


class TestCPLseAgOutAr:
    """Tests for cp_lse_ag_out_ar function (used internally by dcp_reduce_output)."""

    def test_single_rank_passthrough(self, device):
        """Test that world_size=1 returns input unchanged."""
        B, H, D = 2, 8, 64
        attn_output = torch.randn(B, H, D, device=device)
        attn_lse = torch.randn(B, H, device=device)

        cp_group = MockGroupCoordinator(world_size=1, rank=0)

        result = cp_lse_ag_out_ar(attn_output, attn_lse, cp_group)

        assert result.shape == attn_output.shape
        torch.testing.assert_close(result, attn_output)


class TestEndToEndDCPFlow:
    """End-to-end tests simulating the full DCP decode flow."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_prepare_and_reduce_round_trip(self, device, mock_groups):
        """Test that prepare -> attention -> reduce preserves shapes correctly."""
        from vllm.v1.attention.ops.common import dcp_prepare_query, dcp_reduce_output

        tp_group, _ = mock_groups
        B, H_local, D = 4, 4, 64

        query = torch.randn(B, H_local, D, device=device)

        query_all_heads = dcp_prepare_query(query)
        assert query_all_heads.shape == (B, H_local * tp_group.world_size, D)

        attn_output = torch.randn_like(query_all_heads)
        attn_lse = torch.randn(B, H_local * tp_group.world_size, device=device)

        final_output = dcp_reduce_output(attn_output, attn_lse)

        assert final_output.shape == query.shape, (
            f"Round-trip should preserve shape: {query.shape} -> {final_output.shape}"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_different_tp_dcp_sizes(self, device):
        """Test with different TP and DCP world sizes."""
        from vllm.v1.attention.ops.common import dcp_prepare_query, dcp_reduce_output

        tp_group = MockGroupCoordinator(world_size=4, rank=0)
        dcp_group = MockGroupCoordinator(world_size=2, rank=0)

        with (
            patch("vllm.v1.attention.ops.common.get_tp_group", return_value=tp_group),
            patch("vllm.v1.attention.ops.common.get_dcp_group", return_value=dcp_group),
        ):
            B, H_local, D = 2, 2, 64

            query = torch.randn(B, H_local, D, device=device)

            query_all_heads = dcp_prepare_query(query)
            assert query_all_heads.shape == (B, H_local * tp_group.world_size, D)

            attn_output = torch.randn_like(query_all_heads)
            attn_lse = torch.randn(B, H_local * tp_group.world_size, device=device)

            final_output = dcp_reduce_output(attn_output, attn_lse)

            assert final_output.shape == query.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
