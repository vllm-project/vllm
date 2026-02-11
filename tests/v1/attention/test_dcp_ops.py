# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for DCP (Decode Context Parallelism) operations."""

from unittest.mock import patch

import pytest
import torch


class MockGroupCoordinator:
    """Mock GroupCoordinator for testing DCP functions without distributed setup."""

    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank_in_group = rank

    def all_gather(self, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return tensor.repeat_interleave(self.world_size, dim=dim)

    def reduce_scatter(self, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        size = tensor.size(dim) // self.world_size
        start = self.rank_in_group * size
        return torch.narrow(tensor, dim, start, size)

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


def _patch_groups(tp_ws, tp_rank, dcp_ws, dcp_rank, pcp_ws=1, pcp_rank=0):
    """Context manager that patches TP, DCP and PCP group getters."""
    tp = MockGroupCoordinator(tp_ws, tp_rank)
    dcp = MockGroupCoordinator(dcp_ws, dcp_rank)
    pcp = MockGroupCoordinator(pcp_ws, pcp_rank)
    return (
        (
            patch("vllm.v1.attention.ops.common.get_tp_group", return_value=tp),
            patch("vllm.v1.attention.ops.common.get_dcp_group", return_value=dcp),
            patch("vllm.v1.attention.ops.common.get_pcp_group", return_value=pcp),
        ),
        tp,
        dcp,
    )


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def mock_groups():
    """PCP=1, TP=DCP=2 mock groups."""
    patches, tp, dcp = _patch_groups(tp_ws=2, tp_rank=0, dcp_ws=2, dcp_rank=0)
    with patches[0], patches[1], patches[2]:
        yield tp, dcp


class TestDCPPrepareQuery:
    def test_basic_shape(self, device, mock_groups):
        from vllm.v1.attention.ops.common import dcp_prepare_query

        tp_group, _ = mock_groups
        B, H_local, D = 2, 4, 64
        query = torch.randn(B, H_local, D, device=device)
        result = dcp_prepare_query(query)
        assert result.shape == (B, H_local * tp_group.world_size, D)

    def test_single_rank_passthrough(self, device):
        from vllm.v1.attention.ops.common import dcp_prepare_query

        patches, _, _ = _patch_groups(1, 0, 1, 0)
        with patches[0], patches[1], patches[2]:
            B, H, D = 2, 8, 64
            query = torch.randn(B, H, D, device=device)
            result = dcp_prepare_query(query)
            assert result.shape == query.shape
            torch.testing.assert_close(result, query)


class TestDCPReduceOutput:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_basic_shape(self, device, mock_groups):
        from vllm.v1.attention.ops.common import dcp_reduce_output

        tp_group, _ = mock_groups
        B, H_total, D = 2, 8, 64
        attn_output = torch.randn(B, H_total, D, device=device)
        attn_lse = torch.randn(B, H_total, device=device)
        result = dcp_reduce_output(attn_output, attn_lse)
        assert result.shape == (B, H_total // tp_group.world_size, D)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_single_rank_both_groups(self, device):
        from vllm.v1.attention.ops.common import dcp_reduce_output

        patches, _, _ = _patch_groups(1, 0, 1, 0)
        with patches[0], patches[1], patches[2]:
            B, H, D = 2, 8, 64
            attn_output = torch.randn(B, H, D, device=device)
            attn_lse = torch.randn(B, H, device=device)
            result = dcp_reduce_output(attn_output, attn_lse)
            assert result.shape == attn_output.shape


class TestEndToEndDCPFlow:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_prepare_and_reduce_round_trip(self, device, mock_groups):
        from vllm.v1.attention.ops.common import dcp_prepare_query, dcp_reduce_output

        tp_group, _ = mock_groups
        B, H_local, D = 4, 4, 64
        query = torch.randn(B, H_local, D, device=device)
        query_all_heads = dcp_prepare_query(query)
        assert query_all_heads.shape == (B, H_local * tp_group.world_size, D)

        attn_output = torch.randn_like(query_all_heads)
        attn_lse = torch.randn(B, H_local * tp_group.world_size, device=device)
        final_output = dcp_reduce_output(attn_output, attn_lse)
        assert final_output.shape == query.shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_pcp_gt_1_uses_allreduce_path(self, device):
        """TP=4, DCP=2, PCP=2: all-reduce + slice path."""
        from vllm.v1.attention.ops.common import dcp_prepare_query, dcp_reduce_output

        patches, tp, _ = _patch_groups(
            tp_ws=4, tp_rank=0, dcp_ws=2, dcp_rank=0, pcp_ws=2, pcp_rank=0
        )
        with patches[0], patches[1], patches[2]:
            B, H_local, D = 2, 2, 64
            query = torch.randn(B, H_local, D, device=device)
            query_all_heads = dcp_prepare_query(query)
            assert query_all_heads.shape == (B, H_local * tp.world_size, D)

            attn_output = torch.randn_like(query_all_heads)
            attn_lse = torch.randn(B, H_local * tp.world_size, device=device)
            final_output = dcp_reduce_output(attn_output, attn_lse)
            assert final_output.shape == query.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
