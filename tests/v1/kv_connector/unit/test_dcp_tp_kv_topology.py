# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for DcpTpKVTopology.get_target_remote_ranks.

Test cases derived from the sharding examples in:
  private-vllm-neuron/examples/nxdi/vllm/disaggregated_inference/
  context_parallel/cp_di_strategy.md

Each example defines a (P: TP, DCP) and (D: TP, DCP) configuration with
KV_heads=2, S=64, block_size=16. The tests verify that each decode global
rank maps to the correct prefill global rank(s).
"""

import pytest
from unittest.mock import MagicMock

from vllm.distributed.kv_transfer.kv_connector.utils import DcpTpKVTopology


def _mock_attn_backend():
    """Create a mock attention backend that satisfies TpKVTopology.__post_init__."""
    backend = MagicMock()
    # Return a 5-dim shape with num_blocks as first dim (blocks-first layout)
    # Shape: [num_blocks, 2(kv), num_kv_heads, block_size, head_size]
    backend.get_kv_cache_shape.return_value = (1, 2, 1, 16, 1)
    return backend


def _make_topo(
    local_tp_rank: int,
    local_tp_size: int,
    local_dcp_rank: int,
    local_dcp_size: int,
    engine_id: str = "local",
) -> DcpTpKVTopology:
    """Helper to construct a DcpTpKVTopology for a specific decode rank."""
    return DcpTpKVTopology(
        tp_rank=local_tp_rank,
        engine_id=engine_id,
        remote_tp_size={engine_id: local_tp_size},
        remote_block_size={engine_id: 16},
        is_mla=False,
        total_num_kv_heads=2,
        attn_backends=[_mock_attn_backend()],
        dcp_rank=local_dcp_rank,
        dcp_size=local_dcp_size,
    )


class TestExample1:
    """Example 1: P: TP=2 DCP=2 (4 ranks), D: TP=4 DCP=2 (4 ranks).
    1:1 rank mapping.
    """
    P_TP, P_DCP = 2, 2
    D_TP, D_DCP = 4, 2

    @pytest.mark.parametrize(
        "d_tp, d_dcp, expected",
        [
            (0, 0, [0]),  # D0 → P0
            (1, 1, [1]),  # D1 → P1
            (2, 0, [2]),  # D2 → P2
            (3, 1, [3]),  # D3 → P3
        ],
        ids=["D0→P0", "D1→P1", "D2→P2", "D3→P3"],
    )
    def test_mapping(self, d_tp, d_dcp, expected):
        topo = _make_topo(d_tp, self.D_TP, d_dcp, self.D_DCP)
        assert topo.get_target_remote_ranks(self.P_TP, self.P_DCP) == expected


class TestExample2:
    """Example 2: P: TP=2 DCP=4 (8 ranks), D: TP=4 DCP=2 (4 ranks).
    P DCP > D DCP → decode aggregates from multiple prefill DCP ranks.
    """
    P_TP, P_DCP = 2, 4
    D_TP, D_DCP = 4, 2

    @pytest.mark.parametrize(
        "d_tp, d_dcp, expected",
        [
            (0, 0, [0, 2]),  # D0 → P0 + P2
            (1, 1, [1, 3]),  # D1 → P1 + P3
            (2, 0, [4, 6]),  # D2 → P4 + P6
            (3, 1, [5, 7]),  # D3 → P5 + P7
        ],
        ids=["D0→P0,P2", "D1→P1,P3", "D2→P4,P6", "D3→P5,P7"],
    )
    def test_mapping(self, d_tp, d_dcp, expected):
        topo = _make_topo(d_tp, self.D_TP, d_dcp, self.D_DCP)
        assert topo.get_target_remote_ranks(self.P_TP, self.P_DCP) == expected


class TestExample3:
    """Example 3: P: TP=2 DCP=2 (4 ranks), D: TP=8 DCP=4 (8 ranks).
    D DCP > P DCP → decode splits, multiple D ranks read from same P rank.
    """
    P_TP, P_DCP = 2, 2
    D_TP, D_DCP = 8, 4

    @pytest.mark.parametrize(
        "d_tp, d_dcp, expected",
        [
            (0, 0, [0]),  # D0 → P0 (KV_H:0, T:0-15)
            (1, 1, [1]),  # D1 → P1 (KV_H:0, T:16-31)
            (2, 2, [0]),  # D2 → P0 (KV_H:0, T:32-47, different block in P0)
            (3, 3, [1]),  # D3 → P1 (KV_H:0, T:48-63, different block in P1)
            (4, 0, [2]),  # D4 → P2 (KV_H:1, T:0-15)
            (5, 1, [3]),  # D5 → P3 (KV_H:1, T:16-31)
            (6, 2, [2]),  # D6 → P2 (KV_H:1, T:32-47, different block in P2)
            (7, 3, [3]),  # D7 → P3 (KV_H:1, T:48-63, different block in P3)
        ],
        ids=["D0→P0", "D1→P1", "D2→P0", "D3→P1",
             "D4→P2", "D5→P3", "D6→P2", "D7→P3"],
    )
    def test_mapping(self, d_tp, d_dcp, expected):
        topo = _make_topo(d_tp, self.D_TP, d_dcp, self.D_DCP)
        assert topo.get_target_remote_ranks(self.P_TP, self.P_DCP) == expected


class TestExample4:
    """Example 4: P: TP=1 DCP=4 (4 ranks), D: TP=4 DCP=2 (4 ranks).
    Heterogeneous TP + DCP with head splitting.
    """
    P_TP, P_DCP = 1, 4
    D_TP, D_DCP = 4, 2

    @pytest.mark.parametrize(
        "d_tp, d_dcp, expected",
        [
            (0, 0, [0, 2]),  # D0 → P0 + P2
            (1, 1, [1, 3]),  # D1 → P1 + P3
            (2, 0, [0, 2]),  # D2 → P0 + P2 (same P ranks, different KV head)
            (3, 1, [1, 3]),  # D3 → P1 + P3
        ],
        ids=["D0→P0,P2", "D1→P1,P3", "D2→P0,P2", "D3→P1,P3"],
    )
    def test_mapping(self, d_tp, d_dcp, expected):
        topo = _make_topo(d_tp, self.D_TP, d_dcp, self.D_DCP)
        assert topo.get_target_remote_ranks(self.P_TP, self.P_DCP) == expected


class TestNoDcp:
    """DCP=1 on both sides should behave identically to TpKVTopology."""

    def test_homogeneous_tp(self):
        topo = _make_topo(0, 2, 0, 1)
        assert topo.get_target_remote_ranks(2, 1) == [0]
        topo = _make_topo(1, 2, 0, 1)
        assert topo.get_target_remote_ranks(2, 1) == [1]

    def test_hetero_tp_d_gt_p(self):
        topo = _make_topo(0, 4, 0, 1)
        assert topo.get_target_remote_ranks(2, 1) == [0]
        topo = _make_topo(3, 4, 0, 1)
        assert topo.get_target_remote_ranks(2, 1) == [1]

    def test_hetero_tp_p_gt_d(self):
        topo = _make_topo(0, 2, 0, 1)
        assert topo.get_target_remote_ranks(4, 1) == [0, 1]

    def test_default_dcp_size_omitted(self):
        """Calling without remote_dcp_size should default to 1."""
        topo = _make_topo(0, 2, 0, 1)
        assert topo.get_target_remote_ranks(2) == [0]
