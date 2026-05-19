# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for TP mapping and transfer plan utilities.

These tests verify that TP mapping produces correct outputs
(source ranks, split handles, desc IDs).
No GPU or NIXL required.
"""

from __future__ import annotations

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    TPMapping,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
    NixlConnectorWorker,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    TPTransferSlice,
)

# ======================================================================
# Test fixtures / helpers
# ======================================================================


def _make_fa_spec(num_kv_heads: int = 4, total_num_kv_heads: int = 8):
    return FullAttentionSpec(
        block_size=16,
        num_kv_heads=num_kv_heads,
        total_num_kv_heads=total_num_kv_heads,
        head_size=128,
        head_size_v=128,
        dtype=torch.float16,
    )


def _compute_mapping(
    tp_rank: int = 0,
    tp_size: int = 1,
    remote_tp_size: int = 1,
    total_num_kv_heads: int = 8,
    group_specs: list | None = None,
) -> TPMapping:
    if group_specs is None:
        num_kv_heads = total_num_kv_heads // tp_size
        group_specs = [_make_fa_spec(num_kv_heads, total_num_kv_heads)]
    return TPMapping(
        slices_per_group=tuple(
            tuple(
                spec.get_tp_transfer_slices(
                    local_tp_rank=tp_rank,
                    local_tp_size=tp_size,
                    remote_tp_size=remote_tp_size,
                )
            )
            for spec in group_specs
        ),
    )


# ======================================================================
# TP mapping structure tests
# ======================================================================


class TestTPMappingStructure:
    def test_source_ranks_homogeneous(self):
        m = _compute_mapping(tp_size=2, tp_rank=1, remote_tp_size=2)
        assert m.all_source_ranks == (1,)

    def test_source_ranks_d_gt_p(self):
        m = _compute_mapping(tp_size=4, tp_rank=2, remote_tp_size=2)
        assert m.all_source_ranks == (1,)

    def test_source_ranks_p_gt_d(self):
        m = _compute_mapping(tp_size=1, tp_rank=0, remote_tp_size=2)
        assert m.all_source_ranks == (0, 1)

    def test_per_group_slices(self):
        m = _compute_mapping(tp_size=2, tp_rank=0, remote_tp_size=4)
        fa_slices = m.slices_per_group[0]
        assert len(fa_slices) == 2
        assert fa_slices[0].remote_rank == 0
        assert fa_slices[1].remote_rank == 1

    def test_has_rank_in_group(self):
        m = _compute_mapping(tp_size=1, tp_rank=0, remote_tp_size=2)
        assert m.has_rank_in_group(0, 0)
        assert m.has_rank_in_group(0, 1)
        assert not m.has_rank_in_group(0, 2)

    def test_gqa_dedup_load_balanced(self):
        """With total_heads=2, remote_tp=4: picks aligned remote ranks."""
        m0 = _compute_mapping(
            tp_size=2, tp_rank=0, remote_tp_size=4, total_num_kv_heads=2
        )
        m1 = _compute_mapping(
            tp_size=2, tp_rank=1, remote_tp_size=4, total_num_kv_heads=2
        )
        # Rank 0 should read from remote 0, rank 1 from remote 2
        assert m0.slices_per_group[0][0].remote_rank == 0
        assert m1.slices_per_group[0][0].remote_rank == 2


# ======================================================================
# Split handle tests
# ======================================================================


def _make_mock_worker_for_splits(group_spec_types, total_num_kv_heads=8):
    """Build a mock NixlConnectorWorker with _group_spec_types for split tests."""
    worker = object.__new__(NixlConnectorWorker)
    worker._group_spec_types = group_spec_types

    class _FakeTopo:
        def __init__(self, total_num_kv_heads: int):
            self.total_num_kv_heads = total_num_kv_heads

    worker.transfer_topo = _FakeTopo(total_num_kv_heads)
    return worker


class TestBuildSrcSplitHandles:
    @pytest.mark.parametrize("remote_tp_size", [2, 4])
    def test_build_src_split_handles(self, remote_tp_size):
        tp_rank = 0
        tp_size = 1

        plan = _compute_mapping(
            tp_rank=tp_rank,
            tp_size=tp_size,
            remote_tp_size=remote_tp_size,
        )

        worker = _make_mock_worker_for_splits(
            (FullAttentionSpec,), total_num_kv_heads=8
        )
        src_blocks_data = [(0x2000 + i * 1024, 1024, 0) for i in range(8)]
        num_descs = len(src_blocks_data)
        splits = list(
            worker._build_local_splits_from_plan(
                plan,
                src_blocks_data,
                num_descs,
                remote_tp_size,
            )
        )

        assert len(splits) == remote_tp_size
        for handle in splits:
            assert len(handle) == len(src_blocks_data)
            for _, length, _ in handle:
                assert length == 1024 // remote_tp_size


class TestMambaPlanSplitHandles:
    """Verify split handles for Mamba with FA/SSM distinction."""

    def test_fa_and_ssm_different_split_factors(self):
        """Section 0 split by num_attn_reads, section 1 by abs_tp."""
        # Simulate: local_tp=1, remote_tp=2
        # FA: 1 slice (reads from remote 0 only, head range 0:4)
        # Mamba: 2 slices (reads from remote 0 and 1)
        plan = TPMapping(
            slices_per_group=(
                (
                    TPTransferSlice(
                        remote_rank=0,
                        global_range=range(0, 4),
                        local_range=range(0, 4),
                        remote_range=range(0, 4),
                    ),
                ),
                (
                    TPTransferSlice(
                        remote_rank=0,
                        global_range=range(0, 1),
                        local_range=range(0, 1),
                        remote_range=range(0, 1),
                    ),
                    TPTransferSlice(
                        remote_rank=1,
                        global_range=range(0, 1),
                        local_range=range(0, 1),
                        remote_range=range(0, 1),
                    ),
                ),
            ),
        )

        worker = _make_mock_worker_for_splits(
            (FullAttentionSpec, MambaSpec), total_num_kv_heads=4
        )
        # 2 FA descs + 1 SSM desc
        src_blocks_data = [
            (1000, 200, 0),  # FA desc 0
            (2000, 200, 0),  # FA desc 1
            (3000, 400, 0),  # SSM desc 0
        ]

        splits = list(worker._build_local_splits_from_plan(plan, src_blocks_data, 2, 2))

        assert len(splits) == 2  # 2 source ranks

        # Rank 0 (FA source, p_idx=0):
        # FA: chunk=200//1=200, slot=0 -> (1000, 200, 0), (2000, 200, 0)
        # SSM: chunk=400//2=200, idx=0 -> (3000, 200, 0)
        assert splits[0] == [(1000, 200, 0), (2000, 200, 0), (3000, 200, 0)]

        # Rank 1 (not FA source, p_idx=1):
        # FA: chunk=200//1=200, slot=0 -> (1000, 200, 0), (2000, 200, 0)
        # SSM: chunk=400//2=200, idx=1 -> (3200, 200, 0)
        assert splits[1] == [(1000, 200, 0), (2000, 200, 0), (3200, 200, 0)]
