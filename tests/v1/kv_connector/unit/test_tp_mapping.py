# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for TP mapping and transfer plan utilities.

These tests verify that TP mapping produces correct outputs
(source ranks, split handles, desc IDs).
No GPU or NIXL required.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    TPMapping,
    compute_tp_mapping,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
    NixlConnectorWorker,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

# ======================================================================
# Test fixtures / helpers
# ======================================================================


def _compute_mapping(
    tp_rank: int = 0,
    tp_size: int = 1,
    remote_tp_size: int = 1,
    is_mla: bool = False,
    num_kv_heads: int = 8,
    group_spec_types: tuple[type, ...] = (FullAttentionSpec,),
) -> TPMapping:
    transfer_topology = SimpleNamespace(
        tp_rank=tp_rank,
        tp_size=tp_size,
        is_mla=is_mla,
        total_num_kv_heads=num_kv_heads,
    )
    return compute_tp_mapping(
        transfer_topology=transfer_topology,
        remote_tp_size=remote_tp_size,
        group_spec_types=group_spec_types,
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


# ======================================================================
# Split handle tests
# ======================================================================


def _make_mock_worker_for_splits(group_spec_types):
    """Build a mock NixlConnectorWorker with _group_spec_types for split tests."""
    worker = object.__new__(NixlConnectorWorker)
    worker._group_spec_types = group_spec_types
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

        worker = _make_mock_worker_for_splits((FullAttentionSpec,))
        src_blocks_data = [(0x2000 + i * 1024, 1024, 0) for i in range(8)]
        num_descs = len(src_blocks_data)
        splits = list(
            worker._build_local_splits_from_plan(
                plan,
                src_blocks_data,
                num_descs,
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
        fa_readers = (0,)
        ssm_readers = (0, 1)
        plan = TPMapping(
            source_ranks_per_group=(fa_readers, ssm_readers),
            all_source_ranks=(0, 1),
            rank_to_attention_slot={0: 0, 1: 0},
            rank_offset_factor=0,
        )

        worker = _make_mock_worker_for_splits((FullAttentionSpec, MambaSpec))
        # 2 FA descs + 1 SSM desc
        src_blocks_data = [
            (1000, 200, 0),  # FA desc 0
            (2000, 200, 0),  # FA desc 1
            (3000, 400, 0),  # SSM desc 0
        ]

        splits = list(worker._build_local_splits_from_plan(plan, src_blocks_data, 2))

        assert len(splits) == 2  # 2 source ranks

        # Rank 0 (FA source, p_idx=0):
        # FA: chunk=200//1=200, slot=0 → (1000, 200, 0), (2000, 200, 0)
        # SSM: chunk=400//2=200, idx=0 → (3000, 200, 0)
        assert splits[0] == [(1000, 200, 0), (2000, 200, 0), (3000, 200, 0)]

        # Rank 1 (not FA source, p_idx=1):
        # FA: chunk=200//1=200, slot=0 (skip_fa) → (1000, 200, 0), (2000, 200, 0)
        # SSM: chunk=400//2=200, idx=1 → (3200, 200, 0)
        assert splits[1] == [(1000, 200, 0), (2000, 200, 0), (3200, 200, 0)]
