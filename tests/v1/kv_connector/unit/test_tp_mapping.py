# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for TP mapping and transfer plan utilities.

These tests verify that TP mapping produces correct outputs
(source ranks, split handles, desc IDs).
No GPU or NIXL required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.nixl.worker import (
    NixlConnectorWorker,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    ShardRange,
    TPTransferSlice,
)

# ======================================================================
# Test fixtures / helpers
# ======================================================================


def _make_fa_spec(num_kv_heads: int = 4):
    return FullAttentionSpec(
        block_size=16,
        num_kv_heads=num_kv_heads,
        head_size=128,
        head_size_v=128,
        dtype=torch.float16,
    )


def _get_slices(
    tp_rank: int = 0,
    tp_size: int = 1,
    remote_tp_size: int = 1,
    total_num_kv_heads: int = 8,
    spec=None,
) -> dict[int, TPTransferSlice]:
    """Call get_tp_transfer_slices on the given spec (or a default FA spec)."""
    if spec is None:
        num_kv_heads = max(1, total_num_kv_heads // tp_size)
        spec = _make_fa_spec(num_kv_heads)
    return spec.get_tp_transfer_slices(
        tp_rank, tp_size, remote_tp_size, total_num_kv_heads
    )


def _remote_ranks_from_slices(
    *group_slices: dict[int, TPTransferSlice],
) -> tuple[int, ...]:
    """Derive deduplicated sorted source ranks from multiple group slices."""
    return tuple(sorted({r for slices in group_slices for r in slices}))


# ======================================================================
# TP mapping structure tests
# ======================================================================


class TestTPMappingStructure:
    def test_remote_ranks_homogeneous(self):
        slices = _get_slices(tp_size=2, tp_rank=1, remote_tp_size=2)
        assert _remote_ranks_from_slices(slices) == (1,)

    def test_remote_ranks_d_gt_p(self):
        slices = _get_slices(tp_size=4, tp_rank=2, remote_tp_size=2)
        assert _remote_ranks_from_slices(slices) == (1,)

    def test_remote_ranks_p_gt_d(self):
        slices = _get_slices(tp_size=1, tp_rank=0, remote_tp_size=2)
        assert _remote_ranks_from_slices(slices) == (0, 1)

    def test_per_group_slices(self):
        slices = _get_slices(tp_size=2, tp_rank=0, remote_tp_size=4)
        assert len(slices) == 2
        assert 0 in slices
        assert 1 in slices

    def test_has_rank_in_group(self):
        slices = _get_slices(tp_size=1, tp_rank=0, remote_tp_size=2)
        assert 0 in slices
        assert 1 in slices
        assert 2 not in slices

    def test_gqa_dedup_load_balanced(self):
        """With total_heads=2, remote_tp=4: picks aligned remote ranks."""
        slices_r0 = _get_slices(
            tp_size=2, tp_rank=0, remote_tp_size=4, total_num_kv_heads=2
        )
        slices_r1 = _get_slices(
            tp_size=2, tp_rank=1, remote_tp_size=4, total_num_kv_heads=2
        )
        assert 0 in slices_r0
        assert 2 in slices_r1


# ======================================================================
# Split handle tests
# ======================================================================


def _make_mock_worker_for_splits(
    group_specs: list,
    tp_mappings: tuple,
    remote_ranks: tuple[int, ...],
    engine_id: str = "remote_0",
):
    """Build a mock NixlConnectorWorker with the fields _build_local_splits needs."""
    worker = object.__new__(NixlConnectorWorker)
    kv_cache_groups = []
    for spec in group_specs:
        group = MagicMock()
        group.kv_cache_spec = spec
        kv_cache_groups.append(group)
    kv_cache_config = MagicMock()
    kv_cache_config.kv_cache_groups = kv_cache_groups
    worker.kv_cache_config = kv_cache_config
    worker.tp_mappings = {engine_id: tp_mappings}
    worker.remote_ranks = {engine_id: remote_ranks}
    worker.transfer_topo = MagicMock()
    return worker


class TestBuildSrcSplitHandles:
    @pytest.mark.parametrize("remote_tp_size", [2, 4])
    def test_split_shape(self, remote_tp_size):
        """Each split has correct number of descs with correct chunk size."""
        tp_rank = 0
        tp_size = 1
        total_num_kv_heads = 8
        engine_id = "remote_0"

        fa_spec = _make_fa_spec(num_kv_heads=total_num_kv_heads // tp_size)
        fa_slices = fa_spec.get_tp_transfer_slices(
            tp_rank, tp_size, remote_tp_size, total_num_kv_heads
        )
        remote_ranks = _remote_ranks_from_slices(fa_slices)

        worker = _make_mock_worker_for_splits(
            group_specs=[fa_spec],
            tp_mappings=(fa_slices,),
            remote_ranks=remote_ranks,
            engine_id=engine_id,
        )
        src_blocks_data = [(0x2000 + i * 1024, 1024, 0) for i in range(8)]
        num_fa_descs = len(src_blocks_data)
        splits = list(
            worker._build_local_splits(engine_id, src_blocks_data, num_fa_descs)
        )

        assert len(splits) == remote_tp_size
        for handle in splits:
            assert len(handle) == len(src_blocks_data)
            for _, length, _ in handle:
                assert length == 1024 // remote_tp_size

    @pytest.mark.parametrize(
        "remote_tp_size,total_num_kv_heads",
        [(2, 4), (2, 8), (4, 8)],
    )
    def test_fa_offsets_p_gt_d(self, remote_tp_size, total_num_kv_heads):
        """Verify concrete FA offsets for multi-head P>D (the previously buggy path).

        With local_tp=1, the full local block covers all heads. Each remote
        rank's slice should land at the correct byte offset proportional to
        its position in the local tensor.
        """
        tp_rank = 0
        tp_size = 1
        engine_id = "remote_0"
        local_block_len = 1024

        fa_spec = _make_fa_spec(num_kv_heads=total_num_kv_heads // tp_size)
        fa_slices = fa_spec.get_tp_transfer_slices(
            tp_rank, tp_size, remote_tp_size, total_num_kv_heads
        )
        remote_ranks = _remote_ranks_from_slices(fa_slices)

        worker = _make_mock_worker_for_splits(
            group_specs=[fa_spec],
            tp_mappings=(fa_slices,),
            remote_ranks=remote_ranks,
            engine_id=engine_id,
        )
        base_addr = 0x4000
        src_blocks_data = [(base_addr, local_block_len, 0)]
        splits = list(worker._build_local_splits(engine_id, src_blocks_data, 1))

        assert len(splits) == remote_tp_size
        chunk = local_block_len // remote_tp_size
        for idx, (rank, sl) in enumerate(sorted(fa_slices.items())):
            expected_offset = (
                sl.local_write_offset * local_block_len // len(sl.local_shard)
            )
            # Offsets should tile the local block without overlap
            assert expected_offset == idx * chunk
            addr, length, dev = splits[idx][0]
            assert addr == base_addr + expected_offset
            assert length == chunk
            assert dev == 0


class TestMambaPlanSplitHandles:
    """Verify split handles for Mamba with FA/SSM distinction."""

    def test_fa_and_ssm_different_split_factors(self):
        """Section 0 split by num_attn_reads, section 1 by abs_tp."""
        engine_id = "remote_0"
        # total_kv_heads=1 < remote_tp=2 triggers GQA dedup:
        # only remote rank 0 holds unique FA data.
        total_num_kv_heads = 1

        fa_spec = _make_fa_spec(num_kv_heads=1)
        mamba_spec = MagicMock(spec=MambaSpec)

        # local_tp=1, remote_tp=2
        # FA: 1 unique slice (reads from remote 0, GQA dedup skips rank 1)
        # Mamba: 2 slices (reads from remote 0 and 1)
        fa_slices = fa_spec.get_tp_transfer_slices(0, 1, 2, total_num_kv_heads)

        shard_mamba = ShardRange(0, 1, 1)
        ssm_slices = {
            0: TPTransferSlice(
                remote_rank=0,
                remote_shard=shard_mamba,
                local_shard=shard_mamba,
                transfer_range=shard_mamba,
            ),
            1: TPTransferSlice(
                remote_rank=1,
                remote_shard=shard_mamba,
                local_shard=shard_mamba,
                transfer_range=shard_mamba,
            ),
        }
        remote_ranks = _remote_ranks_from_slices(fa_slices, ssm_slices)

        worker = _make_mock_worker_for_splits(
            group_specs=[fa_spec, mamba_spec],
            tp_mappings=(fa_slices, ssm_slices),
            remote_ranks=remote_ranks,
            engine_id=engine_id,
        )

        # 2 FA descs + 1 SSM desc
        src_blocks_data = [
            (1000, 200, 0),  # FA desc 0
            (2000, 200, 0),  # FA desc 1
            (3000, 400, 0),  # SSM desc 0
        ]

        splits = list(worker._build_local_splits(engine_id, src_blocks_data, 2))

        assert len(splits) == 2  # 2 source ranks

        # Rank 0 is in fa_slices -> uses local_write_offset for FA offset
        fa_chunk = 200 // len(fa_slices)
        ssm_chunk = 400 // len(ssm_slices)

        # Rank 0 (remote_idx=0):
        # FA: chunk=200//1=200 (only 1 FA slice)
        # offset = local_write_offset * local_block_len // len(local_shard)
        # SSM: chunk=400//2=200, offset = remote_idx(0) * 200
        sl = fa_slices[0]
        fa_offset_r0 = sl.local_write_offset * 200 // len(sl.local_shard)
        assert splits[0][0] == (1000 + fa_offset_r0, fa_chunk, 0)
        assert splits[0][1] == (2000 + fa_offset_r0, fa_chunk, 0)
        assert splits[0][2] == (3000 + 0 * ssm_chunk, ssm_chunk, 0)

        # Rank 1 (remote_idx=1):
        # FA: rank 1 NOT in fa_slices -> GQA-deduped placeholder (addr, chunk, dev)
        # SSM: chunk=400//2=200, offset = remote_idx(1) * 200
        assert splits[1][0] == (1000, fa_chunk, 0)
        assert splits[1][1] == (2000, fa_chunk, 0)
        assert splits[1][2] == (3000 + 1 * ssm_chunk, ssm_chunk, 0)
