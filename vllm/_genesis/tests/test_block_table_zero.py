# SPDX-License-Identifier: Apache-2.0
"""TDD tests for vllm._genesis.kernels.block_table_zero (Patch 14).

Validates the tail-zero fix that prevents stale block IDs from leaking past
num_blocks_per_row when a block_table row slot is reused by a shorter request.

Author: Sandermage(Sander)-Barzov Aleksandr, Ukraine, Odessa
"""
from __future__ import annotations

import numpy as np
import pytest


class TestZeroBlockTableTail:
    """Group 1: Raw tail-zero primitive."""

    def test_zeros_exact_range(self):
        from vllm._genesis.kernels.block_table_zero import zero_block_table_tail

        table = np.full((3, 8), 99, dtype=np.int32)
        zero_block_table_tail(table, row_idx=1, end=3, max_per_req=8)

        # row 1 tail from col 3 onwards must be 0
        assert (table[1, 3:8] == 0).all()
        # row 1 prefix untouched
        assert (table[1, 0:3] == 99).all()
        # other rows untouched
        assert (table[0] == 99).all()
        assert (table[2] == 99).all()

    def test_safe_noop_when_end_at_boundary(self):
        """end == max_per_req → no zeroing (nothing to zero)."""
        from vllm._genesis.kernels.block_table_zero import zero_block_table_tail

        table = np.full((2, 4), 7, dtype=np.int32)
        zero_block_table_tail(table, row_idx=0, end=4, max_per_req=4)
        # Row 0 completely untouched
        assert (table[0] == 7).all()

    def test_rejects_end_exceeds_max(self):
        """end > max_per_req is a caller bug — rejected loudly, not silently."""
        from vllm._genesis.kernels.block_table_zero import zero_block_table_tail

        table = np.full((2, 4), 7, dtype=np.int32)
        with pytest.raises(ValueError, match="max_per_req"):
            zero_block_table_tail(table, row_idx=0, end=99, max_per_req=4)
        # Table untouched — validation ran before write
        assert (table[0] == 7).all()

    def test_zeros_entire_row_when_end_zero(self):
        from vllm._genesis.kernels.block_table_zero import zero_block_table_tail

        table = np.full((2, 4), 5, dtype=np.int32)
        zero_block_table_tail(table, row_idx=0, end=0, max_per_req=4)
        assert (table[0] == 0).all()
        assert (table[1] == 5).all()

    def test_rejects_negative_end(self):
        from vllm._genesis.kernels.block_table_zero import zero_block_table_tail

        table = np.zeros((2, 4), dtype=np.int32)
        with pytest.raises(ValueError, match="end=-1"):
            zero_block_table_tail(table, row_idx=0, end=-1, max_per_req=4)

    def test_rejects_max_below_end(self):
        from vllm._genesis.kernels.block_table_zero import zero_block_table_tail

        table = np.zeros((2, 4), dtype=np.int32)
        with pytest.raises(ValueError, match="max_per_req"):
            zero_block_table_tail(table, row_idx=0, end=5, max_per_req=4)

    def test_raises_on_bad_row_idx(self):
        from vllm._genesis.kernels.block_table_zero import zero_block_table_tail

        table = np.zeros((2, 4), dtype=np.int32)
        with pytest.raises(IndexError):
            zero_block_table_tail(table, row_idx=5, end=1, max_per_req=4)


class TestAppendRowWithTailZero:
    """Group 2: Full append semantics matching upstream PR #39591."""

    def test_empty_block_ids_is_noop(self):
        from vllm._genesis.kernels.block_table_zero import append_row_with_tail_zero

        table = np.full((2, 4), 99, dtype=np.int32)
        counts = np.array([1, 0], dtype=np.int32)
        append_row_with_tail_zero(table, counts, row_idx=0,
                                  block_ids=[], max_per_req=4)
        # Nothing changed
        assert counts[0] == 1
        assert (table[0] == 99).all()

    def test_append_first_batch_zeros_tail(self):
        """Fresh append should write blocks AND zero the tail."""
        from vllm._genesis.kernels.block_table_zero import append_row_with_tail_zero

        table = np.full((2, 6), 99, dtype=np.int32)  # stale data
        counts = np.zeros(2, dtype=np.int32)

        append_row_with_tail_zero(
            table, counts, row_idx=0,
            block_ids=[10, 20, 30], max_per_req=6,
        )

        assert counts[0] == 3
        assert list(table[0, 0:3]) == [10, 20, 30]
        # Tail from col 3 must be zero, not stale 99
        assert (table[0, 3:6] == 0).all()

    def test_sequential_appends(self):
        from vllm._genesis.kernels.block_table_zero import append_row_with_tail_zero

        table = np.full((2, 6), 99, dtype=np.int32)
        counts = np.zeros(2, dtype=np.int32)

        append_row_with_tail_zero(table, counts, 0, [1, 2], max_per_req=6)
        append_row_with_tail_zero(table, counts, 0, [3, 4], max_per_req=6)

        assert counts[0] == 4
        assert list(table[0, 0:4]) == [1, 2, 3, 4]
        assert (table[0, 4:6] == 0).all()

    def test_tail_always_zeroed_even_when_fully_filled(self):
        """Fill exactly to max → tail zeroing is a no-op but must not crash."""
        from vllm._genesis.kernels.block_table_zero import append_row_with_tail_zero

        table = np.full((1, 3), 99, dtype=np.int32)
        counts = np.zeros(1, dtype=np.int32)

        append_row_with_tail_zero(table, counts, 0, [5, 6, 7], max_per_req=3)
        assert list(table[0]) == [5, 6, 7]
        assert counts[0] == 3


class TestMoveRowWithTailZero:
    """Group 3: Row copy that correctly clears the destination tail."""

    def test_move_shorter_into_longer_clears_tail(self):
        """This is the critical regression: tgt previously held 6 blocks, src 2."""
        from vllm._genesis.kernels.block_table_zero import move_row_with_tail_zero

        table = np.full((2, 8), 0, dtype=np.int32)
        table[0, :2] = [10, 20]           # src shorter
        table[1, :6] = [91, 92, 93, 94, 95, 96]  # tgt previously longer (stale)
        counts = np.array([2, 6], dtype=np.int32)

        move_row_with_tail_zero(table, counts, src=0, tgt=1, max_per_req=8)

        # tgt prefix = src
        assert list(table[1, :2]) == [10, 20]
        # tgt tail from col 2 must be zero — stale 93/94/95/96 must NOT leak
        assert (table[1, 2:8] == 0).all()
        # src untouched
        assert list(table[0, :2]) == [10, 20]
        # counts updated
        assert counts[1] == 2

    def test_move_longer_into_shorter(self):
        from vllm._genesis.kernels.block_table_zero import move_row_with_tail_zero

        table = np.zeros((2, 8), dtype=np.int32)
        table[0, :5] = [1, 2, 3, 4, 5]
        table[1, :2] = [99, 88]
        counts = np.array([5, 2], dtype=np.int32)

        move_row_with_tail_zero(table, counts, src=0, tgt=1, max_per_req=8)

        assert list(table[1, :5]) == [1, 2, 3, 4, 5]
        assert (table[1, 5:8] == 0).all()
        assert counts[1] == 5

    def test_move_equal_length(self):
        from vllm._genesis.kernels.block_table_zero import move_row_with_tail_zero

        table = np.zeros((2, 4), dtype=np.int32)
        table[0, :3] = [1, 2, 3]
        table[1, :3] = [11, 22, 33]
        counts = np.array([3, 3], dtype=np.int32)

        move_row_with_tail_zero(table, counts, 0, 1, max_per_req=4)

        assert list(table[1, :3]) == [1, 2, 3]
        assert table[1, 3] == 0


class TestNeverCrashInvariants:
    """Group 4: Must not crash on edge-case inputs."""

    def test_empty_table(self):
        """0-row table handled without crash."""
        from vllm._genesis.kernels.block_table_zero import zero_block_table_tail

        table = np.zeros((0, 4), dtype=np.int32)
        # No row to address — should raise IndexError cleanly, not segfault
        with pytest.raises(IndexError):
            zero_block_table_tail(table, row_idx=0, end=0, max_per_req=4)

    def test_works_with_int64_dtype(self):
        """Some vLLM builds use int64 for block IDs — must still work."""
        from vllm._genesis.kernels.block_table_zero import append_row_with_tail_zero

        table = np.full((1, 4), 99, dtype=np.int64)
        counts = np.zeros(1, dtype=np.int64)

        append_row_with_tail_zero(table, counts, 0, [1, 2], max_per_req=4)
        assert table.dtype == np.int64
        assert list(table[0]) == [1, 2, 0, 0]
