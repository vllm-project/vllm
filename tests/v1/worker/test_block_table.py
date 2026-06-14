# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``vllm.v1.worker.block_table.BlockTable``.

These tests target the fix for vllm-project/vllm#39589: stale block IDs
leaking past ``num_blocks_per_row[row]`` when a row is reused (or is the
target of a ``move_row``) by a later request whose block count is
smaller than the previous occupant's.

The core invariant the fix must establish is::

    block_table.np[row, num_blocks_per_row[row]:max_num_blocks_per_req] == 0

for every row, after *any* sequence of ``append_row`` / ``add_row`` /
``move_row`` / ``clear_row`` calls. FlashInfer's
``_copy_page_indices_kernel`` can otherwise index past
``num_blocks_per_row[row]`` and read a stale block ID that still points
at live KV from a concurrent request, producing the non-deterministic
output documented in #39589.

Tests here are written from the spec in PLAN.md only — they must fail
against the buggy pre-fix implementation and pass against the fixed one.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vllm.v1.worker.block_table import BlockTable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_block_table(
    block_size: int = 16,
    max_num_reqs: int = 8,
    max_num_blocks_per_req: int = 32,
    max_num_batched_tokens: int = 512,
    kernel_block_size: int | None = None,
    device: str = "cpu",
) -> BlockTable:
    """Construct a non-hybrid ``BlockTable`` for unit testing.

    ``kernel_block_size`` defaults to ``block_size`` so ``use_hybrid_blocks``
    is ``False`` and the test path does not go through
    ``map_to_kernel_blocks``.
    """
    if kernel_block_size is None:
        kernel_block_size = block_size
    return BlockTable(
        block_size=block_size,
        max_num_reqs=max_num_reqs,
        max_num_blocks_per_req=max_num_blocks_per_req,
        max_num_batched_tokens=max_num_batched_tokens,
        pin_memory=False,
        device=torch.device(device),
        kernel_block_size=kernel_block_size,
        cp_kv_cache_interleave_size=1,
    )


def row_tail(bt: BlockTable, row_idx: int) -> np.ndarray:
    """Return ``block_table.np[row_idx, num_blocks_per_row[row_idx]:]``."""
    n = int(bt.num_blocks_per_row[row_idx])
    return bt.block_table.np[row_idx, n:]


def assert_tail_zero(bt: BlockTable, row_idx: int) -> None:
    """The central invariant: everything past the logical end of the row
    must be zero.
    """
    tail = row_tail(bt, row_idx)
    assert np.all(tail == 0), (
        f"row {row_idx} violated tail-zero invariant "
        f"(num_blocks={int(bt.num_blocks_per_row[row_idx])}): "
        f"tail={tail.tolist()}"
    )


def assert_all_tails_zero(bt: BlockTable) -> None:
    for r in range(bt.block_table.np.shape[0]):
        assert_tail_zero(bt, r)


# ---------------------------------------------------------------------------
# Fresh state
# ---------------------------------------------------------------------------


def test_fresh_block_table_is_all_zero():
    bt = make_block_table(max_num_reqs=4, max_num_blocks_per_req=16)
    assert bt.block_table.np.shape == (4, 16)
    assert np.all(bt.block_table.np == 0)
    assert np.all(np.asarray(bt.num_blocks_per_row) == 0)
    assert_all_tails_zero(bt)


def test_fresh_block_table_max_num_blocks_per_req_attribute():
    bt = make_block_table(max_num_blocks_per_req=24)
    # Used by the candidate fix to decide how far to zero the tail.
    assert bt.max_num_blocks_per_req == 24


# ---------------------------------------------------------------------------
# append_row: basic correctness
# ---------------------------------------------------------------------------


def test_append_row_writes_block_ids_at_prefix():
    bt = make_block_table()
    ids = [7, 11, 13, 17]
    bt.append_row(ids, row_idx=2)
    assert int(bt.num_blocks_per_row[2]) == len(ids)
    assert bt.block_table.np[2, : len(ids)].tolist() == ids


def test_append_row_accumulates_within_same_row():
    """append_row is cumulative: multiple calls concatenate block IDs.

    This is the prefix-caching / chunked-prefill shape — cached blocks
    land first, then new blocks are appended.
    """
    bt = make_block_table()
    bt.append_row([1, 2, 3], row_idx=0)
    bt.append_row([4, 5], row_idx=0)
    assert int(bt.num_blocks_per_row[0]) == 5
    assert bt.block_table.np[0, :5].tolist() == [1, 2, 3, 4, 5]
    assert_tail_zero(bt, 0)


def test_append_row_does_not_clobber_cached_prefix():
    """Per PLAN.md §4 edge case 8: ``append_row`` must not zero an
    earlier append's block IDs. The tail-zeroing must only start at the
    *new* end of the row.
    """
    bt = make_block_table()
    bt.append_row([101, 102, 103, 104], row_idx=0)  # "cached" blocks
    bt.append_row([201, 202], row_idx=0)            # "new" blocks
    assert bt.block_table.np[0, :6].tolist() == [101, 102, 103, 104, 201, 202]
    assert_tail_zero(bt, 0)


def test_append_row_empty_list_is_noop():
    """An empty append must not mutate the row or its counter."""
    bt = make_block_table()
    bt.append_row([1, 2, 3], row_idx=0)
    snapshot = bt.block_table.np[0].copy()
    prev_n = int(bt.num_blocks_per_row[0])
    bt.append_row([], row_idx=0)
    assert int(bt.num_blocks_per_row[0]) == prev_n
    assert np.array_equal(bt.block_table.np[0], snapshot)


def test_append_row_updates_only_requested_row():
    bt = make_block_table()
    bt.append_row([1, 2], row_idx=0)
    bt.append_row([10, 20, 30], row_idx=1)
    bt.append_row([100], row_idx=2)
    assert bt.block_table.np[0, :2].tolist() == [1, 2]
    assert bt.block_table.np[1, :3].tolist() == [10, 20, 30]
    assert bt.block_table.np[2, :1].tolist() == [100]
    assert int(bt.num_blocks_per_row[0]) == 2
    assert int(bt.num_blocks_per_row[1]) == 3
    assert int(bt.num_blocks_per_row[2]) == 1
    assert_all_tails_zero(bt)


# ---------------------------------------------------------------------------
# append_row / add_row: tail-zero invariant (core regression for #39589)
# ---------------------------------------------------------------------------


def test_append_row_zeros_full_tail_after_single_append():
    """Minimal tail-zero guarantee: after a single ``append_row`` on a
    previously-unused row, the remainder of the row must be zero.

    Would trivially pass on an untouched row even pre-fix — baseline
    check.
    """
    bt = make_block_table(max_num_blocks_per_req=32)
    bt.append_row([100, 200, 300], row_idx=0)
    assert int(bt.num_blocks_per_row[0]) == 3
    assert np.all(bt.block_table.np[0, 3:] == 0)


def test_add_row_zeros_previous_tail_when_new_row_is_shorter():
    """PLAN.md §5.1.1 — flagship regression test.

    Put a long sequence of block IDs into row 0, then reuse row 0 for a
    shorter request via ``add_row``. The old tail must not survive.
    """
    bt = make_block_table(max_num_reqs=4, max_num_blocks_per_req=16)
    long_ids = [1, 2, 3, 4, 5]
    bt.add_row(long_ids, row_idx=0)

    short_ids = [10, 20]
    bt.add_row(short_ids, row_idx=0)

    assert int(bt.num_blocks_per_row[0]) == 2
    assert bt.block_table.np[0, :2].tolist() == [10, 20]
    # No [3, 4, 5] may leak through.
    leaked = bt.block_table.np[0, 2:]
    assert np.all(leaked == 0), (
        "Stale block IDs from the previous (longer) request leaked into "
        f"row 0 after add_row: {leaked.tolist()}"
    )


def test_append_row_zeros_tail_all_the_way_to_max_num_blocks_per_req():
    """PLAN.md §3.5 candidate fix: tail is zeroed through
    ``max_num_blocks_per_req``, not just up to the previous row length.
    """
    bt = make_block_table(max_num_reqs=4, max_num_blocks_per_req=32)
    # Pollute the raw row buffer directly, simulating any prior write
    # that the fix must still defeat.
    bt.block_table.np[0, :] = 42
    bt.num_blocks_per_row[0] = 0

    bt.append_row([7, 8], row_idx=0)
    assert int(bt.num_blocks_per_row[0]) == 2
    assert bt.block_table.np[0, :2].tolist() == [7, 8]
    assert np.all(bt.block_table.np[0, 2:] == 0), (
        "append_row must zero the tail to max_num_blocks_per_req"
    )


def test_row_reuse_does_not_leak_across_many_rounds():
    """A single row reused many times with varying lengths; after every
    ``add_row`` the tail must be zero.
    """
    bt = make_block_table(max_num_reqs=2, max_num_blocks_per_req=32)
    sequences = [
        [1, 2, 3, 4, 5, 6, 7, 8],
        [10, 20],
        [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        [1],
        [111, 222, 333],
    ]
    for ids in sequences:
        bt.add_row(ids, row_idx=0)
        assert int(bt.num_blocks_per_row[0]) == len(ids)
        assert bt.block_table.np[0, : len(ids)].tolist() == ids
        assert_tail_zero(bt, 0)


def test_concurrent_variable_length_prefill_rows_no_leak():
    """Models the #39589 scenario more literally — two rows holding
    concurrent requests of different lengths, then both are reused by
    new requests with the lengths swapped.
    """
    bt = make_block_table(max_num_reqs=4, max_num_blocks_per_req=32)

    # Round 1: req A (long) in row 0, req B (short) in row 1.
    bt.add_row(list(range(2001, 2021)), row_idx=0)  # 20 blocks
    bt.add_row(list(range(3001, 3005)), row_idx=1)  # 4 blocks
    assert_all_tails_zero(bt)

    # Round 2: both finish; rows reused with swapped lengths.
    bt.add_row(list(range(4001, 4005)), row_idx=0)  # 4 blocks now
    bt.add_row(list(range(5001, 5021)), row_idx=1)  # 20 blocks now

    assert int(bt.num_blocks_per_row[0]) == 4
    assert bt.block_table.np[0, :4].tolist() == list(range(4001, 4005))
    leaked_0 = bt.block_table.np[0, 4:]
    assert np.all(leaked_0 == 0), (
        f"Row 0 still contains stale round-1 block IDs: {leaked_0.tolist()}"
    )

    assert int(bt.num_blocks_per_row[1]) == 20
    assert bt.block_table.np[1, :20].tolist() == list(range(5001, 5021))
    assert_tail_zero(bt, 1)


# ---------------------------------------------------------------------------
# move_row / swap_row
# ---------------------------------------------------------------------------


def test_move_row_copies_src_block_ids_to_tgt():
    bt = make_block_table()
    bt.append_row([5, 6, 7, 8], row_idx=3)
    bt.move_row(src=3, tgt=1)
    assert int(bt.num_blocks_per_row[1]) == 4
    assert bt.block_table.np[1, :4].tolist() == [5, 6, 7, 8]


def test_move_row_tgt_tail_zero_when_tgt_was_previously_empty():
    """PLAN.md §5.1.2 literal: append to row 3, move_row(3, 1), assert
    row 1's tail is zero. Row 1 starts empty here.
    """
    bt = make_block_table()
    bt.append_row([11, 22, 33, 44], row_idx=3)
    bt.move_row(src=3, tgt=1)
    assert_tail_zero(bt, 1)


def test_move_row_zeros_tgt_tail_when_src_shorter_than_previous_tgt():
    """Harder case — ``tgt`` already held a longer request's block IDs.
    After the move, those must not leak past the new ``num_blocks_per_row[tgt]``.
    """
    bt = make_block_table(max_num_reqs=4, max_num_blocks_per_req=16)
    bt.append_row([9001, 9002, 9003, 9004, 9005, 9006, 9007, 9008], row_idx=1)
    bt.append_row([50, 51], row_idx=3)
    bt.move_row(src=3, tgt=1)
    assert int(bt.num_blocks_per_row[1]) == 2
    assert bt.block_table.np[1, :2].tolist() == [50, 51]
    leaked = bt.block_table.np[1, 2:]
    assert np.all(leaked == 0), (
        f"move_row left stale tgt block IDs past num_blocks_per_row: "
        f"{leaked.tolist()}"
    )


def test_move_row_preserves_unrelated_rows():
    bt = make_block_table()
    bt.append_row([1, 2, 3], row_idx=0)
    bt.append_row([40, 41, 42, 43], row_idx=2)
    bt.append_row([99, 88], row_idx=3)
    bt.move_row(src=3, tgt=1)

    # Unrelated rows are untouched.
    assert bt.block_table.np[0, :3].tolist() == [1, 2, 3]
    assert bt.block_table.np[2, :4].tolist() == [40, 41, 42, 43]
    assert_tail_zero(bt, 0)
    assert_tail_zero(bt, 2)
    # tgt is the new home of src's blocks.
    assert bt.block_table.np[1, :2].tolist() == [99, 88]
    assert_tail_zero(bt, 1)


# ---------------------------------------------------------------------------
# clear_row
# ---------------------------------------------------------------------------


def test_clear_row_zeros_prefix_and_resets_counter():
    bt = make_block_table()
    bt.append_row([1, 2, 3, 4, 5], row_idx=2)
    bt.clear_row(row_idx=2)
    # Whatever num_blocks_per_row becomes, the tail-zero invariant must
    # hold over the *whole* row — nothing may remain from [1..5].
    assert_tail_zero(bt, 2)
    # clear_row should make the row appear empty for subsequent reads.
    assert int(bt.num_blocks_per_row[2]) == 0


def test_clear_row_does_not_touch_sibling_rows():
    bt = make_block_table()
    bt.append_row([1, 2, 3], row_idx=0)
    bt.append_row([4, 5], row_idx=1)
    bt.clear_row(row_idx=0)
    assert bt.block_table.np[1, :2].tolist() == [4, 5]
    assert_tail_zero(bt, 1)


def test_append_after_clear_row_starts_fresh():
    bt = make_block_table()
    bt.append_row([1, 2, 3, 4, 5], row_idx=0)
    bt.clear_row(row_idx=0)
    bt.append_row([9, 8], row_idx=0)
    assert int(bt.num_blocks_per_row[0]) == 2
    assert bt.block_table.np[0, :2].tolist() == [9, 8]
    assert_tail_zero(bt, 0)


# ---------------------------------------------------------------------------
# add_row behavior — resets then appends
# ---------------------------------------------------------------------------


def test_add_row_replaces_row_contents():
    bt = make_block_table()
    bt.append_row([1, 2, 3], row_idx=0)
    bt.add_row([7, 8, 9, 10], row_idx=0)
    assert int(bt.num_blocks_per_row[0]) == 4
    assert bt.block_table.np[0, :4].tolist() == [7, 8, 9, 10]
    assert_tail_zero(bt, 0)


def test_add_row_on_empty_row_matches_append_row():
    bt1 = make_block_table()
    bt2 = make_block_table()
    bt1.append_row([1, 2, 3], row_idx=0)
    bt2.add_row([1, 2, 3], row_idx=0)
    assert np.array_equal(bt1.block_table.np[0], bt2.block_table.np[0])
    assert int(bt1.num_blocks_per_row[0]) == int(bt2.num_blocks_per_row[0])


def test_add_row_with_same_length_does_not_leak():
    """Edge case: same-length replacement. No "tail" to leak, but the
    content must still be replaced, not accumulated.
    """
    bt = make_block_table()
    bt.add_row([1, 2, 3], row_idx=0)
    bt.add_row([10, 20, 30], row_idx=0)
    assert int(bt.num_blocks_per_row[0]) == 3
    assert bt.block_table.np[0, :3].tolist() == [10, 20, 30]
    assert_tail_zero(bt, 0)


# ---------------------------------------------------------------------------
# Boundary / edge cases
# ---------------------------------------------------------------------------


def test_append_row_filling_to_max_num_blocks_per_req():
    """Fill a row up to the full ``max_num_blocks_per_req``. The tail
    (empty slice) is vacuously zero.
    """
    bt = make_block_table(max_num_reqs=2, max_num_blocks_per_req=8)
    bt.append_row([1, 2, 3, 4, 5, 6, 7, 8], row_idx=0)
    assert int(bt.num_blocks_per_row[0]) == 8
    assert bt.block_table.np[0].tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert row_tail(bt, 0).size == 0
    assert_tail_zero(bt, 0)


def test_append_row_accepts_numpy_array_of_block_ids():
    bt = make_block_table()
    bt.append_row(np.array([4, 5, 6], dtype=np.int32), row_idx=0)
    assert bt.block_table.np[0, :3].tolist() == [4, 5, 6]
    assert_tail_zero(bt, 0)


def test_large_block_id_values_preserved():
    """Block IDs can be large unsigned ints. Ensure the tail-zero path
    doesn't accidentally mask or truncate valid IDs.
    """
    bt = make_block_table()
    large_id = 2**20
    bt.append_row([large_id, large_id + 1], row_idx=0)
    assert int(bt.block_table.np[0, 0]) == large_id
    assert int(bt.block_table.np[0, 1]) == large_id + 1
    assert_tail_zero(bt, 0)


def test_pure_decode_step_no_append_is_stable():
    """PLAN.md §4 edge case 1: pure-decode steps do not call ``append_row``.
    After an initial build-up, subsequent ``move_row`` / ``clear_row``
    patterns (as used by persistent batching) still preserve the invariant.
    """
    bt = make_block_table(max_num_reqs=4, max_num_blocks_per_req=16)
    for r in range(4):
        bt.append_row([r * 10 + i for i in range(4)], row_idx=r)
    assert_all_tails_zero(bt)
    # Simulate a persistent-batch compaction: move row 3 into slot 1.
    bt.move_row(src=3, tgt=1)
    assert_all_tails_zero(bt)
    bt.clear_row(row_idx=2)
    assert_all_tails_zero(bt)


def test_persistent_batch_reuse_invariant_across_rounds():
    """Simulate five consecutive scheduler steps in which requests come
    and go. After each step the tail-zero invariant must hold.
    """
    bt = make_block_table(max_num_reqs=4, max_num_blocks_per_req=32)

    # Step 1: 3 reqs of different sizes.
    bt.add_row([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], row_idx=0)
    bt.add_row([20, 21, 22], row_idx=1)
    bt.add_row([30, 31, 32, 33, 34, 35], row_idx=2)
    assert_all_tails_zero(bt)

    # Step 2: req in row 1 finishes; row 3 gets a new short req.
    bt.clear_row(row_idx=1)
    bt.add_row([40], row_idx=3)
    assert_all_tails_zero(bt)

    # Step 3: row 2's req finishes and is compacted — move row 3 → 2.
    bt.clear_row(row_idx=2)
    bt.move_row(src=3, tgt=2)
    bt.clear_row(row_idx=3)
    assert_all_tails_zero(bt)

    # Step 4: new very long req lands in row 1, short req in row 3.
    bt.add_row(list(range(500, 530)), row_idx=1)
    bt.add_row([99, 98], row_idx=3)
    assert_all_tails_zero(bt)

    # Step 5: the long req is replaced by a tiny one — the classic
    # leak shape — and row 0's big req is replaced by a medium one.
    bt.add_row([7], row_idx=1)
    bt.add_row([60, 61, 62], row_idx=0)
    assert_all_tails_zero(bt)

    # Spot-check the content of the row that was just shrunk.
    assert int(bt.num_blocks_per_row[1]) == 1
    assert bt.block_table.np[1, :1].tolist() == [7]
    assert np.all(bt.block_table.np[1, 1:] == 0)


def test_all_concurrent_prefill_identical_lengths_remains_stable():
    """PLAN.md §4 edge case 3: N concurrent prefills of identical length
    are reported stable even pre-fix. The fix must not break this.
    """
    bt = make_block_table(max_num_reqs=4, max_num_blocks_per_req=16)
    for r in range(4):
        bt.add_row([1000 + r, 1100 + r, 1200 + r, 1300 + r], row_idx=r)
    for r in range(4):
        assert int(bt.num_blocks_per_row[r]) == 4
        assert bt.block_table.np[r, :4].tolist() == [
            1000 + r,
            1100 + r,
            1200 + r,
            1300 + r,
        ]
        assert_tail_zero(bt, r)


# ---------------------------------------------------------------------------
# Randomized invariant check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 42, 12345])
def test_tail_zero_invariant_survives_random_operation_sequence(seed: int):
    """Fuzz ``add_row`` / ``append_row`` / ``move_row`` / ``clear_row``
    over many steps and assert the tail-zero invariant after every op.
    """
    rng = np.random.default_rng(seed)
    max_num_reqs = 4
    max_num_blocks_per_req = 16
    bt = make_block_table(
        max_num_reqs=max_num_reqs,
        max_num_blocks_per_req=max_num_blocks_per_req,
    )

    for step in range(200):
        op = int(rng.integers(0, 4))
        row = int(rng.integers(0, max_num_reqs))
        if op == 0:  # full reset of this row with a new request
            n = int(rng.integers(1, max_num_blocks_per_req + 1))
            ids = [int(x) for x in rng.integers(1, 10_000, size=n)]
            bt.add_row(ids, row)
        elif op == 1:  # append more blocks if there is room
            remaining = (
                max_num_blocks_per_req - int(bt.num_blocks_per_row[row])
            )
            if remaining > 0:
                n = int(rng.integers(1, remaining + 1))
                ids = [int(x) for x in rng.integers(1, 10_000, size=n)]
                bt.append_row(ids, row)
        elif op == 2:  # move a random row onto another
            src = int(rng.integers(0, max_num_reqs))
            tgt = int(rng.integers(0, max_num_reqs))
            if src != tgt:
                bt.move_row(src=src, tgt=tgt)
        else:  # clear
            bt.clear_row(row)

        for r in range(max_num_reqs):
            n = int(bt.num_blocks_per_row[r])
            tail = bt.block_table.np[r, n:]
            assert np.all(tail == 0), (
                f"step={step} op={op} row={r} num_blocks={n}: "
                f"tail leaked {tail.tolist()}"
            )


# ---------------------------------------------------------------------------
# Hybrid blocks (use_hybrid_blocks=True)
# ---------------------------------------------------------------------------


def test_hybrid_block_table_tail_zero_on_reuse():
    """The fix must cover the hybrid-blocks path (kernel block size !=
    kvcache-manager block size). PLAN.md §4 edge case 9.
    """
    bt = make_block_table(
        block_size=32,
        max_num_reqs=4,
        max_num_blocks_per_req=32,
        max_num_batched_tokens=256,
        kernel_block_size=16,
    )
    assert bt.use_hybrid_blocks is True

    # First, a request that consumes many kernel blocks.
    bt.add_row([1, 2, 3, 4], row_idx=0)
    long_n = int(bt.num_blocks_per_row[0])
    assert long_n > 0

    # Then a much shorter request in the same row.
    bt.add_row([100], row_idx=0)
    short_n = int(bt.num_blocks_per_row[0])
    assert short_n < long_n

    tail = bt.block_table.np[0, short_n:]
    assert np.all(tail == 0), (
        f"hybrid append_row leaked tail on row reuse: {tail.tolist()}"
    )


# ---------------------------------------------------------------------------
# num_blocks_per_row / seq_len invariant (PLAN.md §3.1 investigation hook)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seq_len, block_size, expected_num_blocks",
    [
        (1, 16, 1),
        (15, 16, 1),
        (16, 16, 1),
        (17, 16, 2),
        (31, 16, 2),
        (32, 16, 2),
        (33, 16, 3),
        (128, 16, 8),
        (256, 16, 16),
    ],
)
def test_num_blocks_per_row_matches_ceil_div_after_add_row(
    seq_len: int, block_size: int, expected_num_blocks: int
):
    """PLAN.md §3.1 investigation: after ``add_row`` the caller is
    expected to satisfy ``num_blocks_per_row[req_idx] >=
    ceil(seq_len / block_size)``. For a freshly added row whose block
    count exactly matches the request's length, equality must hold.
    """
    bt = make_block_table(
        block_size=block_size,
        max_num_blocks_per_req=max(32, expected_num_blocks + 4),
    )
    ids = list(range(100, 100 + expected_num_blocks))
    bt.add_row(ids, row_idx=0)
    assert int(bt.num_blocks_per_row[0]) == expected_num_blocks
    assert int(bt.num_blocks_per_row[0]) >= (seq_len + block_size - 1) // block_size
    assert_tail_zero(bt, 0)


# ---------------------------------------------------------------------------
# Regression guard: cross-row isolation
# ---------------------------------------------------------------------------


def test_cross_row_isolation_under_stress():
    """Changes to row R must leave every other row byte-identical,
    including each other row's tail zeros.
    """
    bt = make_block_table(max_num_reqs=4, max_num_blocks_per_req=16)
    bt.append_row([1, 2, 3], row_idx=0)
    bt.append_row([4, 5], row_idx=2)
    snapshot_0 = bt.block_table.np[0].copy()
    snapshot_2 = bt.block_table.np[2].copy()

    # Hammer rows 1 and 3.
    bt.add_row([99, 98, 97, 96, 95], row_idx=1)
    bt.add_row([11], row_idx=3)
    bt.add_row([22, 23], row_idx=1)
    bt.clear_row(row_idx=3)
    bt.add_row([30, 31, 32, 33], row_idx=3)

    assert np.array_equal(bt.block_table.np[0], snapshot_0)
    assert np.array_equal(bt.block_table.np[2], snapshot_2)
    assert_all_tails_zero(bt)


# ---------------------------------------------------------------------------
# PLAN.md §5.1.1 (tail-zero-on-append) and §5.1.2 (tail-zero-on-move)
# — literal transcriptions so the spec's named tests exist.
# ---------------------------------------------------------------------------


def test_spec_5_1_1_tail_zero_on_append():
    """Literal transcription of PLAN.md §5.1.1 "tail-zero-on-append"."""
    bt = make_block_table(max_num_reqs=4, max_num_blocks_per_req=16)
    bt.add_row([1, 2, 3, 4, 5], row_idx=0)
    bt.add_row([10, 20], row_idx=0)
    assert np.all(bt.block_table.np[0, 2:] == 0)


def test_spec_5_1_2_tail_zero_on_move():
    """Literal transcription of PLAN.md §5.1.2 "tail-zero-on-move"."""
    bt = make_block_table(max_num_reqs=4, max_num_blocks_per_req=16)
    bt.append_row([11, 22, 33, 44], row_idx=3)
    bt.move_row(src=3, tgt=1)
    n = int(bt.num_blocks_per_row[1])
    assert np.all(bt.block_table.np[1, n:] == 0)


def test_spec_5_1_3_no_regression_on_standard_append():
    """Literal transcription of PLAN.md §5.1.3 "no-regression-on-
    standard-append": a sequence of ``append_row`` calls within a single
    row produces the same ``[:num_blocks_per_row[i]]`` prefix as before
    the fix.
    """
    bt = make_block_table()
    bt.append_row([1, 2], row_idx=0)
    bt.append_row([3, 4, 5], row_idx=0)
    bt.append_row([6], row_idx=0)
    n = int(bt.num_blocks_per_row[0])
    assert n == 6
    assert bt.block_table.np[0, :n].tolist() == [1, 2, 3, 4, 5, 6]
