# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for BlockPool.free_blocks() duplicate-block handling.

Regression tests for https://github.com/vllm-project/vllm/issues/42571.
When sliding-window attention reuses a physical block within one request,
the same KVCacheBlock object may appear multiple times in the list passed
to ``free_blocks()``.  Without deduplication the free-list accounting
becomes corrupted (num_free_blocks over-counts), eventually crashing the
engine when it tries to allocate more blocks than physically exist.
"""

import pytest

from vllm.v1.core.block_pool import BlockPool

pytestmark = pytest.mark.cpu_test

NUM_GPU_BLOCKS = 8
HASH_BLOCK_SIZE = 16


def _make_pool(
    num_gpu_blocks: int = NUM_GPU_BLOCKS,
    enable_caching: bool = True,
) -> BlockPool:
    """Create a small BlockPool for testing."""
    return BlockPool(
        num_gpu_blocks=num_gpu_blocks,
        enable_caching=enable_caching,
        hash_block_size=HASH_BLOCK_SIZE,
    )


def _free_list_blocks(pool: BlockPool) -> list:
    """Return the actual linked-list contents of the free queue."""
    return pool.free_block_queue.get_all_free_blocks()


class TestFreeBlocksDedup:
    """Verify that ``free_blocks()`` is safe against duplicate entries."""

    def test_duplicate_block_freed_once(self):
        """Passing the same block twice must decrement ref_cnt once and
        append the block to the free queue exactly once."""
        pool = _make_pool()

        # Allocate a single block (ref_cnt becomes 1 via get_new_blocks).
        block = pool.get_new_blocks(1)[0]
        assert block.ref_cnt == 1

        # Simulate a second logical reference (e.g. touch from CPU offload).
        block.ref_cnt += 1
        assert block.ref_cnt == 2

        free_before = pool.get_num_free_blocks()

        # The buggy code path: caller passes [block, block].
        pool.free_blocks([block, block])

        # ref_cnt should have been decremented exactly once (2 -> 1).
        assert (
            block.ref_cnt == 1
        ), f"Expected ref_cnt=1 after dedup, got {block.ref_cnt}"

        # The block still has a reference, so it must NOT be in the free
        # queue.
        assert pool.get_num_free_blocks() == free_before

        # Now drop the last reference normally.
        pool.free_blocks([block])
        assert block.ref_cnt == 0
        assert pool.get_num_free_blocks() == free_before + 1

        # The block appears exactly once in the free list.
        free_blocks = _free_list_blocks(pool)
        assert free_blocks.count(block) == 1

    def test_duplicate_block_freed_to_zero(self):
        """When the single unique block has ref_cnt == 1 and is passed
        twice, dedup means only one decrement happens (1 -> 0) and the
        block is appended to the free queue exactly once."""
        pool = _make_pool()

        block = pool.get_new_blocks(1)[0]
        assert block.ref_cnt == 1

        free_before = pool.get_num_free_blocks()

        pool.free_blocks([block, block])

        # Dedup: only one decrement, so ref_cnt goes 1 -> 0.
        assert block.ref_cnt == 0
        assert pool.get_num_free_blocks() == free_before + 1

        # Block appears exactly once in the actual linked list.
        free_blocks = _free_list_blocks(pool)
        assert free_blocks.count(block) == 1

    def test_free_list_accounting_consistent(self):
        """num_free_blocks must always equal the length of the actual
        linked list after any free_blocks() call with duplicates."""
        pool = _make_pool(num_gpu_blocks=4)

        blocks = pool.get_new_blocks(3)
        for b in blocks:
            b.ref_cnt += 1  # Simulate two references per block.

        # Free with duplicates: [B0, B0, B1, B1, B2, B2]
        duped = [b for b in blocks for _ in range(2)]
        pool.free_blocks(duped)

        # Each block had ref_cnt=2, one dedup decrement -> ref_cnt=1.
        for b in blocks:
            assert b.ref_cnt == 1

        # No blocks freed (ref_cnt > 0), so accounting is unchanged.
        free_count = pool.get_num_free_blocks()
        actual_free = len(_free_list_blocks(pool))
        assert free_count == actual_free, (
            f"num_free_blocks ({free_count}) != actual list length " f"({actual_free})"
        )

    def test_no_regression_unique_blocks(self):
        """Normal case (no duplicates) must still work correctly."""
        pool = _make_pool()

        blocks = pool.get_new_blocks(3)
        for b in blocks:
            assert b.ref_cnt == 1

        free_before = pool.get_num_free_blocks()

        pool.free_blocks(blocks)

        for b in blocks:
            assert b.ref_cnt == 0
        assert pool.get_num_free_blocks() == free_before + 3

        free_blocks = _free_list_blocks(pool)
        for b in blocks:
            assert free_blocks.count(b) == 1

    def test_triple_duplicate(self):
        """Three copies of the same block must still be deduped to one."""
        pool = _make_pool()

        block = pool.get_new_blocks(1)[0]
        assert block.ref_cnt == 1

        pool.free_blocks([block, block, block])

        # Only one decrement: 1 -> 0.
        assert block.ref_cnt == 0
        assert _free_list_blocks(pool).count(block) == 1

    def test_mixed_unique_and_duplicate(self):
        """A list with both unique and duplicated blocks is handled
        correctly."""
        pool = _make_pool()

        b1, b2 = pool.get_new_blocks(2)
        assert b1.ref_cnt == 1
        assert b2.ref_cnt == 1

        free_before = pool.get_num_free_blocks()

        # b1 appears twice, b2 appears once.
        pool.free_blocks([b1, b2, b1])

        assert b1.ref_cnt == 0
        assert b2.ref_cnt == 0
        assert pool.get_num_free_blocks() == free_before + 2

        free_blocks = _free_list_blocks(pool)
        assert free_blocks.count(b1) == 1
        assert free_blocks.count(b2) == 1

    def test_null_block_never_freed(self):
        """Null blocks must never be appended to the free queue, even
        when passed as duplicates."""
        pool = _make_pool()

        null_block = pool.null_block
        assert null_block.is_null

        free_before = pool.get_num_free_blocks()

        # Passing null block (even duplicated) should be a no-op for
        # the free queue.
        null_block.ref_cnt = 2
        pool.free_blocks([null_block, null_block])

        # Dedup: ref_cnt decremented once (2 -> 1).
        assert null_block.ref_cnt == 1
        assert pool.get_num_free_blocks() == free_before

    def test_generator_input(self):
        """free_blocks() accepts any Iterable, including generators."""
        pool = _make_pool()

        block = pool.get_new_blocks(1)[0]
        assert block.ref_cnt == 1

        def gen():
            yield block
            yield block

        pool.free_blocks(gen())

        assert block.ref_cnt == 0
        assert _free_list_blocks(pool).count(block) == 1
