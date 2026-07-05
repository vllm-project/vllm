# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deferral fence for mid-flight skipped-block frees (BlockPool)."""

from vllm.v1.core.block_pool import BlockPool


def _make_pool() -> BlockPool:
    return BlockPool(num_gpu_blocks=100, enable_caching=True, hash_block_size=4)


def test_immediate_free_when_deferral_disabled():
    pool = _make_pool()
    assert pool.defer_skipped_free is False
    free0 = pool.get_num_free_blocks()

    blocks = pool.get_new_blocks(3)
    assert pool.get_num_free_blocks() == free0 - 3

    pool.free_blocks_maybe_deferred(blocks)
    assert pool.get_num_free_blocks() == free0
    assert len(pool.deferred_skipped_frees) == 0


def test_deferred_free_holds_until_fence_processed():
    pool = _make_pool()
    pool.defer_skipped_free = True
    free0 = pool.get_num_free_blocks()

    blocks = pool.get_new_blocks(2)
    assert pool.get_num_free_blocks() == free0 - 2

    # Freed mid-flight while the reader forward (step 5) is still in-flight.
    pool.skipped_free_fence = 5
    pool.free_blocks_maybe_deferred(blocks)

    # Held out of the pool with ref_cnt intact until the reader step is processed.
    assert pool.get_num_free_blocks() == free0 - 2
    assert len(pool.deferred_skipped_frees) == 1
    assert all(b.ref_cnt == 1 for b in blocks)

    pool.drain_skipped_frees(processed_step_seq=4)
    assert pool.get_num_free_blocks() == free0 - 2

    pool.drain_skipped_frees(processed_step_seq=5)
    assert pool.get_num_free_blocks() == free0
    assert len(pool.deferred_skipped_frees) == 0
    assert all(b.ref_cnt == 0 for b in blocks)


def test_drain_respects_monotonic_fences():
    pool = _make_pool()
    pool.defer_skipped_free = True
    free0 = pool.get_num_free_blocks()

    pool.skipped_free_fence = 3
    pool.free_blocks_maybe_deferred(pool.get_new_blocks(1))
    pool.skipped_free_fence = 7
    pool.free_blocks_maybe_deferred(pool.get_new_blocks(1))
    assert pool.get_num_free_blocks() == free0 - 2

    # Draining at 5 releases only the fence<=5 entry; the fence=7 one stays.
    pool.drain_skipped_frees(processed_step_seq=5)
    assert pool.get_num_free_blocks() == free0 - 1
    assert len(pool.deferred_skipped_frees) == 1

    pool.drain_skipped_frees(processed_step_seq=7)
    assert pool.get_num_free_blocks() == free0
    assert len(pool.deferred_skipped_frees) == 0
