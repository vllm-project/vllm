# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for BlockPool.free_blocks() duplicate-block handling.
See: https://github.com/vllm-project/vllm/issues/42571
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
    return BlockPool(
        num_gpu_blocks=num_gpu_blocks,
        enable_caching=enable_caching,
        hash_block_size=HASH_BLOCK_SIZE,
    )


def _free_list_blocks(pool: BlockPool) -> list:
    return pool.free_block_queue.get_all_free_blocks()


class TestFreeBlocksDedup:
    def test_duplicate_block_freed_once(self):
        """Same block passed twice: ref_cnt decremented once, freed once."""
        pool = _make_pool()

        block = pool.get_new_blocks(1)[0]
        block.ref_cnt += 1  # simulate second reference
        assert block.ref_cnt == 2

        free_before = pool.get_num_free_blocks()
        pool.free_blocks([block, block])

        assert block.ref_cnt == 1
        assert pool.get_num_free_blocks() == free_before

        # Drop last reference
        pool.free_blocks([block])
        assert block.ref_cnt == 0
        assert pool.get_num_free_blocks() == free_before + 1
        assert _free_list_blocks(pool).count(block) == 1

    def test_duplicate_block_freed_to_zero(self):
        """ref_cnt==1 block passed twice: goes to 0, appended once."""
        pool = _make_pool()

        block = pool.get_new_blocks(1)[0]
        free_before = pool.get_num_free_blocks()

        pool.free_blocks([block, block])

        assert block.ref_cnt == 0
        assert pool.get_num_free_blocks() == free_before + 1
        assert _free_list_blocks(pool).count(block) == 1

    def test_free_list_accounting_consistent(self):
        """num_free_blocks must equal actual linked list length."""
        pool = _make_pool(num_gpu_blocks=4)

        blocks = pool.get_new_blocks(3)
        for b in blocks:
            b.ref_cnt += 1  # ref_cnt=2 each

        duped = [b for b in blocks for _ in range(2)]
        pool.free_blocks(duped)

        for b in blocks:
            assert b.ref_cnt == 1

        free_count = pool.get_num_free_blocks()
        actual_free = len(_free_list_blocks(pool))
        assert free_count == actual_free

    def test_no_regression_unique_blocks(self):
        """Normal case (no duplicates) still works."""
        pool = _make_pool()

        blocks = pool.get_new_blocks(3)
        free_before = pool.get_num_free_blocks()

        pool.free_blocks(blocks)

        for b in blocks:
            assert b.ref_cnt == 0
        assert pool.get_num_free_blocks() == free_before + 3

        free_blocks = _free_list_blocks(pool)
        for b in blocks:
            assert free_blocks.count(b) == 1

    def test_triple_duplicate(self):
        """Three copies deduped to one."""
        pool = _make_pool()

        block = pool.get_new_blocks(1)[0]
        pool.free_blocks([block, block, block])

        assert block.ref_cnt == 0
        assert _free_list_blocks(pool).count(block) == 1

    def test_mixed_unique_and_duplicate(self):
        """Mixed unique + duplicated blocks handled correctly."""
        pool = _make_pool()

        b1, b2 = pool.get_new_blocks(2)
        free_before = pool.get_num_free_blocks()

        pool.free_blocks([b1, b2, b1])

        assert b1.ref_cnt == 0
        assert b2.ref_cnt == 0
        assert pool.get_num_free_blocks() == free_before + 2

        free_blocks = _free_list_blocks(pool)
        assert free_blocks.count(b1) == 1
        assert free_blocks.count(b2) == 1

    def test_null_block_never_freed(self):
        """Null blocks never added to free queue, even as duplicates."""
        pool = _make_pool()

        null_block = pool.null_block
        assert null_block.is_null

        free_before = pool.get_num_free_blocks()
        pool.free_blocks([null_block, null_block])

        # Free queue must not grow and null block must not appear in it.
        assert pool.get_num_free_blocks() == free_before
        assert null_block not in _free_list_blocks(pool)

    def test_generator_input(self):
        """free_blocks() accepts generators."""
        pool = _make_pool()

        block = pool.get_new_blocks(1)[0]

        def gen():
            yield block
            yield block

        pool.free_blocks(gen())

        assert block.ref_cnt == 0
        assert _free_list_blocks(pool).count(block) == 1


class TestSlidingWindowDoubleFreeIntegration:
    """Integration tests that exercise the SlidingWindowManager lifecycle
    that produced the original crash in issue #42571.

    Crash path: allocate → window-shift (remove_skipped_blocks) → reallocate
    (freed block returns to same request) → free (reversed list has duplicates).
    """

    @pytest.fixture()
    def setup(self):
        import torch

        from vllm.v1.core.single_type_kv_cache_manager import (
            SlidingWindowManager,
        )
        from vllm.v1.kv_cache_interface import SlidingWindowSpec

        block_size = 2
        sliding_window = 4  # 2 blocks
        spec = SlidingWindowSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float32,
            sliding_window=sliding_window,
        )
        pool = BlockPool(
            num_gpu_blocks=20,
            enable_caching=False,
            hash_block_size=block_size,
        )
        manager = SlidingWindowManager(
            spec,
            block_pool=pool,
            enable_caching=False,
            kv_cache_group_id=0,
            max_admission_blocks_per_request=10**9,
        )
        return manager, pool, block_size

    def test_sliding_window_free_no_crash(self, setup):
        """Full lifecycle: blocks freed by window shift get reallocated to the
        same request, then free() must not corrupt the free list."""
        manager, pool, block_size = setup
        req_id = "req-sw-1"
        total_free_before = pool.get_num_free_blocks()

        # Step 1: prefill 4 tokens → allocate 2 blocks
        manager.allocate_new_computed_blocks(
            req_id,
            [],
            num_local_computed_tokens=0,
            num_external_computed_tokens=0,
        )
        manager.allocate_new_blocks(req_id, num_tokens=4, num_tokens_main_model=4)
        blocks_after_step1 = list(manager.req_to_blocks[req_id])
        assert len(blocks_after_step1) == 2

        # Step 2: generate tokens 4-5, window shifts → free block at index 0
        manager.remove_skipped_blocks(req_id, 5)
        manager.allocate_new_blocks(req_id, num_tokens=6, num_tokens_main_model=6)

        # Step 3: generate tokens 6-7, window shifts further
        manager.remove_skipped_blocks(req_id, 7)
        manager.allocate_new_blocks(req_id, num_tokens=8, num_tokens_main_model=8)

        # Free the request — this is where the old code would crash with
        # AssertionError on the free list sentinel.
        manager.free(req_id)

        # All non-null blocks must be back in the pool.
        total_free_after = pool.get_num_free_blocks()
        assert total_free_after == total_free_before

        # Free list must be internally consistent.
        actual_free = len(_free_list_blocks(pool))
        assert pool.get_num_free_blocks() == actual_free

    def test_sliding_window_multiple_requests(self, setup):
        """Multiple requests sharing the pool with window shifts
        must not corrupt free-list accounting."""
        manager, pool, block_size = setup

        for i in range(3):
            req_id = f"req-multi-{i}"
            manager.allocate_new_computed_blocks(
                req_id,
                [],
                num_local_computed_tokens=0,
                num_external_computed_tokens=0,
            )
            manager.allocate_new_blocks(req_id, num_tokens=4, num_tokens_main_model=4)
            manager.remove_skipped_blocks(req_id, 5)
            manager.allocate_new_blocks(req_id, num_tokens=6, num_tokens_main_model=6)

        for i in range(3):
            manager.free(f"req-multi-{i}")

        # All blocks returned to pool.
        actual_free = len(_free_list_blocks(pool))
        assert pool.get_num_free_blocks() == actual_free
        assert pool.get_num_free_blocks() == pool.num_gpu_blocks
