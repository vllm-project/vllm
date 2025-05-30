# SPDX-License-Identifier: Apache-2.0

import torch

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHashType, KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager
from vllm.v1.kv_cache_interface import SlidingWindowSpec


def get_sliding_window_manager(sliding_window_spec, block_pool):
    return SlidingWindowManager(sliding_window_spec,
                                block_pool,
                                use_eagle=False,
                                num_kv_cache_groups=1,
                                caching_hash_fn=lambda x: x)


def test_sliding_window_possible_cached_prefix():
    block_size = 2
    sliding_window_spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=4,
        use_mla=False,
    )

    block_pool = BlockPool(num_gpu_blocks=100, enable_caching=True)
    manager = get_sliding_window_manager(sliding_window_spec, block_pool)

    def run_one_case(block_is_cached, expect_length):
        block_hash_list = [
            BlockHashType(i, ()) for i in range(len(block_is_cached))
        ]

        block_pool.cached_block_hash_to_block.clear()

        # Mock the block pool with the cached blocks
        for i, (block_hash,
                is_cached) in enumerate(zip(block_hash_list, block_is_cached)):
            if is_cached:
                block_pool.cached_block_hash_to_block[block_hash] = {
                    i: block_pool.blocks[i + 10]
                }

        computed_blocks = manager.find_longest_cache_hit(
            block_hash_list,
            len(block_hash_list) * block_size)
        assert len(computed_blocks) == expect_length

        assert all(block == block_pool.null_block
                   for block in computed_blocks[:expect_length - 2])
        for i in range(2):
            if i < expect_length:
                block_index = expect_length - i - 1
                assert computed_blocks[
                    block_index].block_id == block_index + 10

    run_one_case([False] * 10, 0)
    run_one_case([True], 1)
    run_one_case([True, False], 1)
    run_one_case([True, True], 2)
    run_one_case([True, True, False], 2)
    run_one_case([True, True, True], 3)
    run_one_case([True, True, True, False], 3)
    run_one_case([
        True, True, False, True, False, False, True, True, False, True, True,
        True
    ], 12)
    run_one_case([
        True, True, False, True, False, False, True, True, False, False, False
    ], 8)
    run_one_case([
        True, True, False, True, False, False, True, True, False, False, False,
        True
    ], 8)


def test_sliding_window_remove_skipped_blocks():
    sliding_window_spec = SlidingWindowSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=4,
        use_mla=False,
    )

    block_pool = BlockPool(num_gpu_blocks=2000, enable_caching=True)

    manager = get_sliding_window_manager(sliding_window_spec, block_pool)

    null_block_id = block_pool.null_block.block_id

    def id_to_block_table(ids):
        return [
            KVCacheBlock(id_)
            if id_ != null_block_id else block_pool.null_block for id_ in ids
        ]

    def assert_block_id(block_table, ids):
        for block, id_ in zip(block_table, ids):
            if id_ == null_block_id:
                assert block == block_pool.null_block
            else:
                assert block.block_id == id_

    original_block_ids = [
        1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010
    ]
    block_table = id_to_block_table(original_block_ids)
    manager.req_to_blocks["test"] = block_table

    manager.remove_skipped_blocks("test", 0)
    assert_block_id(block_table, original_block_ids)

    # 4 tokens are computed. Only token 0 is out of the sliding window. As
    # block 1000 also contains token 1 that is in the sliding window, block 1000
    # cannot be removed.
    manager.remove_skipped_blocks("test", 4)
    assert_block_id(block_table, original_block_ids)

    # 5 tokens are computed. Token 0 & 1 are out of the sliding window.
    # Block 1000 can be removed.
    manager.remove_skipped_blocks("test", 5)
    assert_block_id(block_table, [null_block_id] + original_block_ids[1:])

    # 6 tokens are computed. Token 0-2 are out of the sliding window.
    # Cannot remove new block as the block 1001 is still used by token 3.
    manager.remove_skipped_blocks("test", 6)
    assert_block_id(block_table, [null_block_id] + original_block_ids[1:])

    # 7 tokens are computed. Token 0-3 are out of the sliding window.
    # Block 1001 can be removed and block 1000 is already removed.
    manager.remove_skipped_blocks("test", 7)
    assert_block_id(block_table, [null_block_id] * 2 + original_block_ids[2:])

    # 11 tokens are computed. Token 0-7 are out of the sliding window.
    # Block 1002 & 1003 can be removed now. Block 1003 represents a longer
    # sequence, and is expected to be evicted earlier than 1002, so the order
    # of removed blocks should be [1003, 1002].
    manager.remove_skipped_blocks("test", 11)
    assert_block_id(block_table, [null_block_id] * 4 + original_block_ids[4:])
