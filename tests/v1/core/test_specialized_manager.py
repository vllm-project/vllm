from collections import deque
import torch
from vllm.v1.core.hybrid_cache_manager.specialized_manager import BlockPoolOperations, SlidingWindowManager
from vllm.v1.core.hybrid_cache_manager.utils import PrefixLengthRange
from vllm.v1.core.kv_cache_utils import BlockHashType, KVCacheBlock
from vllm.v1.kv_cache_interface import SlidingWindowSpec


def test_sliding_window_possible_cached_prefix():
    sliding_window_spec = SlidingWindowSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=4,
    )

    block_pool_result = deque()
    null_block = KVCacheBlock(-1, 0)

    def get_cached_block(_block_hash):
        if isinstance(_block_hash,
                      BlockHashType) and _block_hash.hash_value == -1:
            # the dummy block hash
            return None
        is_cached = block_pool_result.popleft()
        if is_cached:
            return 1
        else:
            return None

    def get_null_block():
        return null_block

    manager = SlidingWindowManager(
        sliding_window_spec,
        BlockPoolOperations(get_cached_block, get_null_block))

    block_pool_result.clear()
    block_pool_result.extend([
        True, True, False, True, False, False, True, True, False, True, True,
        True
    ])
    ranges, computed_blocks = manager.get_possible_cached_prefix(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    assert ranges == [
        PrefixLengthRange(0, 4),
        PrefixLengthRange(16, 16),
        PrefixLengthRange(22, 24)
    ]
    assert computed_blocks == [
        1, 1, null_block, 1, null_block, null_block, 1, 1, null_block, 1, 1, 1
    ]


def test_sliding_window_remove_useless_blocks():
    sliding_window_spec = SlidingWindowSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=4,
    )

    def get_cached_block(_block_hash):
        # should not be called
        raise NotImplementedError

    def get_null_block():
        return KVCacheBlock(-1, 0)

    manager = SlidingWindowManager(
        sliding_window_spec,
        BlockPoolOperations(get_cached_block, get_null_block))

    def id_to_block_table(ids):
        return [
            KVCacheBlock(id_, 0) if id_ != -1 else get_null_block()
            for id_ in ids
        ]

    def assert_block_id(block_table, ids):
        for block, id_ in zip(block_table, ids):
            if id_ == -1:
                assert block == get_null_block()
            else:
                assert block.block_id == id_

    block_table = id_to_block_table([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    removed = manager.remove_useless_blocks(block_table, 0)
    assert_block_id(removed, [])
    assert_block_id(block_table, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    removed = manager.remove_useless_blocks(block_table, 5)
    assert_block_id(removed, [])
    assert_block_id(block_table, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    removed = manager.remove_useless_blocks(block_table, 6)
    assert_block_id(removed, [0])
    assert_block_id(block_table, [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    removed = manager.remove_useless_blocks(block_table, 7)
    assert_block_id(removed, [])
    assert_block_id(block_table, [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    removed = manager.remove_useless_blocks(block_table, 8)
    assert_block_id(removed, [1])
    assert_block_id(block_table, [-1, -1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    removed = manager.remove_useless_blocks(block_table, 12)
    assert_block_id(removed, [3, 2])
    assert_block_id(block_table, [-1, -1, -1, -1, 4, 5, 6, 7, 8, 9, 10])
