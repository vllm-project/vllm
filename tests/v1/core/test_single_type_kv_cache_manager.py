# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random

import pytest
import torch

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    KVCacheBlock,
    make_block_hash_with_group_id,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    ChunkedLocalAttentionManager,
    RSWAManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    RSWASpec,
    SlidingWindowSpec,
)

pytestmark = pytest.mark.cpu_test


def get_sliding_window_manager(sliding_window_spec, block_pool, enable_caching=True):
    # Tests don't exercise admission gating; pass a large cap that is a no-op.
    return SlidingWindowManager(
        sliding_window_spec,
        block_pool=block_pool,
        enable_caching=enable_caching,
        kv_cache_group_id=0,
        scheduler_block_size=sliding_window_spec.block_size,
        max_admission_blocks_per_request=10**9,
    )


def get_chunked_local_attention_manager(
    chunked_local_attention_spec, block_pool, enable_caching=True
):
    return ChunkedLocalAttentionManager(
        chunked_local_attention_spec,
        block_pool=block_pool,
        enable_caching=enable_caching,
        kv_cache_group_id=0,
        scheduler_block_size=chunked_local_attention_spec.block_size,
        max_admission_blocks_per_request=10**9,
    )


def _legacy_sliding_window_find_longest_cache_hit(
    block_hashes,
    max_length,
    kv_cache_group_ids,
    block_pool,
    kv_cache_spec,
    drop_eagle_block,
    alignment_tokens,
) -> tuple[list[KVCacheBlock], ...]:
    sliding_window_contiguous_blocks = (
        SlidingWindowManager._contiguous_blocks_for_hit(
            kv_cache_spec.sliding_window,
            kv_cache_spec.block_size,
            drop_eagle_block,
        )
    )
    sliding_window_contiguous_blocks = max(1, sliding_window_contiguous_blocks)
    max_num_blocks = max_length // kv_cache_spec.block_size
    computed_blocks = tuple(
        [block_pool.null_block] * max_num_blocks
        for _ in range(len(kv_cache_group_ids))
    )
    block_size = kv_cache_spec.block_size
    num_contiguous_blocks = 0
    match_found = False
    for i in range(max_num_blocks - 1, -1, -1):
        if cached_block := block_pool.get_cached_block(
            block_hashes[i], kv_cache_group_ids
        ):
            if num_contiguous_blocks == 0 and block_size != alignment_tokens:
                post_pop_blocks = i if drop_eagle_block else i + 1
                if (post_pop_blocks * block_size) % alignment_tokens != 0:
                    continue
            for computed, cached in zip(computed_blocks, cached_block):
                computed[i] = cached
            num_contiguous_blocks += 1
            if num_contiguous_blocks >= sliding_window_contiguous_blocks:
                for computed in computed_blocks:
                    del computed[i + num_contiguous_blocks :]
                match_found = True
                break
        else:
            num_contiguous_blocks = 0
    if not match_found:
        for computed in computed_blocks:
            del computed[num_contiguous_blocks:]
        while (
            block_size != alignment_tokens
            and len(computed_blocks[0]) * block_size % alignment_tokens != 0
        ):
            for computed in computed_blocks:
                computed.pop()
    if drop_eagle_block and computed_blocks[0]:
        for computed in computed_blocks:
            computed.pop()
        while (
            block_size != alignment_tokens
            and len(computed_blocks[0]) * block_size % alignment_tokens != 0
        ):
            for computed in computed_blocks:
                computed.pop()
    return computed_blocks


def _sliding_window_cache_hit_for_mask(
    mask: list[bool],
    sliding_window_spec: SlidingWindowSpec,
    block_pool: BlockPool,
    drop_eagle_block: bool,
    alignment_tokens: int,
    use_legacy: bool = False,
) -> tuple[list[KVCacheBlock], ...]:
    block_hash_list = [BlockHash(str(i).encode()) for i in range(len(mask))]

    block_pool.cached_block_hash_to_block._cache.clear()
    for i, (block_hash, is_cached) in enumerate(zip(block_hash_list, mask)):
        if is_cached:
            block_pool.cached_block_hash_to_block.insert(
                make_block_hash_with_group_id(block_hash, 0),
                block_pool.blocks[i + 10],
            )

    find_longest_cache_hit = (
        _legacy_sliding_window_find_longest_cache_hit
        if use_legacy
        else SlidingWindowManager.find_longest_cache_hit
    )
    return find_longest_cache_hit(
        block_hashes=block_hash_list,
        max_length=len(block_hash_list) * sliding_window_spec.block_size,
        kv_cache_group_ids=[0],
        block_pool=block_pool,
        kv_cache_spec=sliding_window_spec,
        drop_eagle_block=drop_eagle_block,
        alignment_tokens=alignment_tokens,
    )


def _block_ids_by_group(blocks_by_group: tuple[list[KVCacheBlock], ...]):
    return [[block.block_id for block in blocks] for blocks in blocks_by_group]


def test_chunked_local_attention_possible_cached_prefix():
    block_size = 2
    chunked_local_attention_spec = ChunkedLocalAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        attention_chunk_size=4,
    )

    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=True, hash_block_size=block_size
    )
    manager = get_chunked_local_attention_manager(
        chunked_local_attention_spec, block_pool
    )

    def run_one_case(block_is_cached, tail_token, expect_length):
        block_hash_list = [
            BlockHash(str(i).encode()) for i in range(len(block_is_cached))
        ]

        block_pool.cached_block_hash_to_block._cache.clear()

        # Mock the block pool with the cached blocks
        for i, (block_hash, is_cached) in enumerate(
            zip(block_hash_list, block_is_cached)
        ):
            if is_cached:
                block_pool.cached_block_hash_to_block.insert(
                    make_block_hash_with_group_id(block_hash, 0),
                    block_pool.blocks[i + 10],
                )

        computed_blocks = manager.find_longest_cache_hit(
            block_hashes=block_hash_list,
            max_length=len(block_hash_list) * block_size + tail_token,
            kv_cache_group_ids=[0],
            block_pool=block_pool,
            kv_cache_spec=chunked_local_attention_spec,
            drop_eagle_block=False,
            alignment_tokens=block_size,
        )[0]
        assert len(computed_blocks) == expect_length

        assert all(
            block == block_pool.null_block
            for block in computed_blocks[: (expect_length - 1) // 2]
        )

    run_one_case([True], 0, 1)
    run_one_case([True], 1, 1)
    run_one_case([True, False], 0, 2)
    run_one_case([True, False], 1, 2)
    run_one_case([True, True], 0, 2)
    run_one_case([True, True], 1, 2)
    run_one_case([True, True, False], 0, 2)
    run_one_case([True, True, False], 1, 2)
    run_one_case([True, True, True], 0, 3)
    run_one_case([True, True, True], 1, 3)
    run_one_case([True, True, True, False], 0, 4)
    run_one_case([True, True, True, False], 1, 4)
    run_one_case([random.choice([True, False])] * 8 + [True], 1, 9)
    run_one_case([random.choice([True, False])] * 8 + [False], 1, 8)
    run_one_case([random.choice([True, False])] * 8 + [True, True], 1, 10)
    run_one_case([random.choice([True, False])] * 8 + [True, False], 0, 10)
    run_one_case([random.choice([True, False])] * 8 + [True, False], 1, 10)
    run_one_case([random.choice([True, False])] * 8 + [False, True], 0, 10)
    run_one_case([random.choice([True, False])] * 8 + [False, True], 1, 10)
    run_one_case([random.choice([True, False])] * 8 + [False, False], 0, 10)
    run_one_case([random.choice([True, False])] * 8 + [False, False], 1, 10)


def test_sliding_window_possible_cached_prefix():
    block_size = 2
    sliding_window_spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=4,
    )

    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=True, hash_block_size=block_size
    )
    manager = get_sliding_window_manager(sliding_window_spec, block_pool)

    def run_one_case(block_is_cached, expect_length):
        block_hash_list = [
            BlockHash(str(i).encode()) for i in range(len(block_is_cached))
        ]

        block_pool.cached_block_hash_to_block._cache.clear()

        # Mock the block pool with the cached blocks
        for i, (block_hash, is_cached) in enumerate(
            zip(block_hash_list, block_is_cached)
        ):
            if is_cached:
                block_pool.cached_block_hash_to_block.insert(
                    make_block_hash_with_group_id(block_hash, 0),
                    block_pool.blocks[i + 10],
                )

        computed_blocks = manager.find_longest_cache_hit(
            block_hashes=block_hash_list,
            max_length=len(block_hash_list) * block_size,
            kv_cache_group_ids=[0],
            block_pool=block_pool,
            kv_cache_spec=sliding_window_spec,
            drop_eagle_block=False,
            alignment_tokens=block_size,
        )[0]
        assert len(computed_blocks) == expect_length

        assert all(
            block == block_pool.null_block
            for block in computed_blocks[: expect_length - 2]
        )
        for i in range(2):
            if i < expect_length:
                block_index = expect_length - i - 1
                assert computed_blocks[block_index].block_id == block_index + 10

    run_one_case([False] * 10, 0)
    run_one_case([True], 1)
    run_one_case([True, False], 1)
    run_one_case([True, True], 2)
    run_one_case([True, True, False], 2)
    run_one_case([True, True, True], 3)
    run_one_case([True, True, True, False], 3)
    run_one_case(
        [True, True, False, True, False, False, True, True, False, True, True, True], 12
    )
    run_one_case(
        [True, True, False, True, False, False, True, True, False, False, False], 8
    )
    run_one_case(
        [True, True, False, True, False, False, True, True, False, False, False, True],
        8,
    )


@pytest.mark.parametrize(
    ("block_size", "sliding_window", "alignment_tokens", "drop_eagle_block"),
    [
        (2, 4, 2, False),
        (2, 4, 4, False),
        (2, 4, 4, True),
        (4, 12, 8, False),
        (8, 128, 64, False),
    ],
)
def test_sliding_window_cache_hit_matches_legacy_scan(
    block_size, sliding_window, alignment_tokens, drop_eagle_block
):
    sliding_window_spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=sliding_window,
    )
    block_pool = BlockPool(
        num_gpu_blocks=1000, enable_caching=True, hash_block_size=block_size
    )
    rng = random.Random(0)

    masks = [
        [],
        [False] * 32,
        [True] * 32,
        [True, False] * 16,
        [False, True] * 16,
        [True, True, False, True, False, False, True, True, False, True],
    ]
    masks.extend(
        [rng.choice([False, True]) for _ in range(rng.randrange(1, 64))]
        for _ in range(200)
    )

    for mask in masks:
        actual = _sliding_window_cache_hit_for_mask(
            mask,
            sliding_window_spec,
            block_pool,
            drop_eagle_block,
            alignment_tokens,
        )
        expected = _sliding_window_cache_hit_for_mask(
            mask,
            sliding_window_spec,
            block_pool,
            drop_eagle_block,
            alignment_tokens,
            use_legacy=True,
        )
        assert _block_ids_by_group(actual) == _block_ids_by_group(expected)


def test_sliding_window_cache_hit_skips_windows_on_miss(monkeypatch):
    block_size = 4
    sliding_window_spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=32,
    )
    block_pool = BlockPool(
        num_gpu_blocks=1000, enable_caching=True, hash_block_size=block_size
    )
    block_hash_list = [BlockHash(str(i).encode()) for i in range(64)]
    contiguous_blocks = max(
        1,
        SlidingWindowManager._contiguous_blocks_for_hit(
            sliding_window_spec.sliding_window,
            sliding_window_spec.block_size,
            use_eagle=False,
        ),
    )
    expected_lookups = (len(block_hash_list) + contiguous_blocks - 1) // (
        contiguous_blocks
    )

    num_lookups = 0
    original_get_cached_block = block_pool.get_cached_block

    def counting_get_cached_block(*args, **kwargs):
        nonlocal num_lookups
        num_lookups += 1
        return original_get_cached_block(*args, **kwargs)

    monkeypatch.setattr(block_pool, "get_cached_block", counting_get_cached_block)
    computed_blocks = SlidingWindowManager.find_longest_cache_hit(
        block_hashes=block_hash_list,
        max_length=len(block_hash_list) * block_size,
        kv_cache_group_ids=[0],
        block_pool=block_pool,
        kv_cache_spec=sliding_window_spec,
        drop_eagle_block=False,
        alignment_tokens=block_size,
    )

    assert computed_blocks == ([],)
    assert num_lookups == expected_lookups
    assert num_lookups < len(block_hash_list)


def test_chunked_local_attention_remove_skipped_blocks():
    attention_spec = ChunkedLocalAttentionSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        attention_chunk_size=4,
    )

    block_pool = BlockPool(num_gpu_blocks=2000, enable_caching=True, hash_block_size=2)

    manager = get_chunked_local_attention_manager(attention_spec, block_pool)

    null_block_id = block_pool.null_block.block_id

    def id_to_block_table(ids) -> list[KVCacheBlock]:
        return [
            KVCacheBlock(id_) if id_ != null_block_id else block_pool.null_block
            for id_ in ids
        ]

    def assert_block_id(block_table: list[KVCacheBlock], ids: list[int]):
        for block, id_ in zip(block_table, ids):
            if id_ == null_block_id:
                assert block == block_pool.null_block
            else:
                assert block.block_id == id_

    original_block_ids = [
        1000,
        1001,
        1002,
        1003,
        1004,
        1005,
        1006,
        1007,
        1008,
        1009,
        1010,
    ]
    block_table = id_to_block_table(original_block_ids)
    manager.req_to_blocks["test"] = block_table

    manager.remove_skipped_blocks("test", 0)
    assert_block_id(block_table, original_block_ids)

    # For 4th token (0-indexed), token 0-3 is out of the local attention window.
    manager.remove_skipped_blocks("test", 4)
    assert_block_id(block_table, [null_block_id] * 2)

    # For 6th token (0-indexed), token 4 - 6 are in local attention window,
    # token 0 - 3 are out, 2 blocks can be removed.
    manager.remove_skipped_blocks("test", 6)
    assert_block_id(block_table, [null_block_id] * 2 + original_block_ids[2:])
    # For 12th token (0-indexed),
    # token 0-11 are out, 6 block can be removed.
    manager.remove_skipped_blocks("test", 12)
    assert_block_id(block_table, [null_block_id] * 6)


def test_sliding_window_remove_skipped_blocks():
    sliding_window_spec = SlidingWindowSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=4,
    )

    block_pool = BlockPool(num_gpu_blocks=2000, enable_caching=True, hash_block_size=2)

    manager = get_sliding_window_manager(sliding_window_spec, block_pool)

    null_block_id = block_pool.null_block.block_id

    def id_to_block_table(ids) -> list[KVCacheBlock]:
        return [
            KVCacheBlock(id_) if id_ != null_block_id else block_pool.null_block
            for id_ in ids
        ]

    def assert_block_id(block_table: list[KVCacheBlock], ids: list[int]):
        for block, id_ in zip(block_table, ids):
            if id_ == null_block_id:
                assert block == block_pool.null_block
            else:
                assert block.block_id == id_

    original_block_ids = [
        1000,
        1001,
        1002,
        1003,
        1004,
        1005,
        1006,
        1007,
        1008,
        1009,
        1010,
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


def test_rswa_remove_skipped_blocks_gap_range():
    block_size = 4
    rswa_spec = RSWASpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        rswa_window=8,
    )
    block_pool = BlockPool(num_gpu_blocks=2000, enable_caching=True, hash_block_size=4)
    manager = RSWAManager(
        rswa_spec,
        block_pool=block_pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=block_size,
    )

    null_block_id = block_pool.null_block.block_id
    original_block_ids = list(range(1000, 1010))
    block_table = [
        KVCacheBlock(id_) if id_ != null_block_id else block_pool.null_block
        for id_ in original_block_ids
    ]
    manager.req_to_blocks["test"] = block_table

    prefix_len = 16

    # Without num_prompt_tokens, R-SWA does not evict gap blocks.
    manager.remove_skipped_blocks("test", 28)
    assert [b.block_id for b in block_table] == original_block_ids

    # Gap = block 4 only (tokens [16, 20) fall in the gap).
    manager.remove_skipped_blocks("test", 28, num_prompt_tokens=prefix_len)
    expected = original_block_ids.copy()
    expected[4] = null_block_id
    assert [b.block_id for b in block_table] == expected

    # Window moves: blocks 5 and 6 also enter the gap; block 4 is already null.
    manager.remove_skipped_blocks("test", 36, num_prompt_tokens=prefix_len)
    expected[5] = null_block_id
    expected[6] = null_block_id
    assert [b.block_id for b in block_table] == expected


def test_get_num_blocks_to_allocate():
    block_size = 2
    sliding_window_spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=4,  # Placeholder value, not related to test result
    )

    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=True, hash_block_size=block_size
    )
    manager = get_sliding_window_manager(sliding_window_spec, block_pool)
    cached_blocks_1 = [KVCacheBlock(i + 1) for i in range(10)]
    cached_blocks_2 = [block_pool.null_block for _ in range(5)] + [
        KVCacheBlock(i + 1) for i in range(5)
    ]

    assert (
        manager.get_num_blocks_to_allocate(
            "1", 20 * block_size, cached_blocks_1, 0, 20 * block_size
        )
        == 20
    )
    assert (
        manager.get_num_blocks_to_allocate(
            "2", 20 * block_size, cached_blocks_2, 0, 20 * block_size
        )
        == 15
    )


def test_evictable_cached_blocks_not_double_allocated():
    block_size = 2
    sliding_window_length = 2 * block_size
    sliding_window_spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=sliding_window_length,
    )

    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=True, hash_block_size=block_size
    )
    manager = get_sliding_window_manager(sliding_window_spec, block_pool)

    request_id = "req"
    evictable_block = block_pool.blocks[1]  # ref_cnt == 0, eviction candidate

    num_blocks_to_allocate = manager.get_num_blocks_to_allocate(
        request_id=request_id,
        num_tokens=2 * block_size,
        new_computed_blocks=[evictable_block],
        total_computed_tokens=block_size,
        num_tokens_main_model=2 * block_size,
    )
    # Free capacity check should count evictable cached blocks, but allocation
    # should only allocate the truly new block.
    assert num_blocks_to_allocate == 2

    manager.add_local_computed_blocks(
        request_id,
        [evictable_block],
        num_local_computed_tokens=block_size,
        num_external_computed_tokens=0,
    )
    new_blocks = manager.allocate_new_blocks(
        request_id, num_tokens=4, num_tokens_main_model=4
    )
    assert len(new_blocks) == 1
    assert len(manager.req_to_blocks[request_id]) == 2


def test_chunked_local_attention_get_num_blocks_to_allocate():
    block_size = 2
    attention_spec = ChunkedLocalAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        attention_chunk_size=4,  # Placeholder value, not related to test result
    )

    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=True, hash_block_size=block_size
    )
    manager = get_chunked_local_attention_manager(attention_spec, block_pool)
    cached_blocks_1 = [KVCacheBlock(i + 1) for i in range(10)]
    cached_blocks_2 = [block_pool.null_block for _ in range(5)] + [
        KVCacheBlock(i + 1) for i in range(5)
    ]

    assert (
        manager.get_num_blocks_to_allocate(
            "1", 20 * block_size, cached_blocks_1, 0, 20 * block_size
        )
        == 20
    )
    assert (
        manager.get_num_blocks_to_allocate(
            "2", 20 * block_size, cached_blocks_2, 0, 20 * block_size
        )
        == 15
    )


def test_predictor_matches_allocator_blocks_calculation_with_admission_cap():
    """In forward steps, `get_num_blocks_to_allocate` must return exactly what
    `allocate_new_blocks` will pull; otherwise `block_pool.get_new_blocks`
    raises `ValueError: Cannot get N free blocks from the pool`.
    """
    block_size = 2
    sliding_window = 8  # 4-block live window
    cap = sliding_window // block_size

    spec = SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
        sliding_window=sliding_window,
    )
    block_pool = BlockPool(
        num_gpu_blocks=100, enable_caching=True, hash_block_size=block_size
    )
    manager = SlidingWindowManager(
        spec,
        block_pool=block_pool,
        enable_caching=False,
        kv_cache_group_id=0,
        scheduler_block_size=spec.block_size,
        max_admission_blocks_per_request=cap,
    )

    request_id = "req"
    total_computed = 0
    # Walk through request forward steps. Check num_blocks returned by
    # `get_num_blocks_to_allocate` matches what `allocate_new_blocks` pulls
    for num_tokens in (4, 8, 12, 16):
        predicted = manager.get_num_blocks_to_allocate(
            request_id=request_id,
            num_tokens=num_tokens,
            new_computed_blocks=[],
            total_computed_tokens=total_computed,
            num_tokens_main_model=num_tokens,
        )
        new_blocks = manager.allocate_new_blocks(
            request_id, num_tokens=num_tokens, num_tokens_main_model=num_tokens
        )
        assert predicted == len(new_blocks), (
            f"num_tokens={num_tokens}: predictor returned {predicted} "
            f"but allocator pulled {len(new_blocks)}"
        )
        total_computed = num_tokens
