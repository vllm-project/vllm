# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.core.block_pool import BlockPool

pytestmark = pytest.mark.cpu_test


def make_pool_with_free_order(block_ids: list[int]) -> BlockPool:
    pool = BlockPool(
        num_gpu_blocks=len(block_ids) + 1,
        enable_caching=False,
        hash_block_size=16,
    )
    blocks = pool.get_new_blocks(pool.get_num_free_blocks())
    blocks_by_id = {block.block_id: block for block in blocks}
    pool.free_blocks([blocks_by_id[block_id] for block_id in block_ids])
    return pool


@pytest.mark.parametrize(
    ("free_order", "expected"),
    [
        (
            [20, *range(8, 16), 1, 2, *range(18, 15, -1), *range(7, 2, -1), 19],
            list(range(8, 16)),
        ),
        (
            [20, *range(15, 7, -1), 1, 3, 2, *range(4, 8), *range(16, 20)],
            list(range(8, 16)),
        ),
    ],
)
def test_contiguous_alloc_returns_runs_ascending(
    free_order: list[int], expected: list[int]
):
    pool = make_pool_with_free_order(free_order)
    pool.use_contiguous_allocation = True

    blocks = pool.get_new_blocks(8)

    assert [block.block_id for block in blocks] == expected


def test_contiguous_alloc_fills_remainder_from_lru_head():
    free_order = [14, 4, 5, 6, 1, 8, 2, 10, 3, 12, 13, 11, 9, 7]
    pool = make_pool_with_free_order(free_order)
    pool.use_contiguous_allocation = True

    blocks = pool.get_new_blocks(8)

    assert [block.block_id for block in blocks] == [4, 5, 6, 12, 13, 14, 1, 8]


def test_contiguous_alloc_preserves_queue_links_after_free():
    pool = make_pool_with_free_order([12, 4, 5, 6, 7, 8, 9, 10, 11, 3, 2, 1])
    pool.use_contiguous_allocation = True

    blocks = pool.get_new_blocks(8)
    assert pool.get_num_free_blocks() == 4
    assert [
        block.block_id for block in pool.free_block_queue.get_all_free_blocks()
    ] == [12, 3, 2, 1]
    assert all(
        block.prev_free_block is None and block.next_free_block is None
        for block in blocks
    )

    pool.free_blocks(blocks)

    queue = pool.free_block_queue
    free_blocks = queue.get_all_free_blocks()
    assert queue.num_free_blocks == 12
    assert queue.fake_free_list_head.next_free_block is free_blocks[0]
    assert free_blocks[0].prev_free_block is queue.fake_free_list_head
    assert queue.fake_free_list_tail.prev_free_block is free_blocks[-1]
    assert free_blocks[-1].next_free_block is queue.fake_free_list_tail
    assert all(
        left.next_free_block is right and right.prev_free_block is left
        for left, right in zip(free_blocks, free_blocks[1:])
    )


def test_contiguous_alloc_disabled_keeps_lru_order():
    free_order = [12, 4, 5, 6, 7, 8, 9, 10, 11, 3, 2, 1]
    pool = make_pool_with_free_order(free_order)

    blocks = pool.get_new_blocks(8)

    assert [block.block_id for block in blocks] == free_order[:8]


def test_contiguous_alloc_keeps_lru_order_for_small_allocations():
    free_order = [12, 4, 5, 6, 7, 8, 9, 10, 11, 3, 2, 1]
    pool = make_pool_with_free_order(free_order)
    pool.use_contiguous_allocation = True

    blocks = pool.get_new_blocks(7)

    assert [block.block_id for block in blocks] == free_order[:7]


def test_contiguous_alloc_search_is_bounded():
    free_order = list(range(1, 32, 2)) + list(range(2, 33, 2)) + list(range(33, 41))
    pool = make_pool_with_free_order(free_order)
    pool.use_contiguous_allocation = True

    blocks = pool.get_new_blocks(8)

    assert [block.block_id for block in blocks] == list(range(1, 16, 2))


def test_contiguous_alloc_enabled_from_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VLLM_KV_CONTIG_ALLOC", "1")

    pool = make_pool_with_free_order([12, 4, 5, 6, 7, 8, 9, 10, 11, 3, 2, 1])

    assert pool.use_contiguous_allocation
