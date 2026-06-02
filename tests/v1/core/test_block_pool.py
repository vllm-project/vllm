# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.core.block_pool import BlockPool, CompactBlockPool

pytestmark = pytest.mark.cpu_test


def test_block_pool_reserves_zero_as_null_block():
    block_pool = BlockPool(
        num_gpu_blocks=4,
        enable_caching=False,
        hash_block_size=16,
    )

    assert block_pool.null_block.block_id == 0
    assert block_pool.null_block.is_null
    assert block_pool.get_num_free_blocks() == 3
    assert block_pool.get_usage() == 0


def test_block_pool_allocation_never_returns_null_block():
    block_pool = BlockPool(
        num_gpu_blocks=4,
        enable_caching=False,
        hash_block_size=16,
    )

    blocks = block_pool.get_new_blocks(3)

    assert {block.block_id for block in blocks} == {1, 2, 3}
    assert all(not block.is_null for block in blocks)
    assert block_pool.get_num_free_blocks() == 0
    assert block_pool.get_usage() == 1


def test_block_pool_exhaustion_raises_without_allocating_null_block():
    block_pool = BlockPool(
        num_gpu_blocks=2,
        enable_caching=False,
        hash_block_size=16,
    )

    with pytest.raises(ValueError, match="Cannot get 2 free blocks"):
        block_pool.get_new_blocks(2)

    block = block_pool.get_new_blocks(1)[0]
    assert block.block_id == 1
    assert block_pool.null_block.block_id == 0


def test_block_pool_free_returns_blocks_but_not_null_block():
    block_pool = BlockPool(
        num_gpu_blocks=3,
        enable_caching=False,
        hash_block_size=16,
    )
    blocks = block_pool.get_new_blocks(2)

    block_pool.free_blocks(reversed(blocks))

    assert block_pool.get_num_free_blocks() == 2
    assert block_pool.get_usage() == 0
    reallocated = block_pool.get_new_blocks(2)
    assert {block.block_id for block in reallocated} == {1, 2}


@pytest.mark.parametrize(
    "pool",
    [
        BlockPool(num_gpu_blocks=4, enable_caching=False, hash_block_size=16),
        CompactBlockPool(num_allocatable=3),
    ],
)
def test_block_pool_protocol_conformance(pool):
    assert pool.num_gpu_blocks == 4
    assert pool.null_block.block_id == 0
    assert pool.null_block.is_null

    blocks = pool.get_new_blocks(1)

    assert len(blocks) == 1
    assert blocks[0].block_id != 0
    assert not blocks[0].is_null
    pool.free_blocks(blocks)


def test_compact_block_pool_reserves_zero_as_null_block():
    block_pool = CompactBlockPool(num_allocatable=3)

    assert block_pool.num_gpu_blocks == 4
    assert block_pool.null_block.block_id == 0
    assert block_pool.null_block.is_null
    assert block_pool.get_num_free_blocks() == 3
    assert block_pool.get_usage() == 0


def test_compact_block_pool_allocation_never_returns_null_block():
    block_pool = CompactBlockPool(num_allocatable=3)

    blocks = block_pool.get_new_blocks(3)

    assert {block.block_id for block in blocks} == {1, 2, 3}
    assert all(not block.is_null for block in blocks)
    assert all(block.ref_cnt == 1 for block in blocks)
    assert block_pool.get_num_free_blocks() == 0
    assert block_pool.get_usage() == 1


def test_compact_block_pool_zero_arg_operations_are_noops():
    block_pool = CompactBlockPool(num_allocatable=1)

    assert block_pool.get_new_blocks(0) == []
    block_pool.free_blocks([])

    assert block_pool.get_num_free_blocks() == 1
    assert block_pool.get_usage() == 0


def test_compact_block_pool_exhaustion_matches_shared_pool_error_type():
    block_pool = CompactBlockPool(num_allocatable=1)

    with pytest.raises(ValueError, match="Cannot get 2 free blocks"):
        block_pool.get_new_blocks(2)

    block = block_pool.get_new_blocks(1)[0]
    assert block.block_id == 1
    assert block_pool.null_block.block_id == 0


def test_compact_block_pool_free_returns_blocks_to_pool():
    block_pool = CompactBlockPool(num_allocatable=2)
    blocks = block_pool.get_new_blocks(2)

    block_pool.free_blocks(reversed(blocks))

    assert all(block.ref_cnt == 0 for block in blocks)
    assert block_pool.get_num_free_blocks() == 2
    assert block_pool.get_usage() == 0
    reallocated = block_pool.get_new_blocks(2)
    assert {block.block_id for block in reallocated} == {1, 2}
    assert all(block.ref_cnt == 1 for block in reallocated)


def test_compact_block_pool_rejects_freeing_null_block():
    block_pool = CompactBlockPool(num_allocatable=1)

    with pytest.raises(AssertionError, match="null block must never be freed"):
        block_pool.free_blocks([block_pool.null_block])


def test_compact_block_pool_rejects_freeing_unallocated_block():
    block_pool = CompactBlockPool(num_allocatable=1)
    block = block_pool.get_new_blocks(1)[0]
    block_pool.free_blocks([block])

    with pytest.raises(AssertionError, match="binary ref_cnt semantics"):
        block_pool.free_blocks([block])


def test_compact_block_pool_allows_zero_allocatable_blocks():
    block_pool = CompactBlockPool(num_allocatable=0)

    assert block_pool.num_gpu_blocks == 1
    assert block_pool.null_block.block_id == 0
    assert block_pool.get_num_free_blocks() == 0
    assert block_pool.get_usage() == 0
    assert block_pool.get_new_blocks(0) == []
    with pytest.raises(ValueError, match="Cannot get 1 free blocks"):
        block_pool.get_new_blocks(1)
