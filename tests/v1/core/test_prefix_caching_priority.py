import pytest

from vllm.v1.core.block_pool import BlockPool


def test_free_blocks_priority():
    # Create a BlockPool with 5 blocks and prefix caching enabled
    bp = BlockPool(num_gpu_blocks=6, enable_caching=True)
    # Initially, free list should contain all non-null blocks [1,2,3,4]
    initial_free = bp.free_block_queue.get_all_free_blocks()
    initial_ids = [blk.block_id for blk in initial_free]
    assert initial_ids == [1, 2, 3, 4, 5]

    # Allocate 2 blocks for request R0 (to simulate priority 0)
    r0_blocks = bp.get_new_blocks(2)
    # Allocate 2 blocks for request R1 (to simulate priority 1)
    r1_blocks = bp.get_new_blocks(2)
    # Remaining free blocks
    remaining_ids = [
        blk.block_id for blk in bp.free_block_queue.get_all_free_blocks()
    ]
    assert remaining_ids == [5]

    # Free R0 blocks (priority 0: evict before priority 1 blocks)
    # Reverse within request so tail blocks freed first.
    bp.free_blocks(reversed(r0_blocks), front=True)
    # Free R1 blocks (priority 1: evict after priority 0 blocks)
    bp.free_blocks(reversed(r1_blocks))

    # Collect final free list
    final_free = bp.free_block_queue.get_all_free_blocks()
    final_ids = [blk.block_id for blk in final_free]

    # Expected order: R0 blocks at front (in reverse order), then remaining, then R1 blocks at tail
    expected = remaining_ids + [
        r0_blocks[1].block_id, r0_blocks[0].block_id
    ] + [r1_blocks[1].block_id, r1_blocks[0].block_id]
    assert final_ids == expected
