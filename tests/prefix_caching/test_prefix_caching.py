"""Compare the with and without prefix caching.

Run `pytest tests/prefix_caching/test_prefix_caching.py`.
"""
import pytest

from vllm.core.block_manager import CachedBlockAllocator
from vllm.utils import Device


@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("num_blocks", [16])
def test_block_allocator(
    block_size: int,
    num_blocks: int,
):
    block_hash = 1
    block_allocator = CachedBlockAllocator(Device.CPU, block_size, num_blocks)

    # Allocate two PysicalTokenBlocks with the same hash and check
    # that they are the same PhysicalTokenBlock
    first_block = block_allocator.allocate(block_hash, 0)
    second_block = block_allocator.allocate(block_hash, 0)
    assert (first_block == second_block)
    assert (second_block.ref_count == 2)

    # Free the first_block and confirm that the ref_count is correctly
    # decremented on the second block
    block_allocator.free(first_block)
    assert (second_block.ref_count == 1)

    # Free the second block
    block_allocator.free(second_block)

    # Reallocate the first block and confirm that, even after the block
    # had its ref_count go to 0, we still get the same block back
    first_block = block_allocator.allocate(block_hash, 0)
    assert (first_block == second_block)
    assert (first_block.block_hash == block_hash)


@pytest.mark.parametrize("num_blocks", [16])
def test_eviction(num_blocks: int, ):
    block_size = 16
    block_allocator = CachedBlockAllocator(Device.CPU, block_size, num_blocks)
    blocks = []

    for i in range(num_blocks):
        # use i as the block_hash
        blocks.append(block_allocator.allocate(i, 0))

    #Free all blocks
    for block in blocks:
        block_allocator.free(block)

    # Allocate a new block and confirm that it's the first block freed.
    # I.E The Least Recently Used block
    new_block_hash = block_size
    new_block = block_allocator.allocate(new_block_hash, 0)
    assert (new_block == blocks[0])
    assert (new_block.block_hash == new_block_hash)

    # Reallocate the second in blocks to remove it from the free list
    realloc_block_hash = 1
    realloc_block = block_allocator.allocate(realloc_block_hash, 0)
    assert (realloc_block == blocks[realloc_block_hash])
    assert (realloc_block.block_hash == realloc_block_hash)

    # Allocate a new block and confirm that it's not the realloc_block,
    # since the realloc_block shouldn't be in the free list
    new_block_hash = block_size + 1
    new_block = block_allocator.allocate(new_block_hash, 0)
    assert (realloc_block != new_block)
    assert (new_block.block_hash == new_block_hash)
    assert (new_block.block_number == 2)
