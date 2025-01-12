import pytest

from vllm.core.block.cpu_offloading_block_allocator import (
    CpuOffloadingBlockAllocator)
from vllm.utils import Device, chunk_list


@pytest.mark.parametrize("num_cpu_blocks", [1024])
@pytest.mark.parametrize("num_gpu_blocks", [256])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("allocator_type", ["prefix_caching"])
def test_allocate_mutable_block(num_cpu_blocks: int, num_gpu_blocks: int,
                                block_size: int, allocator_type: str):
    allocator = CpuOffloadingBlockAllocator.create(
        allocator_type=allocator_type,
        num_gpu_blocks=num_gpu_blocks,
        num_cpu_blocks=num_cpu_blocks,
        block_size=block_size,
    )

    assert allocator.get_num_free_blocks(Device.CPU) == num_cpu_blocks
    assert allocator.get_num_free_blocks(Device.GPU) == num_gpu_blocks

    gpu_blocks = [
        allocator.allocate_mutable_block(prev_block=None, device=Device.GPU)
        for _ in range(num_gpu_blocks)
    ]
    assert allocator.get_num_free_blocks(Device.CPU) == num_cpu_blocks
    assert allocator.get_num_free_blocks(Device.GPU) == 0
    assert len(allocator._uncached_blocks) == num_gpu_blocks

    blocks_to_swap_out, blocks_to_swap_in = allocator.get_and_reset_swaps(0.0)
    assert len(blocks_to_swap_out) == 0
    assert len(blocks_to_swap_in) == 0
    assert len(allocator._uncached_blocks) == num_gpu_blocks

    _ = [allocator.free(block) for block in gpu_blocks]
    assert allocator.get_num_free_blocks(Device.CPU) == num_cpu_blocks
    assert allocator.get_num_free_blocks(Device.GPU) == num_gpu_blocks

    blocks_to_swap_out, blocks_to_swap_in = allocator.get_and_reset_swaps(1.0)
    assert len(blocks_to_swap_out) == 0
    assert len(blocks_to_swap_in) == 0
    assert len(allocator._uncached_blocks) == 0


@pytest.mark.parametrize("num_cpu_blocks", [1024])
@pytest.mark.parametrize("num_gpu_blocks", [256])
@pytest.mark.parametrize("block_size", [2])
@pytest.mark.parametrize("allocator_type", ["prefix_caching"])
def test_allocate_immutable_block(num_cpu_blocks: int, num_gpu_blocks: int,
                                  block_size: int, allocator_type: str):
    allocator = CpuOffloadingBlockAllocator.create(
        allocator_type=allocator_type,
        num_gpu_blocks=num_gpu_blocks,
        num_cpu_blocks=num_cpu_blocks,
        block_size=block_size,
    )

    unique_token_ids = list(
        range((num_cpu_blocks + num_gpu_blocks) * block_size))
    gpu_token_ids = list(
        chunk_list(unique_token_ids[:num_gpu_blocks * block_size], block_size))
    gpu_token_ids2 = list(
        chunk_list(
            unique_token_ids[num_gpu_blocks * block_size:2 * num_gpu_blocks *
                             block_size], block_size))

    gpu_blocks = [
        allocator.allocate_immutable_block(prev_block=None,
                                           token_ids=token_ids,
                                           device=Device.GPU)
        for token_ids in gpu_token_ids
    ]

    assert allocator.get_num_free_blocks(Device.CPU) == num_cpu_blocks
    assert allocator.get_num_free_blocks(Device.GPU) == 0
    assert len(allocator._uncached_blocks) == num_gpu_blocks

    blocks_to_swap_out, blocks_to_swap_in = allocator.get_and_reset_swaps(0.0)
    assert len(blocks_to_swap_out) == 0
    assert len(blocks_to_swap_in) == 0
    assert len(allocator._uncached_blocks) == num_gpu_blocks

    allocator.mark_blocks_as_computed([block.block_id for block in gpu_blocks])
    blocks_to_swap_out, blocks_to_swap_in = allocator.get_and_reset_swaps(1.0)
    assert len(blocks_to_swap_out) + len(blocks_to_swap_in) == num_gpu_blocks
    assert len(allocator._uncached_blocks) == 0

    _ = [allocator.free(block) for block in gpu_blocks]
    assert allocator.get_num_free_blocks(Device.CPU) == num_cpu_blocks
    assert allocator.get_num_free_blocks(Device.GPU) == num_gpu_blocks

    blocks_to_swap_out, blocks_to_swap_in = allocator.get_and_reset_swaps(1.0)
    assert len(blocks_to_swap_out) == 0
    assert len(blocks_to_swap_in) == 0
    assert len(allocator._uncached_blocks) == 0

    # allocate another gpu sequence to flush out the GPU cache
    gpu_blocks = [
        allocator.allocate_immutable_block(prev_block=None,
                                           token_ids=token_ids,
                                           device=Device.GPU)
        for token_ids in gpu_token_ids2
    ]

    assert allocator.get_num_free_blocks(Device.CPU) == num_cpu_blocks
    assert allocator.get_num_free_blocks(Device.GPU) == 0
    assert all([
        not allocator._allocators[Device.GPU].block_is_computed(block.block_id)
        for block in gpu_blocks
    ])

    _ = [allocator.free(block) for block in gpu_blocks]
    assert allocator.get_num_free_blocks(Device.GPU) == num_gpu_blocks

    blocks_to_swap_out, blocks_to_swap_in = allocator.get_and_reset_swaps(2.0)
    assert len(blocks_to_swap_out) == 0
    assert len(blocks_to_swap_in) == 0
    assert len(allocator._uncached_blocks) == 0

    # allocate original gpu sequence. It should hit CPU cache.
    gpu_blocks = [
        allocator.allocate_immutable_block(prev_block=None,
                                           token_ids=token_ids,
                                           device=Device.GPU)
        for token_ids in gpu_token_ids
    ]

    delta = num_cpu_blocks - num_gpu_blocks
    assert allocator.get_num_free_blocks(Device.CPU) == delta
    assert allocator.get_num_free_blocks(Device.GPU) == 0
    assert all([
        allocator._allocators[Device.GPU].block_is_computed(block.block_id)
        for block in gpu_blocks
    ])

    blocks_to_swap_out, blocks_to_swap_in = allocator.get_and_reset_swaps(3.0)
    assert allocator.get_num_free_blocks(Device.CPU) == num_cpu_blocks
