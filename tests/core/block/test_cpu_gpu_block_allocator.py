import pytest

from vllm.core.block.cpu_gpu_block_allocator import CpuGpuBlockAllocator
from vllm.utils import Device, chunk_list


@pytest.mark.parametrize("num_cpu_blocks", [0, 512])
@pytest.mark.parametrize("num_gpu_blocks", [1024])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("allocator_type", ["naive", "prefix_caching"])
def test_allocate_mutable_block(num_cpu_blocks: int, num_gpu_blocks: int,
                                block_size: int, allocator_type: str):
    allocator = CpuGpuBlockAllocator.create(
        allocator_type=allocator_type,
        num_gpu_blocks=num_gpu_blocks,
        num_cpu_blocks=num_cpu_blocks,
        block_size=block_size,
    )

    assert allocator.get_num_free_blocks(Device.CPU) == num_cpu_blocks
    assert allocator.get_num_free_blocks(Device.GPU) == num_gpu_blocks

    cpu_blocks = [
        allocator.allocate_mutable_block(prev_block=None, device=Device.CPU)
        for _ in range(num_cpu_blocks)
    ]
    assert allocator.get_num_free_blocks(Device.CPU) == 0
    assert allocator.get_num_free_blocks(Device.GPU) == num_gpu_blocks

    gpu_blocks = [
        allocator.allocate_mutable_block(prev_block=None, device=Device.GPU)
        for _ in range(num_gpu_blocks)
    ]
    assert allocator.get_num_free_blocks(Device.CPU) == 0
    assert allocator.get_num_free_blocks(Device.GPU) == 0

    _ = [allocator.free(block) for block in cpu_blocks]
    assert allocator.get_num_free_blocks(Device.CPU) == num_cpu_blocks
    assert allocator.get_num_free_blocks(Device.GPU) == 0

    _ = [allocator.free(block) for block in gpu_blocks]
    assert allocator.get_num_free_blocks(Device.CPU) == num_cpu_blocks
    assert allocator.get_num_free_blocks(Device.GPU) == num_gpu_blocks


@pytest.mark.parametrize("num_cpu_blocks", [0, 512])
@pytest.mark.parametrize("num_gpu_blocks", [1024])
@pytest.mark.parametrize("block_size", [2])
@pytest.mark.parametrize("allocator_type", ["naive", "prefix_caching"])
def test_allocate_immutable_block(num_cpu_blocks: int, num_gpu_blocks: int,
                                  block_size: int, allocator_type: str):
    allocator = CpuGpuBlockAllocator.create(
        allocator_type=allocator_type,
        num_gpu_blocks=num_gpu_blocks,
        num_cpu_blocks=num_cpu_blocks,
        block_size=block_size,
    )

    unique_token_ids = list(
        range((num_cpu_blocks + num_gpu_blocks) * block_size))
    gpu_token_ids = chunk_list(unique_token_ids[:num_gpu_blocks * block_size],
                               block_size)
    cpu_token_ids = chunk_list(unique_token_ids[num_gpu_blocks * block_size:],
                               block_size)

    assert allocator.get_num_free_blocks(Device.CPU) == num_cpu_blocks
    assert allocator.get_num_free_blocks(Device.GPU) == num_gpu_blocks

    cpu_blocks = [
        allocator.allocate_immutable_block(prev_block=None,
                                           token_ids=token_ids,
                                           device=Device.CPU)
        for token_ids in cpu_token_ids
    ]
    assert allocator.get_num_free_blocks(Device.CPU) == 0
    assert allocator.get_num_free_blocks(Device.GPU) == num_gpu_blocks

    gpu_blocks = [
        allocator.allocate_immutable_block(prev_block=None,
                                           token_ids=token_ids,
                                           device=Device.GPU)
        for token_ids in gpu_token_ids
    ]
    assert allocator.get_num_free_blocks(Device.CPU) == 0
    assert allocator.get_num_free_blocks(Device.GPU) == 0

    _ = [allocator.free(block) for block in cpu_blocks]
    assert allocator.get_num_free_blocks(Device.CPU) == num_cpu_blocks
    assert allocator.get_num_free_blocks(Device.GPU) == 0

    _ = [allocator.free(block) for block in gpu_blocks]
    assert allocator.get_num_free_blocks(Device.CPU) == num_cpu_blocks
    assert allocator.get_num_free_blocks(Device.GPU) == num_gpu_blocks
