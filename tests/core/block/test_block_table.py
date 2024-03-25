import pytest

from vllm.core.block.block_table import BlockTable
from vllm.core.block.cpu_gpu_block_allocator import CpuGpuBlockAllocator
from vllm.utils import Device, chunk_list


@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("sequence_len", [1, 16, 129])
def test_allocate_naive(block_size: int, sequence_len: int):
    assert block_size > 1
    num_gpu_blocks = 1024

    allocator = CpuGpuBlockAllocator.create(
        allocator_type="naive",
        num_gpu_blocks=num_gpu_blocks,
        num_cpu_blocks=1024,
        block_size=block_size,
    )

    token_ids = list(range(sequence_len))
    num_blocks_per_alloc = len(list(chunk_list(token_ids, block_size)))

    block_tables = []
    for i in range(5):
        assert allocator.get_num_free_blocks(
            device=Device.GPU) == num_gpu_blocks - i * num_blocks_per_alloc

        block_tables.append(
            BlockTable(
                block_size=block_size,
                block_allocator=allocator,
            ))
        block_tables[-1].allocate(token_ids=token_ids, device=Device.GPU)


@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("sequence_len", [1, 16, 129])
def test_allocate_prefix_caching(block_size: int, sequence_len: int):
    assert block_size > 1
    num_gpu_blocks = 1024

    allocator = CpuGpuBlockAllocator.create(
        allocator_type="prefix_caching",
        num_gpu_blocks=num_gpu_blocks,
        num_cpu_blocks=1024,
        block_size=block_size,
    )

    token_ids = list(range(sequence_len))
    chunked_tokens = list(chunk_list(token_ids, block_size))
    num_mutable_blocks_per_alloc = 0 if len(
        chunked_tokens[-1]) == block_size else 1
    num_immutable_blocks_per_alloc = len(
        chunked_tokens) - num_mutable_blocks_per_alloc

    block_tables = []
    for alloc_i in range(1, 6):

        block_tables.append(
            BlockTable(
                block_size=block_size,
                block_allocator=allocator,
            ))
        block_tables[-1].allocate(token_ids=token_ids, device=Device.GPU)

        # Expect all sequences to share allocations, except for their last block
        # (which may be mutable).
        assert allocator.get_num_free_blocks(
            device=Device.GPU) == num_gpu_blocks - (
                num_immutable_blocks_per_alloc + num_mutable_blocks_per_alloc *
                (alloc_i))


@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("sequence_len", [1, 16, 129])
@pytest.mark.parametrize("allocator_type", ["naive", "prefix_caching"])
@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_allocate_free(block_size: int, sequence_len: int, allocator_type: str,
                       device: str):
    device = Device[device.upper()]

    num_device_blocks = 1024
    allocator = CpuGpuBlockAllocator.create(
        allocator_type=allocator_type,
        num_gpu_blocks=num_device_blocks,
        num_cpu_blocks=num_device_blocks,
        block_size=block_size,
    )

    token_ids = list(range(sequence_len))
    num_blocks_per_alloc = len(list(chunk_list(token_ids, block_size)))

    block_table = BlockTable(
        block_size=block_size,
        block_allocator=allocator,
    )

    for i in range(5):
        block_table.allocate(token_ids=token_ids, device=device)
        assert allocator.get_num_free_blocks(
            device) == num_device_blocks - num_blocks_per_alloc
        assert all(block_id is not None
                   for block_id in block_table.physical_block_ids)

        block_table.free()
        assert allocator.get_num_free_blocks(device) == num_device_blocks


@pytest.mark.parametrize("block_size", [1, 8])
@pytest.mark.parametrize("sequence_len", [1, 16, 129])
@pytest.mark.parametrize("append_len", [1, 16, 129])
@pytest.mark.parametrize("allocator_type", ["naive", "prefix_caching"])
def test_append_token_ids_allocation(block_size: int, sequence_len: int,
                                     append_len: int, allocator_type: str):
    num_gpu_blocks = 1024

    allocator = CpuGpuBlockAllocator.create(
        allocator_type=allocator_type,
        num_gpu_blocks=num_gpu_blocks,
        num_cpu_blocks=1024,
        block_size=block_size,
    )

    token_ids = list(range(sequence_len))
    token_ids_to_append = list(range(append_len))

    block_table = BlockTable(
        block_size=block_size,
        block_allocator=allocator,
    )

    num_expected_blocks_before_append = len(
        list(chunk_list(token_ids, block_size)))
    num_expected_appended_blocks = len(
        list(chunk_list(token_ids + token_ids_to_append,
                        block_size))) - num_expected_blocks_before_append

    block_table.allocate(token_ids=token_ids, device=Device.GPU)

    assert len(
        block_table.physical_block_ids) == num_expected_blocks_before_append
    block_table.append_token_ids(token_ids_to_append)
    assert len(
        block_table.physical_block_ids
    ) == num_expected_blocks_before_append + num_expected_appended_blocks


@pytest.mark.parametrize("block_size", [1, 8])
@pytest.mark.parametrize("sequence_len", [1, 16, 129])
@pytest.mark.parametrize("num_empty_slots", [1, 16, 129])
@pytest.mark.parametrize("allocator_type", ["naive", "prefix_caching"])
def test_ensure_num_empty_slots_allocation(block_size: int, sequence_len: int,
                                           num_empty_slots: int,
                                           allocator_type: str):
    num_gpu_blocks = 1024

    allocator = CpuGpuBlockAllocator.create(
        allocator_type=allocator_type,
        num_gpu_blocks=num_gpu_blocks,
        num_cpu_blocks=1024,
        block_size=block_size,
    )

    token_ids = list(range(sequence_len))

    block_table = BlockTable(
        block_size=block_size,
        block_allocator=allocator,
    )

    num_expected_blocks_before_append = len(
        list(chunk_list(token_ids, block_size)))
    num_expected_appended_blocks = len(
        list(chunk_list(token_ids + [-1] * num_empty_slots,
                        block_size))) - num_expected_blocks_before_append

    block_table.allocate(token_ids=token_ids, device=Device.GPU)

    # Assert that the empty slots consume the expected number of additional
    # blocks.
    assert len(
        block_table.physical_block_ids) == num_expected_blocks_before_append
    block_table.ensure_num_empty_slots(num_empty_slots)
    assert len(
        block_table.physical_block_ids
    ) == num_expected_blocks_before_append + num_expected_appended_blocks

    # Now, ensure no additional blocks consumed as we fill up the empty slots.
    num_free_blocks = allocator.get_num_free_blocks(device=Device.GPU)
    block_table.append_token_ids(token_ids=list(range(num_empty_slots)))
    assert num_free_blocks == allocator.get_num_free_blocks(device=Device.GPU)


@pytest.mark.parametrize("block_size", [1, 8])
@pytest.mark.parametrize("sequence_len", [1, 9])
@pytest.mark.parametrize("append_len", [1, 16, 129])
@pytest.mark.parametrize("append_size", [1, 4, 129])
@pytest.mark.parametrize("allocator_type", ["naive", "prefix_caching"])
def test_append_token_ids_correct_content(block_size: int, sequence_len: int,
                                          append_len: int, allocator_type: str,
                                          append_size: int):
    """Verify token ids are correctly appended. Appends various amounts of
    token ids in various append sizes, and verifies the final sequence is
    correct.
    """
    num_gpu_blocks = 1024

    allocator = CpuGpuBlockAllocator.create(
        allocator_type=allocator_type,
        num_gpu_blocks=num_gpu_blocks,
        num_cpu_blocks=1024,
        block_size=block_size,
    )

    token_ids = list(range(sequence_len))
    token_ids_to_append = list(range(append_len))

    block_table = BlockTable(
        block_size=block_size,
        block_allocator=allocator,
    )
    block_table.allocate(token_ids=token_ids, device=Device.GPU)

    appended_so_far = []
    for append in chunk_list(token_ids_to_append, append_size):
        block_table.append_token_ids(append)
        appended_so_far.extend(append)

        assert block_table._get_all_token_ids() == token_ids + appended_so_far

    assert block_table._get_all_token_ids() == token_ids + token_ids_to_append


@pytest.mark.parametrize("seq_len", [1, 9, 129])
@pytest.mark.parametrize("block_size", [1, 8])
@pytest.mark.parametrize("allocator_type", ["naive", "prefix_caching"])
def test_fork(seq_len: int, block_size: int, allocator_type: str):
    """Create a sequence using the specified allocator.
        1. Assert that after forking the sequence, the free block count is the
            same.
        2. Assert that the forked sequence has the same physical mappings.
        3. Then free the original sequence; verify that the free block count is
            the same.
        4. Finally, free the forked sequence and verify that the free block
            count drops to zero.
    """
    num_gpu_blocks = 1024

    allocator = CpuGpuBlockAllocator.create(
        allocator_type=allocator_type,
        num_gpu_blocks=num_gpu_blocks,
        num_cpu_blocks=0,
        block_size=block_size,
    )

    token_ids = list(range(seq_len))

    block_table = BlockTable(
        block_size=block_size,
        block_allocator=allocator,
    )

    block_table.allocate(token_ids)

    num_free_blocks_before_fork = allocator.get_num_free_blocks(
        device=Device.GPU)

    forked_block_table = block_table.fork()

    # Expect physical_block_ids and token_ids to match.
    assert (block_table.physical_block_ids ==
            forked_block_table.physical_block_ids)
    assert block_table._get_all_token_ids(
    ) == forked_block_table._get_all_token_ids()

    # Do not expect any additional allocations.
    assert allocator.get_num_free_blocks(
        device=Device.GPU) == num_free_blocks_before_fork

    # Free the original blocks. Assert num free blocks does not change, since
    # refcount is nonzero.
    block_table.free()
    assert allocator.get_num_free_blocks(
        device=Device.GPU) == num_free_blocks_before_fork

    # Expect the forked block table to be unaffected by the free.
    assert all(block_id is not None
               for block_id in forked_block_table.physical_block_ids)

    # Free the forked blocks. Assert num free blocks does change, since
    # refcount is now zero.
    forked_block_table.free()
    assert allocator.get_num_free_blocks(device=Device.GPU) == num_gpu_blocks
