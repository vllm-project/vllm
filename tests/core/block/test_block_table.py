# SPDX-License-Identifier: Apache-2.0

from typing import List

import pytest

from vllm.core.block.block_table import BlockTable
from vllm.core.block.cpu_gpu_block_allocator import CpuGpuBlockAllocator
from vllm.utils import Device, cdiv, chunk_list


@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("sequence_len", [1, 16, 129])
def test_allocate_naive(block_size: int, sequence_len: int):
    """Test the allocation of blocks using the naive allocator.

    This test creates a CpuGpuBlockAllocator with the specified block size and
    number of blocks. It then allocates multiple BlockTables with varying
    sequence lengths and verifies that the number of free blocks decreases as
    expected after each allocation.
    """
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

    block_tables: List[BlockTable] = []
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
    """Test the allocation of blocks using the prefix caching allocator.

    This test creates a CpuGpuBlockAllocator with the specified block size and
    number of blocks, using the prefix caching allocator. It then allocates
    multiple BlockTables with varying sequence lengths and verifies that the
    number of free blocks decreases as expected after each allocation.

    The test expects all sequences to share allocations, except for their last
    block, which may be mutable. It calculates the expected number of immutable
    and mutable blocks per allocation based on the sequence length and block
    size.
    """
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

    block_tables: List[BlockTable] = []
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
    """Test the allocation and freeing of blocks using different allocators and
    devices.

    This test creates a CpuGpuBlockAllocator with the specified block size,
    number of blocks, allocator type, and device. It then allocates a BlockTable
    multiple times with the same sequence and verifies that the number of free
    blocks remains consistent after each allocation and freeing.
    """
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
    """Test the allocation behavior when appending token IDs to a BlockTable.

    This test creates a CpuGpuBlockAllocator with the specified block size,
    number of blocks, and allocator type. It then allocates a BlockTable with an
    initial sequence and appends additional token IDs to it. The test verifies
    that the number of allocated blocks before and after appending matches the
    expected values.
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
    """Test the allocation behavior when ensuring a certain number of empty
    slots in a BlockTable.

    This test creates a CpuGpuBlockAllocator with the specified block size,
    number of blocks, and allocator type. It then allocates a BlockTable with an
    initial sequence and ensures a certain number of empty slots. The test
    verifies that the number of allocated blocks before and after ensuring empty
    slots matches the expected values. It also checks that filling up the empty
    slots does not consume additional blocks.
    """
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

    appended_so_far: List[int] = []
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


@pytest.mark.parametrize("block_size", [8])
@pytest.mark.parametrize("sequence_len", [1, 16, 129])
@pytest.mark.parametrize("append_len", [1, 16, 129])
@pytest.mark.parametrize("appender", ["forked", "original"])
@pytest.mark.parametrize("allocator_type", ["naive", "prefix_caching"])
def test_cow(block_size: int, sequence_len: int, append_len: int,
             allocator_type: str, appender: str):
    """Fork a sequence; append to the forked sequence; verify there's a CoW.
    """
    num_gpu_blocks = 1024

    allocator = CpuGpuBlockAllocator.create(
        allocator_type=allocator_type,
        num_gpu_blocks=num_gpu_blocks,
        num_cpu_blocks=0,
        block_size=block_size,
    )

    token_ids = list(range(sequence_len))
    token_ids_to_append = list(range(append_len))

    original_block_table = BlockTable(
        block_size=block_size,
        block_allocator=allocator,
    )

    num_expected_non_cow_blocks = cdiv(sequence_len, block_size)
    num_expected_cow_blocks = cdiv(sequence_len + append_len,
                                   block_size) - (sequence_len // block_size)

    original_block_table.allocate(token_ids=token_ids, device=Device.GPU)
    original_block_ids = original_block_table.physical_block_ids[:]

    print("original_block_ids = {}".format(original_block_ids))
    forked_block_table = original_block_table.fork()

    # Expect no additional allocation (copy on _write_).
    assert allocator.get_num_free_blocks(
        Device.GPU) == (num_gpu_blocks - num_expected_non_cow_blocks)

    if appender == "forked":
        appender_block_table = forked_block_table
        static_block_table = original_block_table
    elif appender == "original":
        appender_block_table = original_block_table
        static_block_table = forked_block_table
    else:
        raise ValueError(f"unknown test config {appender=}")

    # Write tokens.
    appender_block_table.append_token_ids(token_ids_to_append)

    # Expect the non-appending block table to have no change.
    assert static_block_table.physical_block_ids == original_block_ids
    assert appender_block_table.physical_block_ids != original_block_ids

    # Expect the blocks changed during append to have a CoW.
    assert allocator.get_num_free_blocks(
        Device.GPU) == num_gpu_blocks - (num_expected_non_cow_blocks +
                                         num_expected_cow_blocks)

    cows = allocator.clear_copy_on_writes()
    if sequence_len % block_size > 0:
        # If the last block in the sequence is not full, then when appending we
        # expect a CoW.
        assert cows

        cow_block_id = sequence_len // block_size
        expected_src = static_block_table.physical_block_ids[cow_block_id]
        expected_dst = appender_block_table.physical_block_ids[cow_block_id]

        assert (expected_src, expected_dst) in cows
    else:
        # Otherwise, there should be no copy-on-write.
        assert not cows

    static_block_table.free()
    appender_block_table.free()

    # After free, expect all blocks to be freed.
    assert allocator.get_num_free_blocks(Device.GPU) == num_gpu_blocks


@pytest.mark.parametrize("block_size", [8])
@pytest.mark.parametrize("sequence_len", [1, 16, 129])
@pytest.mark.parametrize("append_len", [1, 16, 129])
@pytest.mark.parametrize("lookahead_slots", [1, 16, 129])
@pytest.mark.parametrize("appender", ["forked", "original"])
@pytest.mark.parametrize("allocator_type", ["naive", "prefix_caching"])
def test_cow_lookahead_simple(block_size: int, sequence_len: int,
                              append_len: int, lookahead_slots: int,
                              allocator_type: str, appender: str):
    """Similar to test_cow, except with lookahead allocation. The assertions are
    less rigorous due to the complexity of the property under test.
    """
    num_gpu_blocks = 1024

    allocator = CpuGpuBlockAllocator.create(
        allocator_type=allocator_type,
        num_gpu_blocks=num_gpu_blocks,
        num_cpu_blocks=0,
        block_size=block_size,
    )

    token_ids = list(range(sequence_len))
    token_ids_to_append = list(range(append_len))

    original_block_table = BlockTable(
        block_size=block_size,
        block_allocator=allocator,
    )

    original_block_table.allocate(token_ids=token_ids, device=Device.GPU)

    # Allocate lookahead slots.
    original_block_table.ensure_num_empty_slots(lookahead_slots)
    original_block_ids = original_block_table.physical_block_ids[:]

    forked_block_table = original_block_table.fork()

    if appender == "forked":
        appender_block_table = forked_block_table
        static_block_table = original_block_table
    elif appender == "original":
        appender_block_table = original_block_table
        static_block_table = forked_block_table
    else:
        raise ValueError(f"unknown test config {appender=}")

    # Write tokens.
    appender_block_table.append_token_ids(token_ids_to_append)

    # Expect the non-appending block table to have no change.
    assert static_block_table.physical_block_ids == original_block_ids
    assert appender_block_table.physical_block_ids != original_block_ids

    cows = allocator.clear_copy_on_writes()

    # Always expect copy-on-write
    assert cows

    if sequence_len % block_size > 0:
        # If the last block in the sequence is not full, then when appending we
        # expect a CoW.
        assert cows

        cow_block_id = sequence_len // block_size
        expected_src = static_block_table.physical_block_ids[cow_block_id]
        expected_dst = appender_block_table.physical_block_ids[cow_block_id]

        assert (expected_src, expected_dst) in cows

    static_block_table.free()
    appender_block_table.free()

    # After free, expect all blocks to be freed.
    assert allocator.get_num_free_blocks(Device.GPU) == num_gpu_blocks


@pytest.mark.parametrize("block_size", [1, 8])
@pytest.mark.parametrize("sequence_len", [1, 16, 129])
@pytest.mark.parametrize("num_new_tokens", [1, 16, 129])
@pytest.mark.parametrize("num_lookahead_slots", [1, 7, 8])
@pytest.mark.parametrize("allocator_type", ["naive", "prefix_caching"])
def test_num_blocks_touched_by_append_slots(block_size: int, sequence_len: int,
                                            num_new_tokens: int,
                                            num_lookahead_slots: int,
                                            allocator_type: str):
    """Verify correct calculation of get_num_blocks_touched_by_append_slots.

    This is done by using copy-on-write, which requires any modified block to
    be copied before write if the refcount > 1. We set the refcount>1 by forking
    a sequence, then measure the free blocks before and after an append. If the
    number of consumed blocks equals what `get_num_blocks_touched_by_append_
    slots` returns, then the calculation is correct.
    """

    num_gpu_blocks = 1024

    allocator = CpuGpuBlockAllocator.create(
        allocator_type=allocator_type,
        num_gpu_blocks=num_gpu_blocks,
        num_cpu_blocks=0,
        block_size=block_size,
    )

    token_ids = list(range(sequence_len))
    token_ids_to_append = list(range(num_new_tokens))

    block_table = BlockTable(
        block_size=block_size,
        block_allocator=allocator,
    )

    block_table.allocate(token_ids=token_ids, device=Device.GPU)

    # Add lookahead before fork so both sequences have the same lookahead
    # blocks.
    block_table.ensure_num_empty_slots(num_empty_slots=num_lookahead_slots)

    # Fork sequence so that every block has refcount > 1.
    _ = block_table.fork()

    # Determine how many blocks should be touched.
    expected_num_touched_blocks = (
        block_table.get_num_blocks_touched_by_append_slots(
            token_ids=token_ids_to_append,
            num_lookahead_slots=num_lookahead_slots))

    # Measure how many blocks are touched by measuring num_free_blocks before
    # and after the append.
    #
    # We expect append_token_ids to CoW all mutated blocks that have refcount>1.
    num_free_blocks_before_append = allocator.get_num_free_blocks(Device.GPU)
    block_table.append_token_ids(token_ids_to_append, num_lookahead_slots)
    num_consumed_blocks = (num_free_blocks_before_append -
                           allocator.get_num_free_blocks(Device.GPU))

    # TODO(cade) ensure equality when num_lookahead_slots > 0.
    # The reason we have < is because lookahead blocks are not copied eagerly;
    # they are copied on first write. This will cause issues for beam search +
    # speculative decoding. This is acceptable for now as it is a large effort
    # to combine the two. To fix this, we can ensure single sequence ownership
    # of lookahead blocks by appending empty slots to each block, which will
    # trigger the CoW.
    #
    # Until then, we can accept that the consumed tokens are <= the expected
    # tokens when appending with lookahead.
    if num_lookahead_slots > 0:
        assert num_consumed_blocks <= expected_num_touched_blocks
    else:
        assert num_consumed_blocks == expected_num_touched_blocks
