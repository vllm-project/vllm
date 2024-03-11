import pytest
import time
from typing import List

from vllm import SamplingParams
from vllm.block import PhysicalTokenBlock
from vllm.core.block_manager import (BlockAllocator, BlockSpaceManager,
                                     AllocStatus)
from vllm.utils import Device
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus, Logprob

from .utils import create_dummy_prompt


def test_block_allocator_allocate():
    block_size = 4
    num_cpu_blocks = 4
    cpu_allocator = BlockAllocator(Device.CPU, block_size, num_cpu_blocks)

    # Allocate all available cpu blocks.
    num_free = num_cpu_blocks
    assert cpu_allocator.get_num_free_blocks() == num_free
    for _ in range(num_cpu_blocks):
        block = cpu_allocator.allocate()
        num_free -= 1

        assert block.block_hash not in cpu_allocator.evictor
        assert cpu_allocator.get_num_free_blocks() == num_free

    with pytest.raises(ValueError):
        cpu_allocator.allocate()


def test_block_allocator_free():
    block_size = 4
    num_cpu_blocks = 4
    cpu_allocator = BlockAllocator(Device.CPU, block_size, num_cpu_blocks)

    # Allocate all available cpu blocks.
    blocks: List[PhysicalTokenBlock] = []
    for _ in range(num_cpu_blocks):
        block = cpu_allocator.allocate()
        blocks.append(block)
        assert block.block_hash not in cpu_allocator.evictor

    # Free all allocated cpu blocks.
    num_free = 0
    assert cpu_allocator.get_num_free_blocks() == num_free
    for block in blocks:
        cpu_allocator.free(block)
        num_free += 1
        assert block.block_hash in cpu_allocator.evictor
        assert cpu_allocator.get_num_free_blocks() == num_free

        with pytest.raises(ValueError):
            cpu_allocator.free(block)


def test_allocate():
    block_size = 4
    num_cpu_blocks = 4
    num_gpu_blocks = 4
    block_manager = BlockSpaceManager(block_size,
                                      num_cpu_blocks,
                                      num_gpu_blocks,
                                      watermark=0)

    # Allocate same sequence group to all available gpu blocks.
    for i in range(num_gpu_blocks):
        _, seq_group = create_dummy_prompt(str(i), block_size)
        assert block_manager.can_allocate(seq_group)
        block_manager.allocate(seq_group)
    assert block_manager.can_allocate(seq_group) != AllocStatus.OK

    # Allocate same sequence group to all available gpu blocks.
    # Use watermark to reserve one gpu block.
    block_manager = BlockSpaceManager(block_size,
                                      num_cpu_blocks,
                                      num_gpu_blocks,
                                      watermark=1 / num_gpu_blocks)
    for i in range(num_gpu_blocks - 1):
        _, seq_group = create_dummy_prompt(str(i), block_size)
        assert block_manager.can_allocate(seq_group)
        block_manager.allocate(seq_group)
    assert block_manager.can_allocate(seq_group) != AllocStatus.OK


def test_append_slot_single_seq():
    block_size = 4
    num_cpu_blocks = 4
    num_gpu_blocks = 4
    block_manager = BlockSpaceManager(block_size,
                                      num_cpu_blocks,
                                      num_gpu_blocks,
                                      watermark=0)

    # Allocate single seq to gpu block.
    prompt, seq_group = create_dummy_prompt("1", block_size)
    block_manager.allocate(seq_group)

    # Nothing to append. Sequence has no new logical blocks.
    assert block_manager.can_append_slot(seq_group)
    before_blocks = block_manager.get_num_free_gpu_blocks()
    assert not block_manager.append_slot(prompt)
    after_blocks = block_manager.get_num_free_gpu_blocks()
    assert before_blocks == after_blocks

    # Add block_size number of new tokens and append slot.
    for i in range(block_size):
        token_id = i + 5
        prompt.append_token_id(token_id, {token_id: Logprob(0.0)})

    assert block_manager.can_append_slot(seq_group)
    before_blocks = block_manager.get_num_free_gpu_blocks()
    assert not block_manager.append_slot(prompt)
    after_blocks = block_manager.get_num_free_gpu_blocks()
    assert before_blocks - after_blocks == 1


def test_append_slot_cow():
    block_size = 4
    num_cpu_blocks = 4
    num_gpu_blocks = 4
    block_manager = BlockSpaceManager(block_size=block_size,
                                      num_cpu_blocks=num_cpu_blocks,
                                      num_gpu_blocks=num_gpu_blocks,
                                      watermark=0)

    # Allocate prompt to gpu block. There is one slot left in the block.
    prompt = Sequence(seq_id=1,
                      prompt="one two three",
                      prompt_token_ids=[1, 2, 3],
                      block_size=block_size)

    # Fork the sequence, such that a COW will be required when we append a new
    # token id.
    child = prompt.fork(new_seq_id=2)

    # Allocate space for the sequence group.
    seq_group = SequenceGroup("1", [prompt, child], SamplingParams(),
                              time.time(), time.perf_counter)
    block_manager.allocate(seq_group)

    # Fork and append a new token id. We expect a COW to be scheduled.
    token_id = 4
    child.append_token_id(token_id, {token_id: Logprob(0.0)})
    block_manager.fork(prompt, child)

    assert block_manager.can_append_slot(seq_group)
    before_blocks = block_manager.get_num_free_gpu_blocks()

    maybe_src_dst_block = block_manager.append_slot(child)
    assert maybe_src_dst_block is not None
    src_block, dst_block = maybe_src_dst_block
    assert src_block != dst_block

    after_blocks = block_manager.get_num_free_gpu_blocks()
    assert before_blocks - after_blocks == 1


def test_fork():
    block_size = 4
    num_cpu_blocks = 4
    num_gpu_blocks = 4
    block_manager = BlockSpaceManager(block_size,
                                      num_cpu_blocks,
                                      num_gpu_blocks,
                                      watermark=0)

    prompt, seq_group = create_dummy_prompt("1",
                                            block_size - 1,
                                            block_size=block_size)
    block_manager.allocate(seq_group)

    # Fork prompt and copy block tables.
    child = prompt.fork(2)
    block_manager.fork(prompt, child)
    assert block_manager.get_block_table(
        prompt) == block_manager.get_block_table(child)
    token_id = 4
    # Append token to child. Block is shared so copy on write occurs.
    child.append_token_id(token_id, {token_id: Logprob(0.0)})
    block_manager.append_slot(child)
    assert block_manager.get_block_table(
        prompt) != block_manager.get_block_table(child)


def test_swap():
    block_size = 4
    num_cpu_blocks = 4
    num_gpu_blocks = 4
    block_manager = BlockSpaceManager(block_size,
                                      num_cpu_blocks,
                                      num_gpu_blocks,
                                      watermark=0)

    prompt, seq_group = create_dummy_prompt("1", prompt_length=block_size - 1)
    prompt.status = SequenceStatus.WAITING
    block_manager.allocate(seq_group)

    # Emulate a forward pass by appending a single token.
    # The block manager then knows how many unprocessed
    # tokens will be written in the next forward pass.
    token_id = 0
    prompt.status = SequenceStatus.RUNNING
    prompt.append_token_id(token_id, {token_id: Logprob(0.0)})

    # Swap seq group from GPU -> CPU.
    gpu_blocks = block_manager.get_block_table(prompt)
    assert block_manager.can_swap_out(seq_group)
    before_cpu_blocks = block_manager.get_num_free_cpu_blocks()
    before_gpu_blocks = block_manager.get_num_free_gpu_blocks()
    mapping = block_manager.swap_out(seq_group)
    assert list(mapping.keys()) == gpu_blocks
    after_cpu_blocks = block_manager.get_num_free_cpu_blocks()
    after_gpu_blocks = block_manager.get_num_free_gpu_blocks()
    assert before_cpu_blocks == after_cpu_blocks + len(gpu_blocks)
    assert before_gpu_blocks + len(gpu_blocks) == after_gpu_blocks
    prompt.status = SequenceStatus.SWAPPED

    # Swap seq group from CPU -> GPU.
    cpu_blocks = block_manager.get_block_table(prompt)
    assert block_manager.can_swap_in(seq_group)
    before_cpu_blocks = block_manager.get_num_free_cpu_blocks()
    before_gpu_blocks = block_manager.get_num_free_gpu_blocks()
    mapping = block_manager.swap_in(seq_group)
    assert list(mapping.keys()) == cpu_blocks
    after_cpu_blocks = block_manager.get_num_free_cpu_blocks()
    after_gpu_blocks = block_manager.get_num_free_gpu_blocks()
    assert before_cpu_blocks + len(cpu_blocks) == after_cpu_blocks
    assert before_gpu_blocks == after_gpu_blocks + len(cpu_blocks)


def test_free():
    block_size = 4
    num_cpu_blocks = 4
    num_gpu_blocks = 4
    block_manager = BlockSpaceManager(block_size,
                                      num_cpu_blocks,
                                      num_gpu_blocks,
                                      watermark=0)

    prompt, seq_group = create_dummy_prompt("1", block_size)
    block_manager.allocate(seq_group)

    # Free allocated seq.
    prompt_blocks = len(block_manager.get_block_table(prompt))
    before_blocks = block_manager.get_num_free_gpu_blocks()
    block_manager.free(prompt)
    after_blocks = block_manager.get_num_free_gpu_blocks()
    assert after_blocks == before_blocks + prompt_blocks

    # Block table for freed seq is deleted.
    with pytest.raises(KeyError):
        block_manager.get_block_table(prompt)


def test_reset():
    block_size = 4
    num_cpu_blocks = 4
    num_gpu_blocks = 4
    block_manager = BlockSpaceManager(block_size,
                                      num_cpu_blocks,
                                      num_gpu_blocks,
                                      watermark=0)

    # Allocate same seq group on all available gpu blocks.
    original_blocks = block_manager.get_num_free_gpu_blocks()
    for i in range(num_gpu_blocks):
        _, seq_group = create_dummy_prompt(str(i), block_size)
        block_manager.allocate(seq_group)
    assert block_manager.get_num_free_gpu_blocks() == 0

    # Resetting block manager frees all allocated blocks.
    block_manager.reset()
    assert block_manager.get_num_free_gpu_blocks() == original_blocks
