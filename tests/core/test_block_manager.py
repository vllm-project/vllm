
import pytest
import time
from typing import List

from vllm import SamplingParams
from vllm.block import PhysicalTokenBlock
from vllm.core.block_manager import BlockAllocator, BlockSpaceManager, AllocStatus
from vllm.utils import Device
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from tests.utils import round_up_to_next_block

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
        assert block not in cpu_allocator.free_blocks
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
        assert block not in cpu_allocator.free_blocks

    # Free all allocated cpu blocks.
    num_free = 0
    assert cpu_allocator.get_num_free_blocks() == num_free
    for block in blocks:
        cpu_allocator.free(block)
        num_free += 1
        assert block in cpu_allocator.free_blocks
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
    assert block_manager.can_append_slots(seq_group)
    before_blocks = block_manager.get_num_free_gpu_blocks()
    assert not block_manager.append_slots(prompt)
    after_blocks = block_manager.get_num_free_gpu_blocks()
    assert before_blocks == after_blocks

    # Add block_size number of new tokens and append slot.
    for i in range(block_size):
        prompt.append_token_id(i + 5, {i + 5: 0})

    assert block_manager.can_append_slots(seq_group)
    before_blocks = block_manager.get_num_free_gpu_blocks()
    assert not block_manager.append_slots(prompt)
    after_blocks = block_manager.get_num_free_gpu_blocks()
    assert before_blocks - after_blocks == 1


@pytest.mark.parametrize("prompt_len", [1, 10, 100])
@pytest.mark.parametrize("num_unprocessed_tokens", [1, 10, 100])
@pytest.mark.parametrize("block_size", [1, 8, 16, 32])
def test_append_multiple_slot_single_seq(prompt_len: int,
                                         num_unprocessed_tokens: int,
                                         block_size: int):
    """Verify correct allocation when multiple tokens need to be processed.
    """
    num_cpu_blocks = 0
    num_gpu_blocks = 8192 // block_size
    block_manager = BlockSpaceManager(block_size=block_size,
                                      num_gpu_blocks=num_gpu_blocks,
                                      num_cpu_blocks=num_cpu_blocks,
                                      watermark=0)

    # Allocate single seq to gpu block.
    prompt, seq_group = create_dummy_prompt(request_id="1",
                                            prompt_length=prompt_len,
                                            block_size=block_size)
    block_manager.allocate(seq_group)

    # Nothing to append. Sequence has no new logical blocks.
    assert block_manager.can_append_slots(seq_group)
    before_blocks = block_manager.get_num_free_gpu_blocks()
    assert not block_manager.append_slots(prompt)
    after_blocks = block_manager.get_num_free_gpu_blocks()
    assert before_blocks == after_blocks

    # Append new tokens, expect correct number of new blocks
    new_token_ids = list(range(num_unprocessed_tokens))
    prompt.append_token_ids(new_token_ids, [{
        token_id: 0
    } for token_id in new_token_ids])

    old_seq_len_in_blocks = round_up_to_next_block(prompt_len, block_size)
    new_seq_len_in_blocks = round_up_to_next_block(
        prompt_len + num_unprocessed_tokens, block_size)
    num_expected_new_blocks = new_seq_len_in_blocks - old_seq_len_in_blocks

    assert block_manager.can_append_slots(seq_group)
    before_blocks = block_manager.get_num_free_gpu_blocks()
    assert not block_manager.append_slots(prompt)
    after_blocks = block_manager.get_num_free_gpu_blocks()
    assert before_blocks - after_blocks == num_expected_new_blocks


def test_append_slot_cow():
    block_size = 4
    num_cpu_blocks = 4
    num_gpu_blocks = 4
    block_manager = BlockSpaceManager(block_size,
                                      num_cpu_blocks,
                                      num_gpu_blocks,
                                      watermark=0)

    # Allocate prompt to gpu block.
    prompt = Sequence(1, "one two three", [1, 2, 3], block_size)
    child = prompt.fork(2)
    child.append_token_id(4, {4: 0})
    seq_group = SequenceGroup("1", [prompt, child], SamplingParams(),
                              time.time(), time.perf_counter)
    block_manager.allocate(seq_group)

    # Append slot for child token.
    # Last block being modified is shared. Copy on write occurs.
    assert block_manager.can_append_slots(seq_group)
    before_blocks = block_manager.get_num_free_gpu_blocks()
    cow_src_dst = block_manager.append_slots(child)

    assert len(cow_src_dst) > 0
    for src_block, dst_block in cow_src_dst.items():
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

    # Append token to child. Block is shared so copy on write occurs.
    child.append_token_id(4, {4: 0})
    block_manager.append_slots(child)
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
    prompt.status = SequenceStatus.RUNNING
    block_manager.allocate(seq_group)

    # Emulate a forward pass by appending a single token.
    # The block manager then knows how many unprocessed
    # tokens will be written in the next forward pass.
    prompt.append_token_id(0, {0: 0.0})

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
