import time
from collections import defaultdict
from typing import List

import pytest

from vllm import SamplingParams
from vllm.block import PhysicalTokenBlock
from vllm.core.block.utils import (STR_NOT_IMPL_ENC_DEC_PREFIX_CACHE,
                                   STR_NOT_IMPL_ENC_DEC_SWA)
from vllm.core.block_manager_v1 import (BlockSpaceManagerV1,
                                        UncachedBlockAllocator)
from vllm.core.interfaces import AllocStatus
from vllm.sequence import Logprob, Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device

from .utils import create_dummy_prompt, create_dummy_prompt_encoder_decoder


def test_block_allocator_allocate():
    block_size = 4
    num_cpu_blocks = 4
    cpu_allocator = UncachedBlockAllocator(Device.CPU, block_size,
                                           num_cpu_blocks)

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
    cpu_allocator = UncachedBlockAllocator(Device.CPU, block_size,
                                           num_cpu_blocks)

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
    block_manager = BlockSpaceManagerV1(block_size,
                                        num_cpu_blocks,
                                        num_gpu_blocks,
                                        watermark=0)

    # Allocate same sequence group to all available gpu blocks.
    for i in range(num_gpu_blocks):
        _, seq_group = create_dummy_prompt(str(i), block_size)
        assert block_manager.can_allocate(seq_group) == AllocStatus.OK
        block_manager.allocate(seq_group)
    assert block_manager.can_allocate(seq_group) != AllocStatus.OK

    # Allocate same sequence group to all available gpu blocks.
    # Use watermark to reserve one gpu block.
    block_manager = BlockSpaceManagerV1(block_size,
                                        num_cpu_blocks,
                                        num_gpu_blocks,
                                        watermark=1 / num_gpu_blocks)
    for i in range(num_gpu_blocks - 1):
        _, seq_group = create_dummy_prompt(str(i), block_size)
        assert block_manager.can_allocate(seq_group) == AllocStatus.OK
        block_manager.allocate(seq_group)
    assert block_manager.can_allocate(seq_group) != AllocStatus.OK


def test_allocate_encoder_decoder():
    block_size = 4
    num_cpu_blocks = 4
    num_gpu_blocks = 4
    block_req_per_seq_group = 2
    block_manager = BlockSpaceManagerV1(block_size,
                                        num_cpu_blocks,
                                        num_gpu_blocks,
                                        watermark=0)

    # Allocate same sequence group to all available gpu blocks.
    for i in range(num_gpu_blocks // block_req_per_seq_group):
        _, _, seq_group = create_dummy_prompt_encoder_decoder(
            str(i),
            decoder_prompt_length=block_size,
            encoder_prompt_length=block_size)
        assert block_manager.can_allocate(seq_group) == AllocStatus.OK
        block_manager.allocate(seq_group)
    assert block_manager.can_allocate(seq_group) != AllocStatus.OK

    # Allocate same sequence group to all available gpu blocks.
    # Use watermark to reserve one gpu block.
    block_manager = BlockSpaceManagerV1(block_size,
                                        num_cpu_blocks,
                                        num_gpu_blocks,
                                        watermark=1 / num_gpu_blocks)
    for i in range((num_gpu_blocks - 1) // block_req_per_seq_group):
        _, _, seq_group = create_dummy_prompt_encoder_decoder(
            str(i),
            decoder_prompt_length=block_size,
            encoder_prompt_length=block_size)
        assert block_manager.can_allocate(seq_group) == AllocStatus.OK
        block_manager.allocate(seq_group)
    assert block_manager.can_allocate(seq_group) != AllocStatus.OK


def test_allocate_encoder_decoder_fails_with_swa():
    # SWA short for sliding window attention

    block_size = 4
    num_cpu_blocks = 4
    num_gpu_blocks = 4
    block_manager = BlockSpaceManagerV1(block_size,
                                        num_cpu_blocks,
                                        num_gpu_blocks,
                                        watermark=0,
                                        sliding_window=5)  # swa

    # Allocate same sequence group to all available gpu blocks.
    _, _, seq_group = create_dummy_prompt_encoder_decoder(
        "0",
        decoder_prompt_length=block_size,
        encoder_prompt_length=block_size)

    # Assert that can_allocate() fails due to SWA
    with pytest.raises(NotImplementedError) as exc_info:
        block_manager.can_allocate(seq_group)

    assert str(exc_info.value) == STR_NOT_IMPL_ENC_DEC_SWA

    # Assert that allocate() fails due to SWA
    with pytest.raises(NotImplementedError) as exc_info:
        block_manager.allocate(seq_group)

    assert str(exc_info.value) == STR_NOT_IMPL_ENC_DEC_SWA


def test_allocate_encoder_decoder_fails_with_prefix_caching():
    block_size = 4
    num_cpu_blocks = 4
    num_gpu_blocks = 4
    block_manager = BlockSpaceManagerV1(block_size,
                                        num_cpu_blocks,
                                        num_gpu_blocks,
                                        watermark=0,
                                        enable_caching=True)  # Prefix cache

    # Allocate same sequence group to all available gpu blocks.
    _, _, seq_group = create_dummy_prompt_encoder_decoder(
        "0",
        decoder_prompt_length=block_size,
        encoder_prompt_length=block_size)

    # Assert that can_allocate() fails due to prefix caching
    with pytest.raises(NotImplementedError) as exc_info:
        block_manager.can_allocate(seq_group)

    assert str(exc_info.value) == STR_NOT_IMPL_ENC_DEC_PREFIX_CACHE

    # Assert that allocate() fails due to prefix caching
    with pytest.raises(NotImplementedError) as exc_info:
        block_manager.allocate(seq_group)

    assert str(exc_info.value) == STR_NOT_IMPL_ENC_DEC_PREFIX_CACHE


def test_append_slot_single_seq():
    block_size = 4
    num_cpu_blocks = 4
    num_gpu_blocks = 4
    block_manager = BlockSpaceManagerV1(block_size,
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
        token_id = i + 5
        prompt.append_token_id(token_id, {token_id: Logprob(0.0)})

    assert block_manager.can_append_slots(seq_group)
    before_blocks = block_manager.get_num_free_gpu_blocks()
    assert not block_manager.append_slots(prompt)
    after_blocks = block_manager.get_num_free_gpu_blocks()
    assert before_blocks - after_blocks == 1


def test_append_slot_cow():
    block_size = 4
    num_cpu_blocks = 4
    num_gpu_blocks = 4
    block_manager = BlockSpaceManagerV1(block_size=block_size,
                                        num_cpu_blocks=num_cpu_blocks,
                                        num_gpu_blocks=num_gpu_blocks,
                                        watermark=0)

    # Allocate prompt to gpu block. There is one slot left in the block.
    prompt = Sequence(seq_id=1,
                      inputs={
                          "prompt": "one two three",
                          "prompt_token_ids": [1, 2, 3],
                      },
                      block_size=block_size)

    # Fork the sequence, such that a COW will be required when we append a new
    # token id.
    child = prompt.fork(new_seq_id=2)

    # Allocate space for the sequence group.
    seq_group = SequenceGroup(request_id="1",
                              seqs=[prompt, child],
                              arrival_time=time.time(),
                              sampling_params=SamplingParams())
    block_manager.allocate(seq_group)

    # Fork and append a new token id. We expect a COW to be scheduled.
    token_id = 4
    child.append_token_id(token_id, {token_id: Logprob(0.0)})
    block_manager.fork(prompt, child)

    assert block_manager.can_append_slots(seq_group)
    before_blocks = block_manager.get_num_free_gpu_blocks()

    cows = block_manager.append_slots(child)
    assert cows
    dict_cows = defaultdict(list)
    for src_block, dst_block in cows:
        dict_cows[src_block].append(dst_block)
    for src_block, dst_blocks in dict_cows.items():
        assert src_block not in dst_blocks

    after_blocks = block_manager.get_num_free_gpu_blocks()
    assert before_blocks - after_blocks == 1


def test_fork():
    block_size = 4
    num_cpu_blocks = 4
    num_gpu_blocks = 4
    block_manager = BlockSpaceManagerV1(block_size,
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
    block_manager.append_slots(child)
    assert block_manager.get_block_table(
        prompt) != block_manager.get_block_table(child)


def test_swap():
    block_size = 4
    num_cpu_blocks = 4
    num_gpu_blocks = 4
    block_manager = BlockSpaceManagerV1(block_size,
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
    assert [x[0] for x in mapping] == gpu_blocks
    after_cpu_blocks = block_manager.get_num_free_cpu_blocks()
    after_gpu_blocks = block_manager.get_num_free_gpu_blocks()
    assert before_cpu_blocks == after_cpu_blocks + len(gpu_blocks)
    assert before_gpu_blocks + len(gpu_blocks) == after_gpu_blocks
    prompt.status = SequenceStatus.SWAPPED

    # Swap seq group from CPU -> GPU.
    cpu_blocks = block_manager.get_block_table(prompt)
    assert block_manager.can_swap_in(seq_group) == AllocStatus.OK
    before_cpu_blocks = block_manager.get_num_free_cpu_blocks()
    before_gpu_blocks = block_manager.get_num_free_gpu_blocks()
    mapping = block_manager.swap_in(seq_group)
    assert [x[0] for x in mapping] == cpu_blocks
    after_cpu_blocks = block_manager.get_num_free_cpu_blocks()
    after_gpu_blocks = block_manager.get_num_free_gpu_blocks()
    assert before_cpu_blocks + len(cpu_blocks) == after_cpu_blocks
    assert before_gpu_blocks == after_gpu_blocks + len(cpu_blocks)


def test_swap_encoder_decoder():
    block_size = 4
    num_cpu_blocks = 4
    num_gpu_blocks = 4
    block_manager = BlockSpaceManagerV1(block_size,
                                        num_cpu_blocks,
                                        num_gpu_blocks,
                                        watermark=0)

    decoder_prompt, encoder_prompt, seq_group = \
        create_dummy_prompt_encoder_decoder(
        "1",
        decoder_prompt_length=block_size,
        encoder_prompt_length=block_size)
    decoder_prompt.status = SequenceStatus.WAITING
    encoder_prompt.status = SequenceStatus.WAITING
    block_manager.allocate(seq_group)

    # Emulate a forward pass by appending a single token.
    # The block manager then knows how many unprocessed
    # tokens will be written in the next forward pass.
    token_id = 0
    decoder_prompt.status = SequenceStatus.RUNNING
    decoder_prompt.append_token_id(token_id, {token_id: Logprob(0.0)})

    # Swap encoder/decoder seq group from GPU -> CPU.
    decoder_gpu_blocks = block_manager.get_block_table(decoder_prompt)
    cross_gpu_blocks = block_manager.get_cross_block_table(seq_group)
    gpu_blocks = decoder_gpu_blocks + cross_gpu_blocks
    assert block_manager.can_swap_out(seq_group)
    before_cpu_blocks = block_manager.get_num_free_cpu_blocks()
    before_gpu_blocks = block_manager.get_num_free_gpu_blocks()
    mapping = block_manager.swap_out(seq_group)
    assert [x[0] for x in mapping] == gpu_blocks
    #assert list(mapping.keys()) == gpu_blocks
    after_cpu_blocks = block_manager.get_num_free_cpu_blocks()
    after_gpu_blocks = block_manager.get_num_free_gpu_blocks()
    assert before_cpu_blocks == after_cpu_blocks + len(gpu_blocks)
    assert before_gpu_blocks + len(gpu_blocks) == after_gpu_blocks
    decoder_prompt.status = SequenceStatus.SWAPPED

    # Swap encoder/decoder seq group from CPU -> GPU.
    decoder_cpu_blocks = block_manager.get_block_table(decoder_prompt)
    cross_cpu_blocks = block_manager.get_cross_block_table(seq_group)
    cpu_blocks = decoder_cpu_blocks + cross_cpu_blocks
    assert block_manager.can_swap_in(seq_group) == AllocStatus.OK
    before_cpu_blocks = block_manager.get_num_free_cpu_blocks()
    before_gpu_blocks = block_manager.get_num_free_gpu_blocks()
    mapping = block_manager.swap_in(seq_group)
    assert [x[0] for x in mapping] == cpu_blocks
    after_cpu_blocks = block_manager.get_num_free_cpu_blocks()
    after_gpu_blocks = block_manager.get_num_free_gpu_blocks()
    assert before_cpu_blocks + len(cpu_blocks) == after_cpu_blocks
    assert before_gpu_blocks == after_gpu_blocks + len(cpu_blocks)


def test_free():
    block_size = 4
    num_cpu_blocks = 4
    num_gpu_blocks = 4
    block_manager = BlockSpaceManagerV1(block_size,
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


def test_free_encoder_decoder():
    block_size = 4
    num_cpu_blocks = 4
    num_gpu_blocks = 4
    block_manager = BlockSpaceManagerV1(block_size,
                                        num_cpu_blocks,
                                        num_gpu_blocks,
                                        watermark=0)

    decoder_prompt, encoder_prompt, seq_group = \
        create_dummy_prompt_encoder_decoder(
        "1",
        decoder_prompt_length=block_size,
        encoder_prompt_length=block_size)
    block_manager.allocate(seq_group)

    # Free allocated seq.
    decoder_prompt_blocks = len(block_manager.get_block_table(decoder_prompt))
    encoder_prompt_blocks = len(block_manager.get_cross_block_table(seq_group))
    prompt_blocks = decoder_prompt_blocks + encoder_prompt_blocks
    before_blocks = block_manager.get_num_free_gpu_blocks()
    block_manager.free(decoder_prompt)
    block_manager.free_cross(seq_group)
    after_blocks = block_manager.get_num_free_gpu_blocks()
    assert after_blocks == before_blocks + prompt_blocks

    # Block table for freed encoder & decoder seq's are deleted.
    with pytest.raises(KeyError):
        block_manager.get_block_table(decoder_prompt)

    # Block table for freed encoder & decoder seq's are deleted.
    with pytest.raises(KeyError):
        block_manager.get_block_table(encoder_prompt)


def test_reset():
    block_size = 4
    num_cpu_blocks = 4
    num_gpu_blocks = 4
    block_manager = BlockSpaceManagerV1(block_size,
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


def test_reset_encoder_decoder():
    block_size = 4
    num_cpu_blocks = 4
    num_gpu_blocks = 4
    block_req_per_seq_group = 2
    block_manager = BlockSpaceManagerV1(block_size,
                                        num_cpu_blocks,
                                        num_gpu_blocks,
                                        watermark=0)

    # Allocate same seq group on all available gpu blocks.
    original_blocks = block_manager.get_num_free_gpu_blocks()
    for i in range(num_gpu_blocks // block_req_per_seq_group):
        _, _, seq_group = create_dummy_prompt_encoder_decoder(
            f"{i}",
            decoder_prompt_length=block_size,
            encoder_prompt_length=block_size)
        block_manager.allocate(seq_group)
    assert block_manager.get_num_free_gpu_blocks() == 0

    # Resetting block manager frees all allocated blocks.
    block_manager.reset()
    assert block_manager.get_num_free_gpu_blocks() == original_blocks


def test_sliding_window_multi_seq():
    """
    Tests that memory allocation and deallocation is handled
    correctly with multiple sequences that exceed the sliding
    window's capacity.
    """
    block_size = 1
    num_cpu_blocks = 8
    num_gpu_blocks = 8
    sliding_window = 2
    block_manager = BlockSpaceManagerV1(block_size,
                                        num_cpu_blocks,
                                        num_gpu_blocks,
                                        sliding_window=sliding_window,
                                        watermark=0)

    assert block_manager.get_num_free_gpu_blocks() == num_gpu_blocks

    parent = Sequence(seq_id=1,
                      inputs={
                          "prompt": "one two three",
                          "prompt_token_ids": [0, 1, 2],
                      },
                      block_size=block_size)
    seq_group = SequenceGroup(request_id="1",
                              seqs=[parent],
                              arrival_time=time.time(),
                              sampling_params=SamplingParams(),
                              lora_request=None)
    block_manager.allocate(seq_group)

    # assert the number of blocks allocated is correct
    # the parent seq has len 3, but since sliding_window is 2,
    # we will use at most 2 blocks
    assert block_manager.get_num_free_gpu_blocks(
    ) == num_gpu_blocks - sliding_window

    # Fork prompt and copy block tables.
    child = parent.fork(2)
    block_manager.fork(parent, child)

    # assert the number of blocks allocated is correct
    # forking does not increase memory consumption
    assert block_manager.get_num_free_gpu_blocks(
    ) == num_gpu_blocks - sliding_window

    # assert both parent and child share all blocks
    assert block_manager.get_block_table(
        parent) == block_manager.get_block_table(child)

    token_id = 4
    # Append token to child. Block is shared so copy on write occurs.
    child.append_token_id(token_id, {token_id: Logprob(0.0)})
    block_manager.append_slots(child)

    # assert the number of blocks allocated is correct
    # we will use now one block more. Each seq will use 2 blocks,
    # but only one can be shared
    assert block_manager.get_num_free_gpu_blocks(
    ) == num_gpu_blocks - sliding_window - 1

    token_id = 5
    parent.append_token_id(token_id, {token_id: Logprob(0.0)})
    block_manager.append_slots(parent)

    # assert the number of blocks allocated is correct
    # no change, because both sequences are still just sharing one block
    assert block_manager.get_num_free_gpu_blocks(
    ) == num_gpu_blocks - sliding_window - 1

    block_table_parent = block_manager.get_block_table(parent)
    block_table_child = block_manager.get_block_table(child)

    assert block_table_parent != block_table_child

    # assert both blocks are sharing the second-last block
    assert block_table_parent[-2] == block_table_child[-2]

    # now let's clean up...
    block_manager.free(parent)

    # assert the number of blocks allocated is correct
    # We have freed one seq, reducing the ref count of two blocks by one.
    # One of the two was only used by the parent seq, so this is now free.
    # The child seq still consumes sliding_window blocks
    assert block_manager.get_num_free_gpu_blocks(
    ) == num_gpu_blocks - sliding_window

    # free all blocks
    block_manager.free(child)

    # assert all blocks are free now
    assert block_manager.get_num_free_gpu_blocks() == num_gpu_blocks


def test_mark_blocks_as_computed_with_prefix_cache_and_chunked_prefill():
    """When prefix cache and chunked prefill are enabled, the block manager
    should only mark a chunk of blocks as computed instead of all blocks.
    """

    block_size = 4
    num_cpu_blocks = 0
    num_gpu_blocks = 16
    block_manager = BlockSpaceManagerV1(block_size,
                                        num_gpu_blocks,
                                        num_cpu_blocks,
                                        watermark=0,
                                        enable_caching=True)

    # Set prompt size to have num_gpu_blocks - 1 full blocks.
    prompt_length = block_size * num_gpu_blocks - 1

    # Allocate (reserve) all blocks.
    _, seq_group = create_dummy_prompt("0",
                                       prompt_length,
                                       block_size=block_size)
    block_manager.allocate(seq_group)
    assert seq_group.seqs[0].n_blocks == num_gpu_blocks

    # 1st chunk: Compute 2 and half blocks. Should mark 2 blocks as computed.
    token_chunk_size = int(block_size * 2.5)
    block_manager.mark_blocks_as_computed(seq_group, token_chunk_size)
    computed_blocks = block_manager.get_all_computed_blocks(seq_group.seqs[0])
    assert len(computed_blocks) == 2

    # Actual computed tokens.
    seq_group.seqs[0].data.update_num_computed_tokens(token_chunk_size)

    # 2nd chunk: Complete 3rd block and additional 4 blocks.
    token_chunk_size = int(block_size * 4.5)
    block_manager.mark_blocks_as_computed(seq_group, token_chunk_size)
    computed_blocks = block_manager.get_all_computed_blocks(seq_group.seqs[0])
    assert len(computed_blocks) == 7
