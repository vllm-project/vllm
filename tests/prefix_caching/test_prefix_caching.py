"""Compare the with and without prefix caching.

Run `pytest tests/prefix_caching/test_prefix_caching.py`.
"""
import pytest

from vllm import LLM, SamplingParams
from vllm.core.block_manager import BlockAllocator
from vllm.utils import Device

prefix = (
    "You are an expert school principal, skilled in effectively managing "
    "faculty and staff. Draft 10-15 questions for a potential first grade "
    "Head Teacher for my K-12, all-girls', independent school that emphasizes "
    "community, joyful discovery, and life-long learning. The candidate is "
    "coming in for a first-round panel interview for a 8th grade Math "
    "teaching role. They have 5 years of previous teaching experience "
    "as an assistant teacher at a co-ed, public school with experience "
    "in middle school math teaching. Based on these information, fulfill "
    "the following paragraph: ")


def allocate_all_blocks(block_allocator, num_blocks):
    blocks = []
    for i in range(num_blocks):
        # use i as the block_hash
        blocks.append(block_allocator.allocate(i))
    return blocks


@pytest.mark.parametrize("model", ["facebook/opt-125m"])
@pytest.mark.parametrize("max_tokens", [16])
def test_prefix_caching(
    example_prompts,
    model: str,
    max_tokens: int,
):
    llm = LLM(model=model)
    # -1 since the last token can change when concatenating prompts.
    prefix_pos = len(llm.llm_engine.tokenizer.encode(prefix)) - 1
    prompts = [prefix + prompt for prompt in example_prompts]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    outputs_without_prefix = llm.generate(prompts, sampling_params)
    outputs_with_prefix = llm.generate(prompts,
                                       sampling_params,
                                       prefix_pos=[prefix_pos] * len(prompts))
    for output_without_prefix, output_with_prefix in zip(
            outputs_without_prefix, outputs_with_prefix):
        assert (output_without_prefix.outputs[0].token_ids ==
                output_with_prefix.outputs[0].token_ids)


@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("num_blocks", [16])
def test_block_allocator(
    block_size: int,
    num_blocks: int,
):
    block_hash = 1
    block_allocator = BlockAllocator(Device.CPU, block_size, num_blocks)

    # Allocate two PysicalTokenBlocks with the same hash and check that they are the same PhysicalTokenBlock
    first_block = block_allocator.allocate(block_hash)
    second_block = block_allocator.allocate(block_hash)
    assert (first_block == second_block)
    assert (second_block.ref_count == 2)

    # Free the first_block and confirm that the ref_count is correctly decremented on the second block
    block_allocator.free(first_block)
    assert (second_block.ref_count == 1)

    # Free the second block and confirm that the block ends up on the free list
    block_allocator.free(second_block)
    assert (len(block_allocator.evictor.free_blocks) == 1)
    free_block = block_allocator.evictor.free_blocks[block_hash]
    assert (free_block == second_block)

    # Reallocate the first block and confirm that, even after the block had its ref_count go to 0, we still get the same block back
    first_block = block_allocator.allocate(block_hash)
    assert (first_block == second_block)
    assert (first_block.block_hash == block_hash)


@pytest.mark.parametrize("num_blocks", [16])
def test_eviction(num_blocks: int, ):
    block_size = 16
    block_allocator = BlockAllocator(Device.CPU, block_size, num_blocks)
    blocks = []

    for i in range(num_blocks):
        # use i as the block_hash
        blocks.append(block_allocator.allocate(i))

    #Free all blocks
    for block in blocks:
        block_allocator.free(block)

    # Allocate a new block and confirm that it's the first block freed. I.E The Least Recently Used block
    new_block_hash = block_size
    new_block = block_allocator.allocate(new_block_hash)
    assert (new_block == blocks[0])
    assert (new_block.block_hash == new_block_hash)

    # Reallocate the second in blocks to remove it from the free list
    realloc_block_hash = 1
    realloc_block = block_allocator.allocate(realloc_block_hash)
    assert (realloc_block == blocks[realloc_block_hash])
    assert (realloc_block.block_hash == realloc_block_hash)

    # Allocate a new block and confirm that it's not the realloc_block, since the realloc_block shouldn't be in the free list
    new_block_hash = block_size + 1
    new_block = block_allocator.allocate(new_block_hash)
    assert (realloc_block != new_block)
    assert (new_block.block_hash == new_block_hash)
    assert (new_block.block_number == 2)
