"""Compare the with and without prefix caching.

Run `pytest tests/prefix_caching/test_prefix_caching.py`.
"""
from typing import List

import pytest

from tests.kernels.utils import override_backend_env_variable
from vllm.block import PhysicalTokenBlock
from vllm.core.block_manager_v1 import CachedBlockAllocator
from vllm.utils import Device

from ..models.utils import check_outputs_equal

MODELS = [
    "facebook/opt-125m",
]


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

    # Check metric: 1 hit of 2 queries
    assert block_allocator.get_prefix_cache_hit_rate() == 0.5

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

    # Allocate one more time to get 3/4 hit rate for easy checking
    block_allocator.allocate(block_hash, 0)
    assert block_allocator.get_prefix_cache_hit_rate() == 0.75


@pytest.mark.parametrize("num_blocks", [16])
def test_eviction(num_blocks: int, ):
    block_size = 16
    block_allocator = CachedBlockAllocator(Device.CPU, block_size, num_blocks)
    blocks: List[PhysicalTokenBlock] = []

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


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("backend", ["FLASH_ATTN", "FLASHINFER", "XFORMERS"])
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [5])
@pytest.mark.parametrize("cached_position", [0, 1])
@pytest.mark.parametrize("use_v2_block_manager", [False, True])
def test_mixed_requests(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    backend: str,
    dtype: str,
    max_tokens: int,
    cached_position: int,
    use_v2_block_manager: bool,
    monkeypatch,
) -> None:
    """
    Test the case when some sequences have the prefix cache hit
    and the others don't. The cached position determines where 
    the sequence is at among the batch of prefills.
    """
    override_backend_env_variable(monkeypatch, backend)

    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_greedy(example_prompts, max_tokens)

    cached_prompt = example_prompts[cached_position]
    with vllm_runner(
            model,
            dtype=dtype,
            enable_prefix_caching=True,
            use_v2_block_manager=use_v2_block_manager,
    ) as vllm_model:
        # Run the first prompt so the cache is populated
        vllm_outputs = vllm_model.generate_greedy([cached_prompt], max_tokens)

        # Run all the promopts
        vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens)

    check_outputs_equal(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )
