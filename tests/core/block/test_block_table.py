import random
import pytest
from typing import Optional, List
import random
from unittest.mock import MagicMock
import math

from vllm.core.block.block_space_manager import BlockSpaceManager, AllocStatus
from ..utils import create_seq_group
#from vllm.core.block.interfaces import NaiveBlockAllocator, NaiveBlock, BlockAllocator, Block
#from vllm.block2 import RefCounter
#from vllm.block2 import PrefixCachingBlock, PrefixCachingBlockAllocator
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
        assert allocator.get_num_free_blocks(device=Device.GPU) == num_gpu_blocks - i * num_blocks_per_alloc

        block_tables.append(BlockTable(
            sequence_id=0,
            token_ids=token_ids,
            block_size=block_size,
            block_allocator=allocator,
        ))
        block_tables[-1].allocate(device=Device.GPU)

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
    num_mutable_blocks_per_alloc = 0 if len(chunked_tokens[-1]) == block_size else 1
    num_immutable_blocks_per_alloc = len(chunked_tokens) - num_mutable_blocks_per_alloc
    
    block_tables = []
    for alloc_i in range(1, 6):

        block_tables.append(BlockTable(
            sequence_id=0,
            token_ids=token_ids,
            block_size=block_size,
            block_allocator=allocator,
        ))
        block_tables[-1].allocate(device=Device.GPU)
        
        # Expect all sequences to share allocations, except for their last block (which may be mutable).
        assert allocator.get_num_free_blocks(device=Device.GPU) == num_gpu_blocks - (num_immutable_blocks_per_alloc + num_mutable_blocks_per_alloc * (alloc_i))
