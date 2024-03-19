import random
import pytest
from typing import Optional, List
import random

from vllm.block2 import NaiveBlockAllocator, NaiveBlock, BlockAllocator, Block
from vllm.block2 import RefCounter

class TestRefCounter:

    @staticmethod
    @pytest.mark.parametrize("seed", list(range(20)))
    @pytest.mark.parametrize("num_incrs", [1, 100])
    @pytest.mark.parametrize("num_blocks", [1024])
    def test_incr(seed: int, num_incrs: int, num_blocks: int):
        random.seed(seed)

        all_block_indices = list(range(num_blocks))
        counter = RefCounter(all_block_indices=all_block_indices)
        
        block_index = random.randint(0, num_blocks - 1)
        for i in range(num_incrs):
            value = counter.incr(block_index)
            assert value == i + 1

    @staticmethod
    @pytest.mark.parametrize("seed", list(range(20)))
    @pytest.mark.parametrize("num_incrs", [1, 100])
    @pytest.mark.parametrize("num_blocks", [1024])
    def test_incr_decr(seed: int, num_incrs: int, num_blocks: int):
        random.seed(seed)

        all_block_indices = list(range(num_blocks))
        counter = RefCounter(all_block_indices=all_block_indices)
        
        block_index = random.randint(0, num_blocks - 1)
        for i in range(num_incrs):
            value = counter.incr(block_index)
            assert value == i + 1

        for i in range(num_incrs):
            value = counter.decr(block_index)
            assert value == num_incrs - (i + 1)

        with pytest.raises(AssertionError):
            counter.decr(block_index)

class TestNaiveBlockAllocator:
    
    @staticmethod
    def create_allocate_lambda(allocate_type: str, allocator: NaiveBlockAllocator, prev_block: Optional[Block], token_ids: List[int]):
        if allocate_type == "immutable":
            allocate_block = lambda: allocator.allocate_immutable(prev_block=prev_block, token_ids=token_ids)
        elif allocate_type == "mutable":
            allocate_block = lambda: allocator.allocate_mutable(prev_block=prev_block)
        else:
            raise ValueError()

        return allocate_block

    @staticmethod
    @pytest.mark.parametrize("allocate_type", ["immutable", "mutable"])
    @pytest.mark.parametrize("num_blocks", [1, 1024])
    @pytest.mark.parametrize("block_size", [1, 16])
    def test_allocate_ooms(allocate_type: str, num_blocks: int, block_size: int):
        allocator = NaiveBlockAllocator(block_cls=NaiveBlock, num_blocks=num_blocks, block_size=block_size)
        allocate_block = TestNaiveBlockAllocator.create_allocate_lambda(allocate_type, allocator, prev_block=None, token_ids=list(range(block_size)))
        
        blocks = [allocate_block() for _ in range(num_blocks)]
        with pytest.raises(BlockAllocator.NoFreeBlocksError):
            oom_block = allocate_block()

    @staticmethod
    @pytest.mark.parametrize("allocate_type", ["immutable", "mutable"])
    @pytest.mark.parametrize("num_blocks", [1, 1024])
    @pytest.mark.parametrize("block_size", [1, 16])
    def test_free_prevents_oom(allocate_type: str, num_blocks: int, block_size: int):
        allocator = NaiveBlockAllocator(block_cls=NaiveBlock, num_blocks=num_blocks, block_size=block_size)
        allocate_block = TestNaiveBlockAllocator.create_allocate_lambda(allocate_type, allocator, prev_block=None, token_ids=list(range(block_size)))
        
        blocks = [allocate_block() for _ in range(num_blocks)]

        with pytest.raises(BlockAllocator.NoFreeBlocksError):
            oom_block = allocate_block()
        
        block_to_free = blocks.pop()

        for _ in range(100):
            physical_block_index = block_to_free.physical_block_index
            allocator.free(block_to_free)
            assert block_to_free.physical_block_index is None

            new_block = allocate_block()
            assert new_block.physical_block_index == physical_block_index

            with pytest.raises(BlockAllocator.NoFreeBlocksError):
                oom_block = allocate_block()

            block_to_free = new_block
