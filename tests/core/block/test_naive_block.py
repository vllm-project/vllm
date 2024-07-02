from typing import List, Optional

import pytest

from vllm.core.block.interfaces import Block, BlockAllocator
from vllm.core.block.naive_block import NaiveBlock, NaiveBlockAllocator


class TestNaiveBlockAllocator:

    @staticmethod
    def create_allocate_lambda(allocate_type: str,
                               allocator: NaiveBlockAllocator,
                               prev_block: Optional[Block],
                               token_ids: List[int]):
        if allocate_type == "immutable":
            allocate_block = lambda: allocator.allocate_immutable_block(
                prev_block=prev_block, token_ids=token_ids)
        elif allocate_type == "mutable":
            allocate_block = lambda: allocator.allocate_mutable_block(
                prev_block=prev_block)
        else:
            raise ValueError()

        return allocate_block

    @staticmethod
    @pytest.mark.parametrize("allocate_type", ["immutable", "mutable"])
    @pytest.mark.parametrize("num_blocks", [1, 1024])
    @pytest.mark.parametrize("block_size", [1, 16])
    def test_allocate_ooms(allocate_type: str, num_blocks: int,
                           block_size: int):
        allocator = NaiveBlockAllocator(create_block=NaiveBlock,
                                        num_blocks=num_blocks,
                                        block_size=block_size)
        allocate_block = TestNaiveBlockAllocator.create_allocate_lambda(
            allocate_type,
            allocator,
            prev_block=None,
            token_ids=list(range(block_size)))

        [allocate_block() for _ in range(num_blocks)]
        with pytest.raises(BlockAllocator.NoFreeBlocksError):
            allocate_block()

    @staticmethod
    @pytest.mark.parametrize("allocate_type", ["immutable", "mutable"])
    @pytest.mark.parametrize("num_blocks", [1, 1024])
    @pytest.mark.parametrize("block_size", [1, 16])
    def test_free_prevents_oom(allocate_type: str, num_blocks: int,
                               block_size: int):
        allocator = NaiveBlockAllocator(create_block=NaiveBlock,
                                        num_blocks=num_blocks,
                                        block_size=block_size)
        allocate_block = TestNaiveBlockAllocator.create_allocate_lambda(
            allocate_type,
            allocator,
            prev_block=None,
            token_ids=list(range(block_size)))

        blocks = [allocate_block() for _ in range(num_blocks)]

        with pytest.raises(BlockAllocator.NoFreeBlocksError):
            allocate_block()

        block_to_free = blocks.pop()

        for _ in range(100):
            block_id = block_to_free.block_id
            allocator.free(block_to_free)
            assert block_to_free.block_id is None

            new_block = allocate_block()
            assert new_block.block_id == block_id

            with pytest.raises(BlockAllocator.NoFreeBlocksError):
                allocate_block()

            block_to_free = new_block

    @staticmethod
    @pytest.mark.parametrize("allocate_type", ["immutable", "mutable"])
    @pytest.mark.parametrize("num_blocks", [1024])
    @pytest.mark.parametrize("block_size", [16])
    def test_get_num_free_blocks(allocate_type: str, num_blocks: int,
                                 block_size: int):
        allocator = NaiveBlockAllocator(create_block=NaiveBlock,
                                        num_blocks=num_blocks,
                                        block_size=block_size)
        allocate_block = TestNaiveBlockAllocator.create_allocate_lambda(
            allocate_type,
            allocator,
            prev_block=None,
            token_ids=list(range(block_size)))

        assert allocator.get_num_free_blocks() == num_blocks

        blocks = [allocate_block() for _ in range(num_blocks)]

        for i, block in enumerate(blocks):
            assert allocator.get_num_free_blocks() == i
            allocator.free(block)
