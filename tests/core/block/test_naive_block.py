# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import pytest

from vllm.core.block.interfaces import Block, BlockAllocator
from vllm.core.block.naive_block import NaiveBlock, NaiveBlockAllocator


class TestNaiveBlockAllocator:

    @staticmethod
    def create_allocate_lambda(allocate_type: str,
                               allocator: NaiveBlockAllocator,
                               prev_block: Optional[Block],
                               token_ids: list[int]):
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

    @staticmethod
    @pytest.mark.parametrize("num_blocks", [4])
    @pytest.mark.parametrize("block_size", [8])
    def test_naive_block_get_num_full_blocks_touched(num_blocks, block_size):
        """ Verify the allocator can correctly return the number of
        full blocks touched.
        """
        allocator_src = NaiveBlockAllocator(create_block=NaiveBlock,
                                            num_blocks=num_blocks,
                                            block_size=block_size)
        allocator_dst = NaiveBlockAllocator(create_block=NaiveBlock,
                                            num_blocks=num_blocks,
                                            block_size=block_size)

        # Create a chain of cacheable blocks in the dst
        allocate_block = TestNaiveBlockAllocator.create_allocate_lambda(
            "immutable",
            allocator_src,
            prev_block=None,
            token_ids=list(range(block_size)))
        src_blocks = [allocate_block() for _ in range(num_blocks - 1)]

        # All blocks are cached
        assert allocator_dst.get_num_full_blocks_touched(
            src_blocks) == num_blocks - 1

        # Insert one non-full block in the src
        allocate_non_full_block = \
            TestNaiveBlockAllocator.create_allocate_lambda(
                "mutable", allocator_src,
                prev_block=src_blocks[-1],token_ids=[]
            )
        src_blocks.append(allocate_non_full_block())
        src_blocks[-1].append_token_ids([0])

        assert allocator_dst.get_num_full_blocks_touched(
            src_blocks) == num_blocks - 1
        # Fill up the last source block and then invoke
        # get_num_blocks_touched
        src_blocks[-1].append_token_ids([0] * (block_size - 1))
        assert allocator_dst.get_num_full_blocks_touched(
            src_blocks) == num_blocks
