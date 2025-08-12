import math
import random
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

from vllm.core.block.interfaces import Block, BlockAllocator
from vllm.core.block.prefix_caching_block import (PrefixCachingBlock,
                                                  PrefixCachingBlockAllocator)


class TestPrefixCachingBlock:

    @staticmethod
    @pytest.mark.parametrize("seed", list(range(10)))
    @pytest.mark.parametrize("block_size", [1, 16])
    @pytest.mark.parametrize("is_curr_block_full", [True, False])
    def test_first_block_has_correct_content_hash(seed: int, block_size: int,
                                                  is_curr_block_full: bool):
        """Verify a block which is first in the sequence has the correct hash.
        """
        random.seed(seed)
        num_to_fill = block_size if is_curr_block_full else random.randint(
            0, block_size - 1)
        token_ids = list(range(num_to_fill))
        mock_allocator = MagicMock(spec=PrefixCachingBlockAllocator)

        block_with_prev = PrefixCachingBlock(
            prev_block=None,
            token_ids=token_ids,
            block_size=block_size,
            prefix_caching_allocator=mock_allocator)

        if is_curr_block_full:
            # Expect hash since block is full.
            assert block_with_prev.content_hash == (
                PrefixCachingBlock.hash_block_tokens(
                    is_first_block=True,
                    prev_block_hash=None,
                    cur_block_token_ids=token_ids))
        else:
            # Do not expect hash since block is not full.
            assert block_with_prev.content_hash is None

    @staticmethod
    @pytest.mark.parametrize("seed", list(range(10)))
    @pytest.mark.parametrize("block_size", [1, 16])
    @pytest.mark.parametrize("is_curr_block_full", [True, False])
    @pytest.mark.parametrize("prev_block_has_hash", [True, False])
    def test_nth_block_has_correct_content_hash(seed: int, block_size: int,
                                                is_curr_block_full: bool,
                                                prev_block_has_hash: bool):
        """Verify a block which is not first in the sequence has the correct
        hash.
        """

        random.seed(seed)

        previous_block = MagicMock(spec=PrefixCachingBlock)
        prev_block_hash = random.randint(0, 1000)
        previous_block.content_hash = (prev_block_hash
                                       if prev_block_has_hash else None)

        num_to_fill = block_size if is_curr_block_full else random.randint(
            0, block_size - 1)
        token_ids = list(range(num_to_fill))
        mock_allocator = MagicMock(spec=PrefixCachingBlockAllocator)

        block_with_prev = PrefixCachingBlock(
            prev_block=previous_block,
            token_ids=token_ids,
            block_size=block_size,
            prefix_caching_allocator=mock_allocator,
        )

        if is_curr_block_full and prev_block_has_hash:
            # Expect hash since block is full and previous block has hash.
            assert (block_with_prev.content_hash ==
                    PrefixCachingBlock.hash_block_tokens(
                        is_first_block=False,
                        prev_block_hash=prev_block_hash,
                        cur_block_token_ids=token_ids))
        else:
            # Do not expect hash since block is not full or the previous block
            # does not have a hash.
            assert block_with_prev.content_hash is None

    @staticmethod
    @pytest.mark.parametrize("block_size", [1, 2, 16])
    @pytest.mark.parametrize("num_tokens", list(range(3)))
    @pytest.mark.parametrize("num_empty_trailing_blocks", [0, 1, 10])
    def test_blocks_have_correct_hash_in_chain(block_size: int,
                                               num_tokens: int,
                                               num_empty_trailing_blocks: int):
        """Create two chains of logical blocks with the same contents.
        Assert the hashes are equal.
        """
        random.seed(0)

        token_ids = [random.randint(0, 50_000) for _ in range(num_tokens)]

        first_chain, second_chain = [
            TestPrefixCachingBlock.create_chain(
                block_size=block_size,
                token_ids=token_ids,
                num_empty_trailing_blocks=num_empty_trailing_blocks)
            for _ in range(2)
        ]

        for first_chain_block, second_chain_block in zip(
                first_chain, second_chain):
            assert (first_chain_block.content_hash ==
                    second_chain_block.content_hash)

        if not first_chain or not second_chain:
            assert first_chain == second_chain
            assert num_tokens == 0

    @staticmethod
    def create_chain(block_size: int,
                     token_ids: List[int],
                     num_empty_trailing_blocks=0) -> List[PrefixCachingBlock]:
        """Helper method which creates a chain of blocks.
        """
        blocks = []
        num_blocks = math.ceil(
            len(token_ids) / block_size) + num_empty_trailing_blocks

        if num_blocks == 0:
            return []

        allocator = MagicMock(spec=PrefixCachingBlockAllocator)

        prev_block = None
        for block_number in range(0, num_blocks):
            prev_block = PrefixCachingBlock(
                prev_block=prev_block,
                token_ids=[],
                block_size=block_size,
                prefix_caching_allocator=allocator,
            )

            tokens_to_append = token_ids[block_number *
                                         block_size:(block_number + 1) *
                                         block_size]
            if tokens_to_append:
                prev_block.append_token_ids(tokens_to_append)

            blocks.append(prev_block)

        return blocks


class TestPrefixCachingBlockAllocator:

    @staticmethod
    def create_allocate_lambda(allocate_type: str, allocator: BlockAllocator,
                               prev_block: Optional[Block],
                               token_ids: List[int]):
        if allocate_type == "immutable":
            allocate_block = lambda: allocator.allocate_immutable(
                prev_block=prev_block, token_ids=token_ids)
        elif allocate_type == "mutable":
            allocate_block = lambda: allocator.allocate_mutable(prev_block=
                                                                prev_block)
        else:
            raise ValueError()

        return allocate_block

    @staticmethod
    @pytest.mark.parametrize("num_blocks", [1, 1024])
    @pytest.mark.parametrize("block_size", [1, 16])
    def test_allocate_mutable_ooms(num_blocks: int, block_size: int):
        allocator = PrefixCachingBlockAllocator(num_blocks=num_blocks,
                                                block_size=block_size)
        allocate_block = TestPrefixCachingBlockAllocator.create_allocate_lambda(
            allocate_type="mutable",
            allocator=allocator,
            prev_block=None,
            token_ids=list(range(block_size)),
        )

        [allocate_block() for _ in range(num_blocks)]
        with pytest.raises(BlockAllocator.NoFreeBlocksError):
            allocate_block()

    @staticmethod
    @pytest.mark.parametrize("num_blocks", [1, 1024])
    @pytest.mark.parametrize("block_size", [1, 16])
    def test_allocate_immutable_does_not_oom_single_hash(
            num_blocks: int, block_size: int):
        allocator = PrefixCachingBlockAllocator(num_blocks=num_blocks,
                                                block_size=block_size)
        allocate_block = TestPrefixCachingBlockAllocator.create_allocate_lambda(
            allocate_type="immutable",
            allocator=allocator,
            prev_block=None,
            token_ids=list(range(block_size)),
        )

        blocks = [allocate_block() for _ in range(num_blocks)]

        # Expect no OOM. If these were mutable blocks, this would OOM.
        non_oom_block = allocate_block()

        # Expect all blocks to have same physical block index.
        for block in blocks:
            assert (block.block_id == non_oom_block.block_id)

    @staticmethod
    @pytest.mark.parametrize("num_blocks", [1, 1024])
    @pytest.mark.parametrize("block_size", [1, 16])
    def test_allocate_immutable_ooms_many_hash(num_blocks: int,
                                               block_size: int):
        """Consume all blocks using many different hashes/block content.

        Do this by creating a sequence that is very long.
        Expect next block to OOM.
        """
        allocator = PrefixCachingBlockAllocator(num_blocks=num_blocks,
                                                block_size=block_size)

        # Create token ids that will exhaust all blocks.
        token_ids = list(range(num_blocks * block_size))

        chain = TestPrefixCachingBlockAllocator.create_immutable_chain(
            block_size=block_size,
            token_ids=token_ids,
            allocator=allocator,
        )

        # Expect allocation with unseen hash to fail.
        with pytest.raises(BlockAllocator.NoFreeBlocksError):
            allocator.allocate_immutable(prev_block=chain[-1],
                                         token_ids=list(range(block_size)))

        # Expect mutable allocation to fail.
        with pytest.raises(BlockAllocator.NoFreeBlocksError):
            allocator.allocate_mutable(prev_block=chain[-1])

        # Expect allocation of exact same chain to pass.
        second_chain = TestPrefixCachingBlockAllocator.create_immutable_chain(
            block_size=block_size,
            token_ids=token_ids,
            allocator=allocator,
        )

        # Expect physical block indices to be the same in both chains.
        assert chain and second_chain
        for first_chain_block, second_chain_block in zip(chain, second_chain):
            assert (first_chain_block.block_id == second_chain_block.block_id)

    @staticmethod
    @pytest.mark.parametrize("num_blocks", [1, 1024])
    @pytest.mark.parametrize("block_size", [1, 16])
    def test_free_prevents_oom(num_blocks: int, block_size: int):
        allocator = PrefixCachingBlockAllocator(num_blocks=num_blocks,
                                                block_size=block_size)

        # Create token ids that will exhaust all blocks.
        token_ids = list(range(num_blocks * block_size))

        chain = TestPrefixCachingBlockAllocator.create_immutable_chain(
            block_size=block_size,
            token_ids=token_ids,
            allocator=allocator,
        )

        # Expect mutable allocation to fail.
        with pytest.raises(BlockAllocator.NoFreeBlocksError):
            allocator.allocate_mutable(prev_block=None)

        block_to_free = chain[-1]

        # Expect free/allocate loop to succeed many times.
        for i in range(100):
            block_id = block_to_free.block_id
            allocator.free(block_to_free)
            assert block_to_free.block_id is None, i

            new_block = allocator.allocate_mutable(prev_block=None)
            assert new_block.block_id == block_id, i

            with pytest.raises(BlockAllocator.NoFreeBlocksError):
                allocator.allocate_mutable(prev_block=None)

            block_to_free = new_block

    @staticmethod
    @pytest.mark.parametrize("num_blocks", [1024])
    @pytest.mark.parametrize("block_size", [16])
    @pytest.mark.parametrize("seed", list(range(20)))
    def test_get_num_free_blocks(num_blocks: int, block_size: int, seed: int):
        random.seed(seed)
        allocator = PrefixCachingBlockAllocator(num_blocks=num_blocks,
                                                block_size=block_size)
        num_blocks_to_consume = random.randint(1, num_blocks - 1)

        # Create token ids that will exhaust all blocks.
        token_ids = list(range(num_blocks_to_consume * block_size))

        chain = TestPrefixCachingBlockAllocator.create_immutable_chain(
            block_size=block_size,
            token_ids=token_ids,
            allocator=allocator,
        )

        # Free each block in chain, assert num free blocks includes new free
        # block.
        for i, block in enumerate(chain):
            assert allocator.get_num_free_blocks() == (num_blocks -
                                                       num_blocks_to_consume +
                                                       i)
            allocator.free(block)

    @staticmethod
    @pytest.mark.parametrize("num_blocks", [1024])
    @pytest.mark.parametrize("block_size", [16])
    @pytest.mark.parametrize("seed", list(range(20)))
    def test_get_num_free_blocks_shared(num_blocks: int, block_size: int,
                                        seed: int):
        """Verify sharing occurs by allocating two sequences that share prefixes
        and incrementally freeing blocks.
        """
        random.seed(seed)
        allocator = PrefixCachingBlockAllocator(num_blocks=num_blocks,
                                                block_size=block_size)
        num_blocks_to_consume = random.randint(1, num_blocks - 1)

        # Create token ids that will exhaust all blocks.
        token_ids = list(range(num_blocks_to_consume * block_size))

        first_chain = TestPrefixCachingBlockAllocator.create_immutable_chain(
            block_size=block_size,
            token_ids=token_ids,
            allocator=allocator,
        )
        second_chain = TestPrefixCachingBlockAllocator.create_immutable_chain(
            block_size=block_size,
            token_ids=token_ids,
            allocator=allocator,
        )

        # Free each block in the first chain. Since all blocks are shared, the
        # free count should stay constant.
        for i, block in enumerate(first_chain):
            assert allocator.get_num_free_blocks() == (num_blocks -
                                                       num_blocks_to_consume)
            allocator.free(block)

        # Free each block in the second chain. Since the refcount is now zero,
        # the free count should increment with each free.
        for i, block in enumerate(second_chain):
            assert allocator.get_num_free_blocks() == (num_blocks -
                                                       num_blocks_to_consume +
                                                       i)
            allocator.free(block)

    @staticmethod
    def create_immutable_chain(
        block_size: int,
        token_ids: List[int],
        allocator: PrefixCachingBlockAllocator,
    ) -> List[PrefixCachingBlock]:
        """Helper method which creates a chain of blocks.
        """
        blocks = []
        num_blocks = math.ceil(len(token_ids) / block_size)

        if num_blocks == 0:
            return []

        prev_block = None
        for block_number in range(0, num_blocks):
            block_token_ids = token_ids[block_number *
                                        block_size:(block_number + 1) *
                                        block_size]
            prev_block = allocator.allocate_immutable(
                prev_block=prev_block, token_ids=block_token_ids)
            blocks.append(prev_block)

        return blocks
