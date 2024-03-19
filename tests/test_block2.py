import random
import pytest
from typing import Optional, List
import random
from unittest.mock import MagicMock
import math

from vllm.block2 import NaiveBlockAllocator, NaiveBlock, BlockAllocator, Block
from vllm.block2 import RefCounter
from vllm.block2 import PrefixCachingBlock


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
    
        block_with_prev = PrefixCachingBlock(prev_block=None, token_ids=token_ids, block_size=block_size)
    
        if is_curr_block_full:
            # Expect hash since block is full.
            assert block_with_prev.content_hash == PrefixCachingBlock.hash_block_tokens(is_first_block=True,
                                                    prev_block_hash=None,
                                                    cur_block_token_ids=token_ids)
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
        """Verify a block which is not first in the sequence has the correct hash.
        """

        random.seed(seed)
    
        previous_block = MagicMock(spec=PrefixCachingBlock)
        prev_block_hash = random.randint(0, 1000)
        previous_block.content_hash = (
            prev_block_hash if prev_block_has_hash else None)

        num_to_fill = block_size if is_curr_block_full else random.randint(
            0, block_size - 1)
        token_ids = list(range(num_to_fill))
    
        block_with_prev = PrefixCachingBlock(prev_block=previous_block,
                                            token_ids=token_ids,
                                            block_size=block_size,
                                            )
    
    
        if is_curr_block_full and prev_block_has_hash:
            # Expect hash since block is full and previous block has hash.
            assert block_with_prev.content_hash == PrefixCachingBlock.hash_block_tokens(
                is_first_block=False,
                prev_block_hash=prev_block_hash,
                cur_block_token_ids=token_ids)
        else:
            # Do not expect hash since block is not full or the previous block
            # does not have a hash.
            assert block_with_prev.content_hash is None

    @staticmethod
    @pytest.mark.parametrize("block_size", [1, 2, 16])
    @pytest.mark.parametrize("num_tokens", list(range(3)))
    @pytest.mark.parametrize("num_empty_trailing_blocks", [0, 1, 10])
    def test_blocks_have_correct_hash_in_chain(block_size: int, num_tokens: int,
                                               num_empty_trailing_blocks: int):
        """Create two chains of logical blocks with the same contents.
        Assert the hashes are equal.
        """
        random.seed(0)
    
        token_ids = [random.randint(0, 50_000) for _ in range(num_tokens)]
    
        first_chain, second_chain = [
            TestPrefixCachingBlock.create_chain(block_size=block_size,
                         token_ids=token_ids,
                         num_empty_trailing_blocks=num_empty_trailing_blocks)
            for _ in range(2)
        ]
    
        for first_chain_block, second_chain_block in zip(first_chain,
                                                         second_chain):
            assert first_chain_block.content_hash == second_chain_block.content_hash
    
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
    
        prev_block = None
        for block_number in range(0, num_blocks):
            prev_block = PrefixCachingBlock(
                                           prev_block=prev_block,
                                           token_ids=[],
                                           block_size=block_size,
                                           )
    
            tokens_to_append = token_ids[block_number *
                                         block_size:(block_number + 1) *
                                         block_size]
            if tokens_to_append:
                prev_block.append_token_ids(tokens_to_append)
    
            blocks.append(prev_block)
    
        return blocks

#class TestPrefixCachingBlockAllocator:
#    
#    @staticmethod
#    def test_allocate_with_cache():
#        allocator = PrefixCachingBlockAllocator()
#
#
