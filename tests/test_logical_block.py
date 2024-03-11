import pytest
import random
import math
from typing import List
from unittest.mock import MagicMock

from vllm.block import LogicalTokenBlock


@pytest.mark.parametrize("seed", list(range(10)))
@pytest.mark.parametrize("block_size", [1, 16])
@pytest.mark.parametrize("is_curr_block_full", [True, False])
def test_first_block_has_correct_content_hash(seed: int, block_size: int,
                                              is_curr_block_full: bool):
    """Verify a block which is first in the sequence has the correct hash.
    """
    random.seed(seed)

    block_with_prev = LogicalTokenBlock(block_number=2,
                                        block_size=block_size,
                                        previous_block=None)

    num_to_fill = block_size if is_curr_block_full else random.randint(
        0, block_size - 1)
    token_ids = list(range(num_to_fill))
    block_with_prev.append_tokens(token_ids)

    if is_curr_block_full:
        # Expect hash since block is full.
        assert block_with_prev.maybe_get_content_hash(
        ) == LogicalTokenBlock.get_content_hash(is_first_block=True,
                                                prev_block_hash=None,
                                                cur_block_token_ids=token_ids)
    else:
        # Do not expect hash since block is not full.
        assert block_with_prev.maybe_get_content_hash() is None


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

    previous_block = MagicMock(spec=LogicalTokenBlock)
    prev_block_hash = random.randint(0, 1000)
    previous_block.maybe_get_content_hash.return_value = (
        prev_block_hash if prev_block_has_hash else None)

    block_with_prev = LogicalTokenBlock(block_number=2,
                                        block_size=block_size,
                                        previous_block=previous_block)

    num_to_fill = block_size if is_curr_block_full else random.randint(
        0, block_size - 1)
    token_ids = list(range(num_to_fill))
    block_with_prev.append_tokens(token_ids)

    if is_curr_block_full and prev_block_has_hash:
        # Expect hash since block is full and previous block has hash.
        assert block_with_prev.maybe_get_content_hash(
        ) == LogicalTokenBlock.get_content_hash(
            is_first_block=False,
            prev_block_hash=prev_block_hash,
            cur_block_token_ids=token_ids)
    else:
        # Do not expect hash since block is not full or the previous block
        # does not have a hash.
        assert block_with_prev.maybe_get_content_hash() is None


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
        create_chain(block_size=block_size,
                     token_ids=token_ids,
                     num_empty_trailing_blocks=num_empty_trailing_blocks)
        for _ in range(2)
    ]

    for first_chain_block, second_chain_block in zip(first_chain,
                                                     second_chain):
        assert first_chain_block.maybe_get_content_hash(
        ) == second_chain_block.maybe_get_content_hash()

    if not first_chain or not second_chain:
        assert first_chain == second_chain
        assert num_tokens == 0


def create_chain(block_size: int,
                 token_ids: List[int],
                 num_empty_trailing_blocks=0) -> List[LogicalTokenBlock]:
    """Helper method which creates a chain of blocks.
    """
    blocks = []
    num_blocks = math.ceil(
        len(token_ids) / block_size) + num_empty_trailing_blocks

    if num_blocks == 0:
        return []

    prev_block = None
    for block_number in range(0, num_blocks):
        prev_block = LogicalTokenBlock(block_number=block_number,
                                       block_size=block_size,
                                       previous_block=prev_block)

        tokens_to_append = token_ids[block_number *
                                     block_size:(block_number + 1) *
                                     block_size]
        if tokens_to_append:
            prev_block.append_tokens(tokens_to_append)

        blocks.append(prev_block)

    return blocks
