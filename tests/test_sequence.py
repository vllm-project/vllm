import random
import pytest

from vllm.sequence import Sequence

@pytest.mark.parametrize("block_size", [1, 16, 256])
@pytest.mark.parametrize("prompt_len", [1, 1024])
def test_prefix_hash_equality(block_size: int, prompt_len: int):
    random.seed(0)
    prompt_token_ids = [random.randint(0, 50_000) for _ in range(prompt_len)]
    
    first_seq, second_seq = [Sequence(
       seq_id=i,
       prompt="",
       prompt_token_ids=prompt_token_ids,
       block_size=block_size,
    ) for i in range(2)]
    
    for token_index in range(0, len(prompt_token_ids), block_size):
        block_index = token_index // block_size
        assert first_seq.maybe_get_hash_of_block(block_index) == second_seq.maybe_get_hash_of_block(block_index)
