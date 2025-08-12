"""Test hashing of cache blocks.

Run `pytest tests/test_cache_block_hashing.py`.
"""
import pytest

from vllm.transformers_utils.tokenizer import TokenizerGroup
from vllm.sequence import Sequence

# Make two prefixes with different first blocks.
prefix_start = [("You are an expert"), ("You are a")]
prefix_common = (
    " school principal, skilled in effectively managing "
    "faculty and staff. Draft 10-15 questions for a potential first grade "
    "Head Teacher for my K-12, all-girls', independent school that emphasizes "
    "community, joyful discovery, and life-long learning. The candidate is "
    "coming in for a first-round panel interview for a 8th grade Math "
    "teaching role. They have 5 years of previous teaching experience "
    "as an assistant teacher at a co-ed, public school with experience "
    "in middle school math teaching. Based on this, fulfill "
    "the following: ")
prefixes = [start + prefix_common for start in prefix_start]

# Sample prompts.
sample_prompts = [
    "Hello, my name is", "The president of the United States is",
    "The capital of France is", "The future of AI is"
]


# Helper function.
def flatten_2d(li):
    return [lss for ls in li for lss in ls]


@pytest.mark.parametrize("model", ["facebook/opt-125m"])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("max_num_seqs", [256])
def test_auto_prefix_caching(model: str, block_size: int, max_num_seqs: int):

    tokenizer = TokenizerGroup(
        tokenizer_id="facebook/opt-125m",
        enable_lora=False,
        max_num_seqs=max_num_seqs,
        max_input_length=None,
    )

    hashes = []

    for prefix in prefixes:
        hashes.append([])
        prompts = [prefix + prompt for prompt in sample_prompts]
        seq_id = 0
        for prompt in prompts:
            hashes[-1].append([])
            prompt_token_ids = tokenizer.encode(prompt)
            seq = Sequence(seq_id, prompt, prompt_token_ids, block_size)

            num_blocks = len(prompt_token_ids) // block_size
            for idx in range(num_blocks):
                hashes[-1][-1].append(seq.hash_of_block(idx))

            seq_id += 1

    # Check that hashes made with two prefixes with different first blocks are
    # different everywhere.
    for hash0, hash1 in zip(flatten_2d(hashes[0]), flatten_2d(hashes[1])):
        assert (hash0 != hash1)

    # Check that hashes of different prompts made with the same prefix are the
    # same until the hashes that contain the prompt.
    for hash_pref in hashes:
        same_hashes = [tuple(h[:-1]) for h in hash_pref]
        different_hashes = [h[-1] for h in hash_pref]
        assert (len(set(same_hashes)) == 1)
        assert (len(set(different_hashes)) == len(different_hashes))
