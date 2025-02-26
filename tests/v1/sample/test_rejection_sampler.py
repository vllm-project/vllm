# SPDX-License-Identifier: Apache-2.0
from typing import List

import pytest
import torch

from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import INVALID_TOKEN_ID, RejectionSampler


@pytest.fixture
def sampler():
    return RejectionSampler()


def create_logits_tensor(token_ids: List[int],
                         vocab_size: int = 100) -> torch.Tensor:
    """Helper function to create logits tensor that 
       will produce desired token ids on argmax"""
    logits = torch.full((len(token_ids), vocab_size), -100.0).cuda()
    for i, token_id in enumerate(token_ids):
        logits[i, token_id] = 100.0
    return logits


def create_sampling_metadata(spec_tokens: List[List[int]]) -> SamplingMetadata:
    batch_size = len(spec_tokens)
    return SamplingMetadata(
        temperature=torch.tensor([]),
        all_greedy=True,
        all_random=False,
        top_p=None,
        top_k=None,
        min_p=torch.empty(batch_size, ),
        generators={},
        max_num_logprobs=0,
        no_penalties=False,
        prompt_token_ids=None,
        frequency_penalties=torch.tensor([]),
        presence_penalties=torch.tensor([]),
        repetition_penalties=torch.tensor([]),
        output_token_ids=[],
        min_tokens={},
        logit_bias=[None] * batch_size,
        allowed_token_ids_mask=None,
    )


def test_perfect_match(sampler):
    """Test when output tokens perfectly match speculated tokens"""
    spec_tokens = [[1, 2, 3]]
    output_tokens = [1, 2, 3, 4]  # 4 is the bonus token

    metadata = create_sampling_metadata(spec_tokens)
    logits = create_logits_tensor(output_tokens)

    output = sampler(spec_tokens, logits, metadata)
    expected = torch.tensor([[1, 2, 3, 4]],
                            dtype=torch.int,
                            device=logits.device)
    assert torch.equal(output.sampled_token_ids, expected)


def test_early_mismatch(sampler):
    """Test when there's an early mismatch in tokens"""
    spec_tokens = [[1, 2, 3]]
    output_tokens = [1, 5, 3, 4]  # Mismatch at position 1

    metadata = create_sampling_metadata(spec_tokens)
    logits = create_logits_tensor(output_tokens)

    output = sampler(spec_tokens, logits, metadata)
    expected = torch.tensor([[1, 5, INVALID_TOKEN_ID, INVALID_TOKEN_ID]],
                            dtype=torch.int,
                            device=logits.device)
    assert torch.equal(output.sampled_token_ids, expected)


def test_multiple_sequences(sampler):
    """Test handling multiple sequences of speculated tokens"""
    spec_tokens = [[1, 2], [3]]
    output_tokens = [1, 2, 5, 3, 4]  # Two sequences with bonus tokens 5 and 4

    metadata = create_sampling_metadata(spec_tokens)
    logits = create_logits_tensor(output_tokens)

    output = sampler(spec_tokens, logits, metadata)
    expected = torch.tensor([[1, 2, 5], [3, 4, INVALID_TOKEN_ID]],
                            dtype=torch.int,
                            device=logits.device)
    assert torch.equal(output.sampled_token_ids, expected)


def test_single_token_sequence(sampler):
    """Test handling sequences with single token"""
    spec_tokens = [[1]]
    output_tokens = [1, 2]  # Single token with bonus token 2

    metadata = create_sampling_metadata(spec_tokens)
    logits = create_logits_tensor(output_tokens)

    output = sampler(spec_tokens, logits, metadata)
    expected = torch.tensor([[1, 2]], dtype=torch.int, device=logits.device)
    assert torch.equal(output.sampled_token_ids, expected)


def test_empty_sequence(sampler):
    """Test handling empty sequence of speculated tokens"""
    spec_tokens: List[List[int]] = [[]]
    output_tokens = [5]  # Just the bonus token

    metadata = create_sampling_metadata(spec_tokens)
    logits = create_logits_tensor(output_tokens)

    output = sampler(spec_tokens, logits, metadata)
    expected = torch.tensor([[5]], dtype=torch.int, device=logits.device)
    assert torch.equal(output.sampled_token_ids, expected)


def test_multiple_mismatches(sampler):
    """Test handling multiple sequences with mismatches"""
    spec_tokens = [[1, 2, 3], [4, 5, 6]]
    output_tokens = [1, 2, 7, 6, 4, 8, 6, 9]  # Mismatches in both sequences

    metadata = create_sampling_metadata(spec_tokens)
    logits = create_logits_tensor(output_tokens)

    output = sampler(spec_tokens, logits, metadata)
    expected = torch.tensor([[1, 2, 7, INVALID_TOKEN_ID],
                             [4, 8, INVALID_TOKEN_ID, INVALID_TOKEN_ID]],
                            dtype=torch.int,
                            device=logits.device)
    assert torch.equal(output.sampled_token_ids, expected)


@pytest.mark.parametrize(
    "spec_tokens,output_tokens,expected",
    [
        ([[1, 2]], [1, 2, 3], [[1, 2, 3]]),  # Perfect match with bonus
        ([[1]], [2, 3], [[2, INVALID_TOKEN_ID]]),  # First mismatch
        ([[1, 2], [3, 4]], [1, 5, 6, 3, 4, 7], [[1, 5, INVALID_TOKEN_ID],
                                                [3, 4, 7]]),  # Mixed matches
    ])
def test_parametrized_cases(sampler, spec_tokens, output_tokens, expected):
    """Parametrized test for various matching scenarios"""
    metadata = create_sampling_metadata(spec_tokens)
    logits = create_logits_tensor(output_tokens)

    output = sampler(spec_tokens, logits, metadata)
    expected_tensor = torch.tensor(expected,
                                   dtype=torch.int,
                                   device=logits.device)
    assert torch.equal(output.sampled_token_ids, expected_tensor)


def test_logits_shape_handling(sampler):
    """Test handling of different logits tensor shapes"""
    spec_tokens = [[1, 2]]
    output_tokens = [1, 2, 3]
    vocab_size = 1000

    metadata = create_sampling_metadata(spec_tokens)
    logits = create_logits_tensor(output_tokens, vocab_size)

    output = sampler(spec_tokens, logits, metadata)
    expected = torch.tensor([[1, 2, 3]], dtype=torch.int, device=logits.device)
    assert torch.equal(output.sampled_token_ids, expected)
    assert logits.shape[-1] == vocab_size
