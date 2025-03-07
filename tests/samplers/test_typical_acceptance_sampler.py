# SPDX-License-Identifier: Apache-2.0
"""Tests for rejection sampling."""

import pytest
import torch

from vllm.model_executor.layers.typical_acceptance_sampler import (
    TypicalAcceptanceSampler)
from vllm.model_executor.utils import set_random_seed

CUDA_DEVICES = [f"cuda:{i}" for i in range(1)]


def get_zero_temperature_prob_dist(batch_size, k, vocab_size):
    """
    Generates a fake temperature zero probability distribution.
    Returns:
        1. A fake temperature zero probability distribution of shape
           [batch_size, k, vocab_size]
        2. Tensor of shape [batch_size, k] containing the token ids 
           of the probability 1.0 tokens at each position.
    """
    # Simulate temperature 0 probability distribution for target probabilities
    # and create target probabilities such that only 1 token id has
    # probability 1.0
    target_probs = torch.rand(batch_size, k, vocab_size, dtype=torch.float32)
    probs = torch.rand(batch_size, k, vocab_size)
    _, zero_temperature_token_ids = torch.max(probs, dim=-1)
    # set the probability of the tokens with ids in zero_temperature_token_ids
    # to 1 and the rest to 0.
    target_probs = torch.zeros_like(probs).scatter_(
        -1, zero_temperature_token_ids.unsqueeze(-1), 1.0)
    return target_probs, zero_temperature_token_ids


def get_draft_token_ids(batch_size: int, k: int, vocab_size: int,
                        token_ids_to_exclude: torch.Tensor):
    """
    Returns a tensor of shape [batch_size, k] of fake draft token ids
    drawn randomly from a vocab of size vocab_size. We however ensure
    that token_ids from token_ids_to_exclude are excluded at the 
    corresponding positions.
    """
    draft_token_ids = torch.empty(batch_size, k, dtype=torch.long)
    for i in range(batch_size):
        for j in range(k):
            # Generate a random token ID excluding token_ids_to_exclude[i, j]
            while True:
                token_id = torch.randint(0, vocab_size, (1, )).item()
                if token_id != token_ids_to_exclude[i, j]:
                    draft_token_ids[i, j] = token_id
                    break
    return draft_token_ids


def get_acceptance_sampler(
    posterior_threshold: float = 0.03,
    posterior_alpha: float = 0.9,
    strict_mode: bool = False,
) -> TypicalAcceptanceSampler:
    """
    Initializes and returns a TypicalAcceptanceSampler.
    """
    return TypicalAcceptanceSampler(posterior_threshold, posterior_alpha,
                                    strict_mode)


@pytest.mark.parametrize("k", list(range(1, 6)))
@pytest.mark.parametrize("vocab_size", [30_000, 50_000])
@pytest.mark.parametrize("batch_size", list(range(1, 32)))
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_no_crash_with_varying_dims(k: int, vocab_size: int, batch_size: int,
                                    device: str):
    """
    Tests that the TypicalAcceptancSampler forward succeeds for
    different combinations of k, vocab_size, batch_size and num devices.
    """
    torch.set_default_device(device)
    typical_acceptance_sampler = get_acceptance_sampler()
    typical_acceptance_sampler.init_gpu_tensors(device=device)
    target_with_bonus_probs = torch.rand(batch_size,
                                         k + 1,
                                         vocab_size,
                                         dtype=torch.float32)
    bonus_token_ids = torch.randint(low=0,
                                    high=vocab_size,
                                    size=(batch_size, 1),
                                    dtype=torch.int64)
    draft_token_ids = torch.randint(low=0,
                                    high=vocab_size,
                                    size=(batch_size, k),
                                    dtype=torch.int64)
    # Verify that sampling succeeds for all cases.
    typical_acceptance_sampler(target_with_bonus_probs,
                               bonus_token_ids,
                               draft_probs=None,
                               draft_token_ids=draft_token_ids)


@pytest.mark.parametrize("above_or_below_vocab_range", ["above", "below"])
@pytest.mark.parametrize("which_token_ids",
                         ["bonus_token_ids", "draft_token_ids"])
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_raises_when_vocab_oob(above_or_below_vocab_range: str,
                               which_token_ids: str, device: str):
    """
    Tests that we throw an exception of the token ids fall outside
    the bound of the provided vocabulary.
    """
    k = 3
    batch_size = 5
    vocab_size = 30_000
    torch.set_default_device(device)
    typical_acceptance_sampler = get_acceptance_sampler(strict_mode=True)
    typical_acceptance_sampler.init_gpu_tensors(device=device)
    target_with_bonus_probs = torch.rand(batch_size,
                                         k + 1,
                                         vocab_size,
                                         dtype=torch.float32)
    bonus_token_ids = torch.randint(low=0,
                                    high=vocab_size,
                                    size=(batch_size, 1),
                                    dtype=torch.int64)
    draft_token_ids = torch.randint(low=0,
                                    high=vocab_size,
                                    size=(batch_size, k),
                                    dtype=torch.int64)
    # Verify that appropriate exceptions are thrown for out
    # of bound vocabs.
    oob_token_ids = None
    if which_token_ids == "bonus_token_ids":
        oob_token_ids = bonus_token_ids
    elif which_token_ids == "draft_token_ids":
        oob_token_ids = draft_token_ids
    else:
        raise AssertionError()

    if above_or_below_vocab_range == "above":
        rogue_token_id = vocab_size + 1
    elif above_or_below_vocab_range == "below":
        rogue_token_id = -1
    else:
        raise AssertionError()

    oob_token_ids[0][0] = rogue_token_id

    with pytest.raises(AssertionError):
        typical_acceptance_sampler(target_with_bonus_probs,
                                   bonus_token_ids,
                                   draft_probs=None,
                                   draft_token_ids=draft_token_ids)


@pytest.mark.parametrize("seed", list(range(10)))
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_uniform_target_distribution_accepts_all_tokens(
        seed: int, device: str):
    """
     Test the TypicalAcceptanceSampler with a uniform target probability 
     distribution.
    
    This test verifies that when provided with a uniform target probability
    distribution, the TypicalAcceptanceSampler accepts all draft tokens. The
    entropy of the uniform target distribution being high should lead to all
    draft tokens being accepted.
    """
    set_random_seed(seed)
    k = 3
    batch_size = 5
    vocab_size = 30_000
    torch.set_default_device(device)
    typical_acceptance_sampler = get_acceptance_sampler(strict_mode=True)
    typical_acceptance_sampler.init_gpu_tensors(device=device)
    target_with_bonus_probs = torch.rand(batch_size,
                                         k + 1,
                                         vocab_size,
                                         dtype=torch.float32)
    draft_token_ids = torch.randint(low=0,
                                    high=vocab_size,
                                    size=(batch_size, k),
                                    dtype=torch.int64)
    bonus_token_ids = torch.randint(low=0,
                                    high=vocab_size,
                                    size=(batch_size, 1),
                                    dtype=torch.int64)
    output_token_ids = typical_acceptance_sampler(
        target_with_bonus_probs,
        bonus_token_ids,
        draft_probs=None,
        draft_token_ids=draft_token_ids)
    # We are using a uniform target probability distribution.
    # For a uniform distribution the entropy is very high and it
    # should lead to all draft tokens being accepted. Verify that.
    assert output_token_ids.shape[0] == batch_size
    assert output_token_ids.shape[1] == (k + 1)
    assert torch.all(output_token_ids[:, -1] == bonus_token_ids.squeeze())

    assert torch.all(output_token_ids[:, :k] == draft_token_ids)


@pytest.mark.parametrize("seed", list(range(10)))
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_temperature_zero_target_distribution(seed: int, device: str):
    """
    Test the TypicalAcceptanceSampler with a zero-temperature target
    probability distribution.

    This test verifies that when using a zero-temperature target probability
    distribution, where only one token has a probability of 1.0, the
    TypicalAcceptanceSampler correctly rejects all draft tokens that do not
    match this probability. Additionally, it ensures that when all draft
    tokens are rejected, the sampler falls back to greedy sampling to select a
    single token from the target distribution.
    """
    set_random_seed(seed)
    k = 3
    batch_size = 5
    vocab_size = 30_000
    torch.set_default_device(device)

    typical_acceptance_sampler = get_acceptance_sampler(strict_mode=True)
    typical_acceptance_sampler.init_gpu_tensors(device=device)
    # Simulate temperature 0 probability distribution for target probabilities
    # and create target probabilities such that only 1 token id has
    # probability 1.0
    target_with_bonus_probs, zero_temperature_token_ids = \
        get_zero_temperature_prob_dist(batch_size, k + 1, vocab_size)
    zero_temperature_token_ids = zero_temperature_token_ids[:, :-1]
    # Populate draft_token_ids such that they exclude the token_ids
    # with probability = 1.0
    draft_token_ids = get_draft_token_ids(batch_size, k, vocab_size,
                                          zero_temperature_token_ids)
    bonus_token_ids = torch.randint(low=0,
                                    high=vocab_size,
                                    size=(batch_size, 1),
                                    dtype=torch.int64)
    # The target probaility distribution is a temperature zero distribution
    # with zero entroy. Since our draft token ids don't match the probability
    # 1.0 tokens in the target distribution we will reject all of them and
    # fallback to the greedy sampling for selecting 1 token for each sequence.
    # Verify the same.
    output_token_ids = typical_acceptance_sampler(
        target_with_bonus_probs,
        bonus_token_ids,
        draft_probs=None,
        draft_token_ids=draft_token_ids)
    assert output_token_ids.shape[0] == batch_size
    assert output_token_ids.shape[1] == (k + 1)
    assert torch.all(output_token_ids[:, -1] == -1)
    assert torch.all(output_token_ids[:, 0] == zero_temperature_token_ids[:,
                                                                          0])


@pytest.mark.parametrize("seed", list(range(10)))
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_mixed_target_distribution(seed: int, device: str):
    """
    Test the TypicalAcceptanceSampler with a mixed target probability
    distribution.

    This test ensures that the TypicalAcceptanceSampler handles a mixed
    target probability distribution correctly. Specifically, it uses a 
    zero-temperature distribution for some sequences and a uniform
    distribution for others. The test verifies that:
    
    - For sequences with a zero-temperature distribution, only the token
    with a probability of 1.0 is accepted, and all other tokens are rejected.
    - For sequences with a uniform distribution, all draft tokens are
    accepted.
    """
    set_random_seed(seed)
    k = 3
    batch_size = 4
    vocab_size = 30_000
    torch.set_default_device(device)
    typical_acceptance_sampler = get_acceptance_sampler(strict_mode=True)
    typical_acceptance_sampler.init_gpu_tensors(device=device)
    # For sequences 0 and 2 set the distribution to a temperature
    # zero distribution. For sequences 1 and 3 set it to a uniform
    # distribution.
    target_with_bonus_probs, zero_temperature_token_ids = \
        get_zero_temperature_prob_dist(batch_size, k + 1, vocab_size)
    zero_temperature_token_ids = zero_temperature_token_ids[:, :-1]
    target_probs = target_with_bonus_probs[:, :-1]
    draft_token_ids = get_draft_token_ids(batch_size, k, vocab_size,
                                          zero_temperature_token_ids)
    uniform_probs = torch.rand(2, k, vocab_size, dtype=torch.float32)
    target_probs[[1, 3]] = uniform_probs
    bonus_token_ids = torch.randint(low=0,
                                    high=vocab_size,
                                    size=(batch_size, 1),
                                    dtype=torch.int64)
    output_token_ids = typical_acceptance_sampler(
        target_with_bonus_probs,
        bonus_token_ids,
        draft_probs=None,
        draft_token_ids=draft_token_ids)
    # verify the shape of output_token_ids
    assert output_token_ids.shape[0] == batch_size
    assert output_token_ids.shape[1] == (k + 1)
    # For sequences 0 and 2 verify that only 1 token is accepted
    # which is the token with probability 1.0 in the target distribution
    # at position 0.
    assert torch.all(output_token_ids[[0, 2], 1:] == -1)
    assert (torch.all(output_token_ids[[0, 2],
                                       0] == zero_temperature_token_ids[[0, 2],
                                                                        0]))
    # For sequences 1 and 3 verify that all tokens are accepted since the
    # target probability distribution is uniform. In addition verify that
    # we also accept the bonus tokens.
    assert torch.all(
        output_token_ids[[1, 3], :-1] == draft_token_ids[[1, 3], :])
    assert torch.all(output_token_ids[[1, 3], -1] != -1)


@pytest.mark.parametrize("seed", list(range(10)))
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_accept_tokens_partially(seed: int, device: str):
    """
    Test the TypicalAcceptanceSampler's behavior when only a subset of draft
    tokens should be accepted.

    This test verifies that the TypicalAcceptanceSampler correctly accepts or
    rejects draft tokens based on a zero-temperature target probability
    distribution. Specifically, it ensures that:
    
    - When all draft tokens match tokens with a probability of 1.0 in the
    target distribution, all draft tokens are accepted.
    - When only some draft tokens match tokens with a probability of 1.0 in
    the target distribution, only those matching tokens are accepted, and the
    rest are rejected.
    """
    set_random_seed(seed)
    k = 5
    batch_size = 1
    vocab_size = 30_000
    torch.set_default_device(device)
    typical_acceptance_sampler = get_acceptance_sampler(strict_mode=True)
    typical_acceptance_sampler.init_gpu_tensors(device=device)
    # Create a temperature zero target probability distribution and ensure
    # all draft token ids correspond to the tokens with 1.0 probability.
    # Verify that all of them are accepted.
    target_with_bonus_probs, zero_temperature_token_ids = \
        get_zero_temperature_prob_dist(batch_size, k + 1, vocab_size)
    zero_temperature_token_ids = zero_temperature_token_ids[:, :-1]
    draft_token_ids = zero_temperature_token_ids
    bonus_token_ids = torch.randint(low=0,
                                    high=vocab_size,
                                    size=(batch_size, 1),
                                    dtype=torch.int64)
    output_token_ids = typical_acceptance_sampler(
        target_with_bonus_probs,
        bonus_token_ids,
        draft_probs=None,
        draft_token_ids=draft_token_ids)
    assert output_token_ids.shape[0] == batch_size
    assert output_token_ids.shape[1] == (k + 1)
    assert torch.all(output_token_ids[:, 0:-1] == draft_token_ids)
    assert torch.all(output_token_ids[:, -1] == bonus_token_ids)
    # Next only keep the first 2 draft tokens same as the zero temperature
    # tokens. For the remaining 3 choose some other tokens. In the
    # response we will expect the first 2 tokens to be the same as the
    # draft tokens and the recovered token and rest as -1
    draft_token_ids_to_replace = get_draft_token_ids(
        batch_size, k, vocab_size, zero_temperature_token_ids)
    draft_token_ids = torch.cat(
        (draft_token_ids[:, :2], draft_token_ids_to_replace[:, -3:]), dim=1)
    output_token_ids = typical_acceptance_sampler(
        target_with_bonus_probs,
        bonus_token_ids,
        draft_probs=None,
        draft_token_ids=draft_token_ids)
    assert output_token_ids.shape[0] == batch_size
    assert output_token_ids.shape[1] == (k + 1)
    assert torch.all(output_token_ids[:, :2] == draft_token_ids[:, :2])
    assert torch.all(
        output_token_ids[:, 2] == target_with_bonus_probs.argmax(-1)[:, 2])
    assert torch.all(output_token_ids[:, -3:] == -1)


@pytest.mark.parametrize("seed", list(range(1)))
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_accept_tokens_set_non_default_posteriors(seed: int, device: str):
    """
    Test the TypicalAcceptanceSampler with custom posterior thresholds and 
    alpha values. This test verifies that by modifying the posterior
    thresholds and alpha values we can change the acceptance behavior of the
    sampler. 
    """
    set_random_seed(seed)
    k = 5
    batch_size = 1
    vocab_size = 30_000
    torch.set_default_device(device)
    typical_acceptance_sampler = get_acceptance_sampler(strict_mode=True)
    typical_acceptance_sampler.init_gpu_tensors(device=device)
    # Simulate temperature 0 probability distribution for target
    # probabilities and create target probabilities such that only 1 token
    # id has probability 1.0 and others have a very low probability of
    # 0.00001. Populate draft_token_ids such that they exclude the token_ids
    # with probability = 1.0. Without any changes to the posterior thresholds
    # none of the draft tokens are accepted.
    target_probs, zero_temperature_token_ids = get_zero_temperature_prob_dist(
        batch_size, k + 1, vocab_size)
    zero_temperature_token_ids = zero_temperature_token_ids[:, :-1]
    target_probs[target_probs == 0] = 0.00001
    draft_token_ids = get_draft_token_ids(batch_size, k, vocab_size,
                                          zero_temperature_token_ids)
    bonus_token_ids = torch.randint(low=0,
                                    high=vocab_size,
                                    size=(batch_size, 1),
                                    dtype=torch.int64)
    output_token_ids = typical_acceptance_sampler(
        target_probs,
        bonus_token_ids,
        draft_probs=None,
        draft_token_ids=draft_token_ids)
    assert output_token_ids.shape[0] == batch_size
    assert output_token_ids.shape[1] == (k + 1)
    assert torch.all(output_token_ids[:, 1:-1] == -1)

    # Change the posterior threshold values to 0.0 so that we will
    # now accept even draft tokens with very low probability in the
    # target distribution. Simulate and verify the same.
    typical_acceptance_sampler = TypicalAcceptanceSampler(
        strict_mode=True, posterior_threshold=0.0, posterior_alpha=0.0)
    typical_acceptance_sampler.init_gpu_tensors(device=device)
    output_token_ids = typical_acceptance_sampler(
        target_probs,
        bonus_token_ids,
        draft_probs=None,
        draft_token_ids=draft_token_ids)
    assert output_token_ids.shape[0] == batch_size
    assert output_token_ids.shape[1] == (k + 1)
    assert torch.all(output_token_ids[:, 0:-1] == draft_token_ids)
    assert torch.all(output_token_ids[:, -1] == bonus_token_ids)


@pytest.mark.parametrize("seed", list(range(10)))
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_get_recovered_token_ids(seed: int, device: str):
    """
    Test the TypicalAcceptanceSampler's method for generating
    replacement token IDs.

    This test verifies that the `_get_recovered_token_ids` method of the 
    TypicalAcceptanceSampler correctly identifies the token IDs to be used
    as recovered token IDs based on the target probability distribution.
    Specifically, it ensures that the method correctly identifies the
    tokens with the highest probability for each sequence in the batch.
    """
    set_random_seed(seed)
    k = 10
    batch_size = 5
    vocab_size = 30_000
    torch.set_default_device(device)
    typical_acceptance_sampler = get_acceptance_sampler(strict_mode=True)
    typical_acceptance_sampler.init_gpu_tensors(device=device)
    target_probs = torch.rand(batch_size, k, vocab_size, dtype=torch.float32)
    expected_replacement_tokens = torch.argmax(target_probs, dim=-1)
    actual_replacement_tokens = (
        typical_acceptance_sampler._get_recovered_token_ids(target_probs))
    assert torch.all(expected_replacement_tokens == actual_replacement_tokens)
