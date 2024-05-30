"""Tests for rejection sampling."""
from typing import List, Tuple

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.typical_acceptance_sampler import (
    TypicalAcceptanceSampler)
from vllm.model_executor.utils import set_random_seed

CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1)
]

def get_zero_temperature_prob_dist(batch_size, k, vocab_size):
    # Simulate temperature 0 probability distribution for target probabilities
    # and create target probabilities such that only 1 token id has
    # probability 1.0
    target_probs = torch.rand(batch_size, k, vocab_size, dtype=torch.float32)
    probs = torch.rand(batch_size, k, vocab_size)
    _, max_indices = torch.max(probs, dim=-1)
    # set the probability of the tokens with ids in max_indices to 1 and
    # the rest to 0.
    target_probs = torch.zeros_like(probs).scatter_(
        -1, max_indices.unsqueeze(-1), 1.0)
    return target_probs, max_indices

def get_draft_token_ids(
    batch_size: int, k: int, vocab_size: int, token_ids_to_exclude: torch.Tensor):
    # Populate draft_token_ids such that they exclude the token_ids
    # with probability = 1.0
    draft_token_ids = torch.empty(batch_size, k, dtype=torch.long)
    for i in range(batch_size):
        for j in range(k):
            # Generate a random token ID excluding max_indices[i, j]
            while True:
                token_id = torch.randint(0, vocab_size, (1,)).item()
                if token_id != token_ids_to_exclude[i, j]:
                    draft_token_ids[i, j] = token_id
                    break
    return draft_token_ids

@pytest.mark.parametrize("k", list(range(1, 6)))
@pytest.mark.parametrize("vocab_size", [30_000, 50_000])
@pytest.mark.parametrize("batch_size", list(range(1, 32)))
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_no_crash_with_varying_dims(k: int, vocab_size: int, batch_size: int,
                                    device: str):
    torch.set_default_device(device)
    typical_acceptance_sampler = TypicalAcceptanceSampler()
    typical_acceptance_sampler.init_gpu_tensors(rank=0)
    target_probs = torch.rand(batch_size, k, vocab_size, dtype=torch.float32)
    bonus_token_ids = torch.randint(low=0,
                                    high=vocab_size,
                                    size=(batch_size, 1),
                                    dtype=torch.int64)
    draft_token_ids = torch.randint(low=0,
                                    high=vocab_size,
                                    size=(batch_size, k),
                                    dtype=torch.int64)

    typical_acceptance_sampler(target_probs, bonus_token_ids, draft_token_ids)


@pytest.mark.parametrize("above_or_below_vocab_range", ["above", "below"])
@pytest.mark.parametrize("which_token_ids",
                         ["bonus_token_ids", "draft_token_ids"])
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_raises_when_vocab_oob(above_or_below_vocab_range: str,
                               which_token_ids: str, device: str):
    k = 3
    batch_size = 5
    vocab_size = 30_000
    torch.set_default_device(device)
    typical_acceptance_sampler = TypicalAcceptanceSampler(strict_mode=True)
    typical_acceptance_sampler.init_gpu_tensors(rank=0)
    target_probs = torch.rand(batch_size, k, vocab_size, dtype=torch.float32)
    bonus_token_ids = torch.randint(low=0,
                                    high=vocab_size,
                                    size=(batch_size, 1),
                                    dtype=torch.int64)
    draft_token_ids = torch.randint(low=0,
                                    high=vocab_size,
                                    size=(batch_size, k),
                                    dtype=torch.int64)
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
        typical_acceptance_sampler(target_probs, bonus_token_ids, draft_token_ids)


@pytest.mark.parametrize("seed", list(range(10)))
@pytest.mark.parametrize("disable_bonus_tokens", [True, False])
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_uniform_target_distribution_accepts_all_tokens(
    seed: int, disable_bonus_tokens: bool, device: str):
    set_random_seed(seed)
    k = 3
    batch_size = 5
    vocab_size = 30_000
    torch.set_default_device(device)
    typical_acceptance_sampler = TypicalAcceptanceSampler(
        strict_mode=True, disable_bonus_tokens=disable_bonus_tokens)
    typical_acceptance_sampler.init_gpu_tensors(rank=0)
    target_probs = torch.rand(batch_size, k, vocab_size, dtype=torch.float32)
    draft_token_ids = torch.randint(low=0,
                                    high=vocab_size,
                                    size=(batch_size, k),
                                    dtype=torch.int64)
    bonus_token_ids = torch.randint(low=0,
                                    high=vocab_size,
                                    size=(batch_size, 1),
                                    dtype=torch.int64)
    output_token_ids = typical_acceptance_sampler(
        target_probs, bonus_token_ids, draft_token_ids)
    assert output_token_ids.shape[0] == batch_size
    assert output_token_ids.shape[1] == (k + 1)
    if disable_bonus_tokens:
        assert torch.all(output_token_ids[:, -1] == -1)
    else:
        assert torch.all(output_token_ids[:, -1] == bonus_token_ids.squeeze())
    
    assert torch.all(output_token_ids[:, : k] == draft_token_ids)


@pytest.mark.parametrize("seed", list(range(10)))
@pytest.mark.parametrize("disable_bonus_tokens", [True, False])
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_temperature_zero_target_distribution(
    seed: int, disable_bonus_tokens: bool, device: str):
    set_random_seed(seed)
    k = 3
    batch_size = 5
    vocab_size = 30_000
    torch.set_default_device(device)

    typical_acceptance_sampler = TypicalAcceptanceSampler(
        strict_mode=True, disable_bonus_tokens=disable_bonus_tokens)
    typical_acceptance_sampler.init_gpu_tensors(rank=0)
    # Simulate temperature 0 probability distribution for target probabilities
    # and create target probabilities such that only 1 token id has
    # probability 1.0
    target_probs, zero_temperature_token_ids = get_zero_temperature_prob_dist(
        batch_size, k, vocab_size)
    # Populate draft_token_ids such that they exclude the token_ids
    # with probability = 1.0
    draft_token_ids = get_draft_token_ids(
        batch_size, k, vocab_size, zero_temperature_token_ids)
    bonus_token_ids = torch.randint(low=0,
                                    high=vocab_size,
                                    size=(batch_size, 1),
                                    dtype=torch.int64)
    output_token_ids = typical_acceptance_sampler(
        target_probs, bonus_token_ids, draft_token_ids)
    assert output_token_ids.shape[0] == batch_size
    assert output_token_ids.shape[1] == (k + 1)
    assert torch.all(output_token_ids[:, -1] == -1)
    assert torch.all(
        output_token_ids[:, 0] == zero_temperature_token_ids[:, 0])


@pytest.mark.parametrize("seed", list(range(10)))
@pytest.mark.parametrize("disable_bonus_tokens", [True, False])
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_mixed_target_distribution(
    seed: int, disable_bonus_tokens: bool, device: str):
    set_random_seed(seed)
    k = 3
    batch_size = 4
    vocab_size = 30_000
    torch.set_default_device(device)
    typical_acceptance_sampler = TypicalAcceptanceSampler(
        strict_mode=True, disable_bonus_tokens=disable_bonus_tokens)
    typical_acceptance_sampler.init_gpu_tensors(rank=0)
    # For batches 0 and 2 set the distribution to an uniform distribution. For 
    # batches 1 and 3 set it to a temperature 0 distribution.
    target_probs, zero_temperature_token_ids = (
        get_zero_temperature_prob_dist(batch_size, k, vocab_size))
    draft_token_ids = get_draft_token_ids(
        batch_size, k, vocab_size, zero_temperature_token_ids)
    # Create target_probs such that only one token_id has probability 1.0
    uniform_probs = torch.rand(2, k, vocab_size, dtype=torch.float32)
    target_probs[[1, 3]] = uniform_probs
    bonus_token_ids = torch.randint(low=0,
                                    high=vocab_size,
                                    size=(batch_size, 1),
                                    dtype=torch.int64)
    output_token_ids = typical_acceptance_sampler(
        target_probs, bonus_token_ids, draft_token_ids)
    assert output_token_ids.shape[0] == batch_size
    assert output_token_ids.shape[1] == (k + 1)
    assert torch.all(output_token_ids[[0, 2], 1:] == -1)
    assert (
        torch.all(output_token_ids[[0, 2], 0] == zero_temperature_token_ids[[0, 2], 0]))
    assert  torch.all(output_token_ids[[1, 3], :-1] == draft_token_ids[[1, 3], :])
    if disable_bonus_tokens:
        assert torch.all(output_token_ids[[1, 3], -1] == -1)
    else:
        assert torch.all(output_token_ids[[1, 3], -1] != -1)
