import gc

import pytest
import torch
import triton
import triton.language as tl

from vllm.model_executor.layers.ops.sample import (
    MAX_TRITON_N_COLS, _uniform_to_exponential, get_num_triton_sampler_splits,
    sample)
from vllm.model_executor.sampling_metadata import SamplingTensors
from vllm.model_executor.utils import set_random_seed

SINGLE_SPLIT_VOCAB_SIZE = 32000  # llama/mistral/mixtral vocab size
MULTI_SPLIT_VOCAB_SIZE = MAX_TRITON_N_COLS + 100


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.cuda.empty_cache()


@triton.jit
def _uniform_to_exponential_kernel(input, output, n: tl.constexpr):
    idx = tl.arange(0, n)
    x = tl.load(input + idx)
    y = _uniform_to_exponential(x)
    tl.store(output + idx, y)


def test_uniform_to_exponential():
    """Test that we can convert uniform to exponential without div by 0."""
    input = torch.tensor([0.0, 1.0 - torch.finfo(torch.float32).eps],
                         dtype=torch.float32,
                         device="cuda")
    output = torch.zeros(input.shape, dtype=torch.float32, device="cuda")
    _uniform_to_exponential_kernel[(1, )](input, output, 2)
    assert torch.all(torch.isfinite(output))
    assert torch.all(output > 0)
    assert torch.all(torch.isfinite(torch.full_like(output, 1.0) / output))


@pytest.mark.parametrize("random_sampling", [True, False, "mixed"])
@pytest.mark.parametrize("max_best_of", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("modify_greedy_probs", [True, False])
@pytest.mark.parametrize("seed", [1337])
@pytest.mark.parametrize("vocab_size",
                         [SINGLE_SPLIT_VOCAB_SIZE, MULTI_SPLIT_VOCAB_SIZE])
@pytest.mark.parametrize("save_logprobs", [True, False])
def test_sample_decoding_only(random_sampling, max_best_of,
                              modify_greedy_probs, seed, vocab_size,
                              save_logprobs):
    set_random_seed(seed)
    bs = 8
    probs = torch.zeros((bs, vocab_size), dtype=torch.float32, device="cuda")
    for i in range(bs):
        probs[i, i * (vocab_size // bs)] = 1.0
    logprobs = torch.rand_like(probs)
    sample_indices = torch.arange(bs, dtype=torch.long, device="cuda")
    n_splits = get_num_triton_sampler_splits(probs.shape[1])
    if random_sampling == "mixed":
        random_sampling_mask = (torch.rand(
            (1, bs), device="cuda") < 0.5).expand(n_splits, bs)
    elif random_sampling:
        random_sampling_mask = torch.ones((n_splits, bs),
                                          dtype=torch.bool,
                                          device="cuda")
    else:
        random_sampling_mask = torch.zeros((n_splits, bs),
                                           dtype=torch.bool,
                                           device="cuda")

    seeds = torch.randint(1,
                          torch.iinfo(torch.long).max, (n_splits, bs),
                          device="cuda").mul_(random_sampling_mask)
    sampled_tokens, sampled_logprobs, sampled_modified_probs = sample(
        probs=probs,
        logprobs=logprobs,
        sample_indices=sample_indices,
        seeds=seeds,
        max_best_of=max_best_of,
        modify_greedy_probs=modify_greedy_probs,
        save_logprobs=save_logprobs,
        _save_modified_probs=True)
    assert sampled_tokens.shape == (bs, max_best_of)
    for i in range(bs):
        assert torch.all(sampled_tokens[i] == i * (vocab_size // bs))
        request_uses_random_sampling = random_sampling_mask[0, i]
        if modify_greedy_probs and not request_uses_random_sampling:
            # If we are modifying greedy probs and the request is greedy,
            # we want to make sure the probs tensor is modified in place
            assert torch.allclose(
                probs[i][sampled_tokens[i]],
                torch.full_like(probs[i][sampled_tokens[i]], 1.0))
            assert torch.sum(probs[i]) == 1.0
            assert torch.allclose(
                sampled_modified_probs[i][0],
                torch.full_like(sampled_modified_probs[i][0], 1.0))
        elif request_uses_random_sampling:
            # If the request is random, we want to make sure
            # sampled_modified_probs tensor has noise added
            # (and thus is different from probs tensor)
            assert not torch.allclose(sampled_modified_probs[i][0],
                                      probs[i][sampled_tokens[i]])
        elif not request_uses_random_sampling:
            # If the request is greedy and we are not modifying greedy probs,
            # we want to make sure sampled_modified_probs tensor is the same as
            # the probs tensor.
            assert torch.allclose(sampled_modified_probs[i][0],
                                  probs[i][sampled_tokens[i]])

    if save_logprobs:
        assert sampled_logprobs.shape == (bs, max_best_of)
        for i in range(bs):
            for best_of in range(max_best_of):
                assert torch.all(sampled_logprobs[i] == logprobs[i][
                    sampled_tokens[i, best_of]])
    else:
        assert sampled_logprobs is None


@pytest.mark.parametrize("random_sampling", [True, False, "mixed"])
@pytest.mark.parametrize("max_best_of", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("modify_greedy_probs", [True, False])
@pytest.mark.parametrize("seed", [1337])
@pytest.mark.parametrize("vocab_size",
                         [SINGLE_SPLIT_VOCAB_SIZE, MULTI_SPLIT_VOCAB_SIZE])
def test_sample_prompt_logprobs(random_sampling, max_best_of,
                                modify_greedy_probs, seed, vocab_size):
    set_random_seed(seed)
    prompt_sizes = [16, 32, 64, 128] * 2
    samples = 8
    bs = samples + sum(prompt_sizes)
    probs = torch.zeros((bs, vocab_size), dtype=torch.float32, device="cuda")
    for i in range(bs):
        probs[i, i * (vocab_size // bs)] = 1.0
    logprobs = torch.rand_like(probs)
    sample_indices = torch.tensor(prompt_sizes,
                                  dtype=torch.long,
                                  device="cuda").cumsum_(0)
    n_splits = get_num_triton_sampler_splits(probs.shape[1])
    if random_sampling == "mixed":
        random_sampling_mask = torch.rand(
            (n_splits, samples), device="cuda") < 0.5
    elif random_sampling:
        random_sampling_mask = torch.ones((n_splits, samples),
                                          dtype=torch.bool,
                                          device="cuda")
    else:
        random_sampling_mask = torch.zeros((n_splits, samples),
                                           dtype=torch.bool,
                                           device="cuda")

    seeds = torch.randint(1,
                          torch.iinfo(torch.long).max, (n_splits, samples),
                          device="cuda").mul_(random_sampling_mask)
    sampled_tokens, sampled_logprobs, _ = sample(
        probs=probs,
        logprobs=logprobs,
        sample_indices=sample_indices,
        seeds=seeds,
        max_best_of=max_best_of,
        modify_greedy_probs=modify_greedy_probs,
        save_logprobs=True)
    assert sampled_tokens.shape == (samples, max_best_of)
    assert sampled_logprobs.shape == (samples, max_best_of)
    for i, t in enumerate(sample_indices):
        assert torch.all(sampled_tokens[i] == t * (vocab_size // bs))
        for best_of in range(max_best_of):
            assert torch.all(sampled_logprobs[i] == logprobs[sample_indices[i]]
                             [sampled_tokens[i, best_of]])


@pytest.mark.parametrize("seed", list(range(16)))
def test_get_sequence_seeds(seed):
    """Ensure that we get a different child seed from base 
    seed + extra entropy"""
    starting_seed = seed
    seq_seed = None
    extra_entropy = 1
    for i in range(512):
        new_seq_seed = SamplingTensors._get_sequence_seeds(starting_seed,
                                                           i,
                                                           seeds_to_generate=1,
                                                           is_greedy=False)[0]
        new_seq_seed_extra_entropy = SamplingTensors._get_sequence_seeds(
            starting_seed,
            i,
            extra_entropy,
            seeds_to_generate=1,
            is_greedy=False)[0]
        assert new_seq_seed_extra_entropy != new_seq_seed
        assert seq_seed != new_seq_seed
        seq_seed = new_seq_seed
