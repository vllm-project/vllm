# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import random
from typing import NamedTuple

import numpy as np
import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from vllm.benchmarks.datasets import RandomDataset, SampleRequest


@pytest.fixture(scope="session")
def hf_tokenizer() -> PreTrainedTokenizerBase:
    # Use a small, commonly available tokenizer
    return AutoTokenizer.from_pretrained("gpt2")


class Params(NamedTuple):
    num_requests: int
    prefix_len: int
    range_ratio: float
    input_len: int
    output_len: int


@pytest.fixture(scope="session")
def random_dataset_params() -> Params:
    return Params(num_requests=16,
                  prefix_len=7,
                  range_ratio=0.3,
                  input_len=50,
                  output_len=20)


def _fingerprint_sample(req: SampleRequest) -> tuple[str, int, int]:
    """Project a SampleRequest into a comparable tuple."""
    return (req.prompt, req.prompt_len, req.expected_output_len)


def _collect_samples(dataset: RandomDataset,
                    tokenizer: PreTrainedTokenizerBase,
                     num_requests: int = 16,
                     prefix_len: int = 7,
                     range_ratio: float = 0.3,
                     input_len: int = 50,
                     output_len: int = 20) -> list[tuple[str, int, int]]:
    samples = dataset.sample(
        tokenizer=tokenizer,
        num_requests=num_requests,
        prefix_len=prefix_len,
        range_ratio=range_ratio,
        input_len=input_len,
        output_len=output_len,
    )
    return [_fingerprint_sample(s) for s in samples]


@pytest.mark.benchmark
def test_random_dataset_same_seed(
        hf_tokenizer: PreTrainedTokenizerBase,
        random_dataset_params: Params) -> None:
    """Same seed should yield identical outputs, even if global RNGs change.

    This guards against accidental reliance on Python's random or np.random
    in RandomDataset after moving to numpy.default_rng.
    """
    p = random_dataset_params
    common_seed = 123
    dataset_a = RandomDataset(random_seed=common_seed)
    dataset_b = RandomDataset(random_seed=common_seed)
    a = _collect_samples(dataset_a,
                         hf_tokenizer,
                         num_requests=p.num_requests,
                         prefix_len=p.prefix_len,
                         range_ratio=p.range_ratio,
                         input_len=p.input_len,
                         output_len=p.output_len)

    # Perturb global RNG state to ensure isolation
    random.seed(999)
    np.random.seed(888)

    b = _collect_samples(dataset_b,
                         hf_tokenizer,
                         num_requests=p.num_requests,
                         prefix_len=p.prefix_len,
                         range_ratio=p.range_ratio,
                         input_len=p.input_len,
                         output_len=p.output_len)
    assert a == b

@pytest.mark.benchmark
def test_random_dataset_different_seeds(
        hf_tokenizer: PreTrainedTokenizerBase,
        random_dataset_params: Params) -> None:
    """Different seeds should change outputs with overwhelming likelihood."""
    p = random_dataset_params
    seed_a = 0
    seed_b = 1
    dataset_a = RandomDataset(random_seed=seed_a)
    dataset_b = RandomDataset(random_seed=seed_b)
    a = _collect_samples(dataset_a,
                         hf_tokenizer,
                         num_requests=p.num_requests,
                         prefix_len=p.prefix_len,
                         range_ratio=p.range_ratio,
                         input_len=p.input_len,
                         output_len=p.output_len)

    # Perturb global RNG with same seed as dataset_a to ensure isolation
    random.seed(seed_a)
    np.random.seed(seed_a)
    b = _collect_samples(dataset_b,
                         hf_tokenizer,
                         num_requests=p.num_requests,
                         prefix_len=p.prefix_len,
                         range_ratio=p.range_ratio,
                         input_len=p.input_len,
                         output_len=p.output_len)
    assert a != b


