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


def _collect_samples(tokenizer: PreTrainedTokenizerBase,
                     seed: int,
                     num_requests: int = 16,
                     prefix_len: int = 7,
                     range_ratio: float = 0.3,
                     input_len: int = 50,
                     output_len: int = 20) -> list[tuple[str, int, int]]:
    dataset = RandomDataset(random_seed=seed)
    samples = dataset.sample(
        tokenizer=tokenizer,
        num_requests=num_requests,
        prefix_len=prefix_len,
        range_ratio=range_ratio,
        input_len=input_len,
        output_len=output_len,
    )
    return [_fingerprint_sample(s) for s in samples]


def _diff_message(a: list[tuple[str, int, int]],
                  b: list[tuple[str, int, int]],
                  **context: object) -> str:
    parts = [f"{k}={v}" for k, v in context.items()]
    msg = " | ".join(parts)
    out = [f"Context: {msg}"]
    if len(a) != len(b):
        out.append(f"Length differ: {len(a)} vs {len(b)}")
    min_len = min(len(a), len(b))
    mismatch_index = None
    for i in range(min_len):
        if a[i] != b[i]:
            mismatch_index = i
            break
    if mismatch_index is None and len(a) != len(b):
        mismatch_index = min_len
    if mismatch_index is not None:
        a_item = a[mismatch_index] if mismatch_index < len(a) else None
        b_item = b[mismatch_index] if mismatch_index < len(b) else None
        out.append(f"First mismatch at index {mismatch_index}:")
        out.append(f"A: {a_item}")
        out.append(f"B: {b_item}")
    return "\n".join(out)

@pytest.mark.benchmark
def test_random_dataset_deterministic_with_fixed_seed_and_global_rng_perturb(
        hf_tokenizer: PreTrainedTokenizerBase,
        random_dataset_params: Params) -> None:
    """Same seed should yield identical outputs, even if global RNGs change.

    This guards against accidental reliance on Python's random or np.random
    in RandomDataset after moving to numpy.default_rng.
    """
    p = random_dataset_params
    a = _collect_samples(hf_tokenizer,
                         seed=123,
                         num_requests=p.num_requests,
                         prefix_len=p.prefix_len,
                         range_ratio=p.range_ratio,
                         input_len=p.input_len,
                         output_len=p.output_len)

    # Perturb global RNG state to ensure isolation
    random.seed(999)
    np.random.seed(888)

    b = _collect_samples(hf_tokenizer,
                         seed=123,
                         num_requests=p.num_requests,
                         prefix_len=p.prefix_len,
                         range_ratio=p.range_ratio,
                         input_len=p.input_len,
                         output_len=p.output_len)
    assert a == b, _diff_message(
        a,
        b,
        test="deterministic",
        seed=123,
        num_requests=p.num_requests,
        prefix_len=p.prefix_len,
        range_ratio=p.range_ratio,
        input_len=p.input_len,
        output_len=p.output_len,
    )

@pytest.mark.benchmark
def test_random_dataset_changes_with_different_seed(
        hf_tokenizer: PreTrainedTokenizerBase,
        random_dataset_params: Params) -> None:
    """Different seeds should change outputs with overwhelming likelihood."""
    p = random_dataset_params
    a = _collect_samples(hf_tokenizer,
                         seed=0,
                         num_requests=p.num_requests,
                         prefix_len=p.prefix_len,
                         range_ratio=p.range_ratio,
                         input_len=p.input_len,
                         output_len=p.output_len)
    b = _collect_samples(hf_tokenizer,
                         seed=1,
                         num_requests=p.num_requests,
                         prefix_len=p.prefix_len,
                         range_ratio=p.range_ratio,
                         input_len=p.input_len,
                         output_len=p.output_len)
    if a == b:
        pytest.fail(
            _diff_message(
                a,
                b,
                test="different_seeds_should_differ",
                seed_a=0,
                seed_b=1,
                num_requests=p.num_requests,
                prefix_len=p.prefix_len,
                range_ratio=p.range_ratio,
                input_len=p.input_len,
                output_len=p.output_len,
            )
        )


