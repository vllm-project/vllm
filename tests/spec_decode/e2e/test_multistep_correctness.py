"""The tests in this file verify end-to-end speculative decoding correctness.

This docstring details important information on the testing methodology.

Most of the tests rely on "greedy equality", where we expect the output of
speculative decoding on a sequence to exactly match the output of normal non-
speculative decoding.

Since speculative decoding with rejection sampling guarantees that the output
distribution matches the target model's output distribution (up to hardware
numerics, see https://arxiv.org/pdf/2302.01318.pdf), we can expect greedy
equality. This gives us good coverage of temp=0.

For temp>0, we rely on unit tests on the rejection sampler to verify that the
output distribution is the same with spec decode vs. no spec decode (this would
be prohibitively expensive to run with a real model).

NOTE: Speculative decoding's distribution equality requires that the measured
distributions of the target model and proposal model be deterministic given the
same input. vLLM largely guarantees this.

@cadedaniel has seen cases where the output probabilities of a draft/target
model change slightly with certain batch sizes or prompts, even with Torch
determinism flags set. It is unclear if this is a bug in vLLM, due to non-
determinism in on-device batched operations, a bug in vLLM's spec decode
implementation, or the "hardware numerics" limitations. Either way, rejection
sampling ensures the output distribution matches the target model, but it breaks
greedy-equality tests for those batch sizes/prompts.
"""

from itertools import cycle

import pytest
from transformers import AutoTokenizer

from vllm import SamplingParams

from .conftest import (get_output_from_llm_generator,
                       run_greedy_equality_correctness_test)


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Required for spec decode.
        "use_v2_block_manager": True
    }])
@pytest.mark.parametrize(
    "per_test_common_llm_kwargs",
    [
        # Try two different tiny base models.
        # Note that one is equal to the draft model, another isn't.
        {
            "model": "JackFram/llama-68m",
        },
        {
            "model": "JackFram/llama-160m",
        },
    ])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("test_llm_kwargs", [
    {
        "speculative_model": "JackFram/llama-68m",
        "num_speculative_tokens": 5,
    },
])
@pytest.mark.parametrize("max_output_len", [
    256,
])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("seed", [1])
def test_spec_decode_e2e_greedy_correctness_tiny_model_large_bs_diff_output_len(
        baseline_llm_generator, test_llm_generator, batch_size: int,
        max_output_len: int):
    """Verify greedy equality on a tiny model, with a large batch size, and when
    sampling respects the EOS token.
    """
    run_greedy_equality_correctness_test(baseline_llm_generator,
                                         test_llm_generator,
                                         batch_size,
                                         max_output_len,
                                         force_output_len=False)
