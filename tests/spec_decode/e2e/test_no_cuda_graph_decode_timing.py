from itertools import cycle

import pytest
from transformers import AutoTokenizer

from vllm import SamplingParams

from .conftest import (get_output_from_llm_generator,
                       run_greedy_equality_correctness_test)
@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        # Use a small model for a fast test.
        # Note this is repeated in the test body; to initialize a tokenizer.
        "model": "JackFram/llama-68m",

        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Required for spec decode.
        "use_v2_block_manager": True,

        # Use AsyncLLM engine
        "use_async": False,

        "tensor_parallel_size": 1,
    }])
@pytest.mark.parametrize("baseline_llm_kwargs", [{}])
@pytest.mark.parametrize("per_test_common_llm_kwargs", [{}])
@pytest.mark.parametrize("batch_size", [5])
@pytest.mark.parametrize("seed", [1])
def test_spec_decode_e2e_with_async_engine(baseline_llm_generator,
                                           batch_size: int):
    
    temperature = 0.0

    prompts = [
        "The current president of the United States of America is ",
        "The capital of France is",
        "The future of AI is",
        "San Francisco is know for its",
        "Facebook was created in 2004 by",
    ]

    output_len = 64
    temperature = 0.0

    prompts = [prompt for prompt, _ in zip(cycle(prompts), range(batch_size))]

    sampling_params = SamplingParams(
        max_tokens=output_len,
        ignore_eos=True,
        temperature=temperature,
    )

    batch_tokens, batch_token_ids = get_output_from_llm_generator(
        baseline_llm_generator, prompts, sampling_params)

