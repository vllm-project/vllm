from itertools import cycle
from typing import List, Tuple

import pytest
from transformers import AutoTokenizer

from vllm import SamplingParams


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        # Use a small model for a fast test.
        # Note this is repeated in the test body; to initialize a tokenizer.
        "model": "JackFram/llama-68m",

        # Skip real loading for fast test.
        "load_format": "dummy",

        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Required for spec decode.
        "use_v2_block_manager": True
    }])
@pytest.mark.parametrize(
    "per_test_common_llm_kwargs",
    [
        {
            "speculative_model": "JackFram/llama-68m",
            "num_speculative_tokens": 5,
        },
        {
            "speculative_model": "JackFram/llama-68m",
            "num_speculative_tokens": 1,
        },
        {
            # No spec decode.
        },
    ])
@pytest.mark.parametrize("test_llm_kwargs", [{}])
@pytest.mark.parametrize("batch_size", [1])
# NOTE: We should run more permutations of this test (more BS, more seeds). But
# because our spec decode generates gibberish token ids, the likelihood of
# emitting an invalid token combination is nontrivial. This causes divergence in
# behavior of vLLM detokenization vs. hf tokenizer, for example when two "utf-
# start" bytes are emitted.
@pytest.mark.parametrize("seed", [1])
def test_spec_decode_e2e_logical_flow(test_llm_generator, batch_size: int):
    """Run generation with speculative decoding on a batch. Verify the engine
    generates the correct number of tokens (via ignore_eos=True), and that the
    detokenization matches HF transformers.
    """
    output_len = 32
    temperature = 0.0

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    prompts = [prompt for prompt, _ in zip(cycle(prompts), range(batch_size))]

    sampling_params = SamplingParams(
        max_tokens=output_len,
        ignore_eos=True,
        temperature=temperature,
        skip_special_tokens=True,
        spaces_between_special_tokens=False,
    )

    batch_tokens, batch_token_ids = get_output_from_llm_generator(
        test_llm_generator, prompts, sampling_params)

    # Expect a generation for each prompt in the batch.
    assert len(batch_token_ids) == len(prompts)

    # Expect each generation to have expected number of tokens (note
    # ignore_eos=True).
    assert all(len(token_ids) == output_len for token_ids in batch_token_ids)

    # Expect detokenized string to match.
    tok = AutoTokenizer.from_pretrained("JackFram/llama-68m")
    for actual_tokens, actual_token_ids in zip(batch_tokens, batch_token_ids):
        expected_tokens = tok.decode(actual_token_ids)
        print(f"{actual_token_ids=}")
        assert actual_tokens.strip() == expected_tokens.strip()


@pytest.mark.parametrize(
    "common_llm_kwargs",
    [{
        # Use a small model for a fast test.
        "model": "JackFram/llama-68m",
        "speculative_model": "JackFram/llama-68m",
        "num_speculative_tokens": 5,

        # Skip real loading for fast test.
        "load_format": "dummy",

        # Skip cuda graph recording for fast test.
        "enforce_eager": True,

        # Required for spec decode.
        "use_v2_block_manager": True
    }])
@pytest.mark.parametrize(
    "per_test_common_llm_kwargs",
    [
        {
            # Expect failure as spec decode not supported by
            # Ray backend.
            "worker_use_ray": True,
        },
    ])
@pytest.mark.parametrize("test_llm_kwargs", [{}])
@pytest.mark.parametrize("seed", [1])
def test_spec_decode_xfail(test_llm_generator):
    """Verify that speculative decoding with Ray fails.
    """
    output_len = 128
    temperature = 0.0

    prompts = [
        "Hello, my name is",
    ]

    sampling_params = SamplingParams(
        max_tokens=output_len,
        ignore_eos=True,
        temperature=temperature,
    )

    with pytest.raises(AssertionError,
                       match="Speculative decoding not yet supported for "):
        get_output_from_llm_generator(test_llm_generator, prompts,
                                      sampling_params)


def get_output_from_llm_generator(
        llm_generator, prompts,
        sampling_params) -> Tuple[List[str], List[List[int]]]:
    for llm in llm_generator:
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        token_ids = [output.outputs[0].token_ids for output in outputs]
        tokens = [output.outputs[0].text for output in outputs]
        del llm

    return tokens, token_ids
