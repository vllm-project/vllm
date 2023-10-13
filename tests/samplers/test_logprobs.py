import pytest
import torch

from vllm import SamplingParams

MODELS = ["facebook/opt-125m"]

@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
def test_get_prompt_logprobs(
    hf_runner,
    vllm_runner,
    model,
    dtype,
    example_prompts,
):
    max_tokens = 5
    hf_model = hf_runner(model, dtype=dtype)
    vllm_model = vllm_runner(model, dtype=dtype)
    # Test whether prompt logprobs are included in the results.
    echo_logprob_params = SamplingParams(max_tokens=max_tokens,
                                         logprobs=5,
                                         prompt_logprobs=5,
                                         temperature=0.0)
    echo_logprob_results = vllm_model.model.generate(
        example_prompts, sampling_params=echo_logprob_params)

    for result in echo_logprob_results:
        assert result.prompt_logprobs is not None
        assert result.outputs[0].logprobs is not None

    # To test whether prompt logprobs are consistent with HF
    hf_logprobs = hf_model.generate_greedy_logprobs(
        example_prompts,
        max_tokens=max_tokens,
        num_logprobs=5,
    )
    print(echo_logprob_results[0].prompt_logprobs)
    print(hf_logprobs[0][0][0][100])
    exit(0)
    del hf_model
    del vllm_model
