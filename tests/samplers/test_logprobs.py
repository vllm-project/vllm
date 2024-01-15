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
    hf_logprobs = hf_model.generate_greedy_logprobs(
        example_prompts,
        max_tokens=max_tokens,
    )
    del hf_model

    vllm_model = vllm_runner(model, dtype=dtype)
    vllm_sampling_params = SamplingParams(max_tokens=max_tokens,
                                          logprobs=5,
                                          prompt_logprobs=5,
                                          temperature=0.0)
    vllm_results = vllm_model.model.generate(
        example_prompts, sampling_params=vllm_sampling_params)
    del vllm_model

    # Test whether logprobs are included in the results.
    for result in vllm_results:
        assert result.prompt_logprobs is not None
        assert result.outputs[0].logprobs is not None

    # Test whether prompt logprobs are consistent with HF
    for vllm_result, hf_logprob in zip(vllm_results, hf_logprobs):
        # Check prompt logprobs
        vllm_prompt_logprobs = vllm_result.prompt_logprobs[1:]
        for i, vllm_prompt_logprob_dict in enumerate(vllm_prompt_logprobs):
            for token_id, logprob in vllm_prompt_logprob_dict.items():
                torch.testing.assert_close(logprob,
                                           hf_logprob[0][i][token_id].item(),
                                           atol=1e-2,
                                           rtol=1e-2)
        vllm_sample_logprobs = vllm_result.outputs[0].logprobs
        for i, vllm_sample_logprob_dict in enumerate(vllm_sample_logprobs):
            for token_id, logprob in vllm_sample_logprob_dict.items():
                torch.testing.assert_close(logprob,
                                           hf_logprob[i][-1][token_id].item(),
                                           atol=1e-2,
                                           rtol=1e-2)
