import pytest
import torch

from tests.conftest import VllmRunner
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
    num_top_logprobs = 6
    hf_model = hf_runner(model, dtype=dtype)
    hf_logprobs = hf_model.generate_greedy_logprobs(
        example_prompts,
        max_tokens=max_tokens,
    )
    del hf_model

    vllm_model = vllm_runner(model, dtype=dtype, max_logprobs=num_top_logprobs)
    vllm_sampling_params = SamplingParams(max_tokens=max_tokens,
                                          logprobs=num_top_logprobs,
                                          prompt_logprobs=5,
                                          temperature=0.0)
    vllm_results = vllm_model.model.generate(
        example_prompts, sampling_params=vllm_sampling_params)

    # Test whether logprobs are included in the results.
    for result in vllm_results:
        assert result.prompt_logprobs is not None
        assert result.outputs[0].logprobs is not None
        assert len(result.outputs[0].logprobs) == max_tokens
        for logprobs in result.outputs[0].logprobs:
            assert len(logprobs) == num_top_logprobs
        output_text = result.outputs[0].text
        output_string_from_most_likely_tokens = []
        for top_logprobs in result.outputs[0].logprobs:
            top_logprob = next(iter(top_logprobs.values()))
            output_string_from_most_likely_tokens.append(
                top_logprob.decoded_token)
        output_string_from_most_likely_tokens = "".join(
            output_string_from_most_likely_tokens)
        assert output_text == output_string_from_most_likely_tokens, (
            "The output text from the top logprob for each token position "
            "should be the same as the output text in the result.")

    # Test whether prompt logprobs are consistent with HF
    for vllm_result, hf_logprob in zip(vllm_results, hf_logprobs):
        # Check prompt logprobs
        vllm_prompt_logprobs = vllm_result.prompt_logprobs[1:]
        for i, vllm_prompt_logprob_dict in enumerate(vllm_prompt_logprobs):
            for token_id, logprob in vllm_prompt_logprob_dict.items():
                torch.testing.assert_close(logprob.logprob,
                                           hf_logprob[0][i][token_id].item(),
                                           atol=1e-2,
                                           rtol=1e-2)
        vllm_sample_logprobs = vllm_result.outputs[0].logprobs
        for i, top_logprobs in enumerate(vllm_sample_logprobs):
            for token_id, sample_logprob in top_logprobs.items():
                logprob = sample_logprob.logprob
                torch.testing.assert_close(logprob,
                                           hf_logprob[i][-1][token_id].item(),
                                           atol=1e-2,
                                           rtol=1e-2)
                assert isinstance(sample_logprob.decoded_token, str), (
                    "The token should be decoded by the time it is returned "
                    " to the user.")


def test_max_logprobs():
    runner = VllmRunner("facebook/opt-125m", max_logprobs=1)
    vllm_sampling_params = SamplingParams(logprobs=1)
    # should pass
    runner.generate(["Hello world"], sampling_params=vllm_sampling_params)

    bad_sampling_params = SamplingParams(logprobs=2)
    with pytest.raises(ValueError):
        runner.generate(["Hello world"], sampling_params=bad_sampling_params)
