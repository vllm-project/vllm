import pytest
import torch

from vllm import SamplingParams

MODELS = ["facebook/opt-125m"]

TEST_PROMPTS = [
    "Hello world",
    "Hello world. This is a test.",
    "Hello world. This is a test. This is a test.",
    "To be or not to be,",
    "This is a question.",
    "Baltimore is the greatest city in",
]
# Test IDs that have the same prefix.
SAME_PREFIX_TEST_IDS = [0, 1, 2]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
def test_get_prompt_logprobs(
    hf_runner,
    vllm_runner,
    model,
    dtype,
):
    hf_model = hf_runner(model, dtype=dtype)
    vllm_model = vllm_runner(model, dtype=dtype)

    # Test whether engine results include prompts.
    echo_params = SamplingParams(get_prompt_logprobs=True, max_tokens=5)
    echo_results = vllm_model.generate(TEST_PROMPTS,
                                       sampling_params=echo_params)
    for orig, (_, echoed) in zip(TEST_PROMPTS, echo_results):
        assert orig == echoed[0][:len(orig)]

    # Test whether prompt logprobs are included in the results.
    echo_logprob_params = SamplingParams(get_prompt_logprobs=True,
                                         max_tokens=5,
                                         logprobs=0,
                                         temperature=0.0)
    echo_logprob_results = vllm_model.model.generate(
        TEST_PROMPTS, sampling_params=echo_logprob_params)

    # This is for the case `logprobs=0` indicating only the chosen tokens.
    for result in echo_logprob_results:
        assert result.outputs[0].logprobs is not None

    # This also ensures that same prompts have the same prefix logprobs.
    same_prefix_logprobs = None
    for i in SAME_PREFIX_TEST_IDS:
        result = echo_logprob_results[i]
        prefix_logprobs = result.outputs[0].logprobs[:len(result.
                                                          prompt_token_ids)]
        if same_prefix_logprobs is None:
            same_prefix_logprobs = prefix_logprobs
        else:
            assert all(x == y
                       for x, y in zip(same_prefix_logprobs, prefix_logprobs))

    # To test whether prompt logprobs are consistent with HF
    hf_outputs = hf_model.generate(
        TEST_PROMPTS,
        raw_output=True,
        do_sample=False,
        max_new_tokens=5,
        output_hidden_states=True,
        output_scores=True,
    )
    hf_logprobs_list = []
    for output in hf_outputs:
        logits = torch.matmul(
            output.hidden_states[0][-1],
            hf_model.model.get_output_embeddings().weight.t(),
        )[0]  # batch_size=1
        if hf_model.model.get_output_embeddings().bias is not None:
            logits += hf_model.model.get_output_embeddings().bias.unsqueeze(0)
        hf_logprobs_list.append(logits.log_softmax(dim=-1))

    for vllm_result, hf_result in zip(echo_logprob_results, hf_logprobs_list):
        prompt_token_ids = torch.tensor(vllm_result.prompt_token_ids[1:],
                                        dtype=torch.long)
        vllm_logprobs = torch.tensor([
            vllm_result.outputs[0].logprobs[i + 1][tid]
            for i, tid in enumerate(vllm_result.prompt_token_ids[1:])
        ])
        hf_logprobs = hf_result[:-1].cpu()[
            torch.arange(prompt_token_ids.shape[0]), prompt_token_ids]
        assert vllm_logprobs.shape[0] == hf_logprobs.shape[0]
        # This is not super tight due to multiple float point conversions
        assert torch.isclose(
            vllm_logprobs,
            hf_logprobs.float(),
            atol=3e-2,
            rtol=5e-3,
        ).all()

    del hf_model
    del vllm_model
