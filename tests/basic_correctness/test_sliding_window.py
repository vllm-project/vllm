import pytest
from vllm import LLM, SamplingParams
from vllm.transformers_utils.config import get_config


@pytest.mark.parametrize(
    "model", ["mistralai/Mistral-7B-v0.1",])
            #   "HuggingFaceH4/zephyr-7b-beta"])
def test_no_crash(model: str):
    """Basic test to verify that models with sliding windows do not crash.
    """
    hf_config = get_config(model, trust_remote_code=False, revision=None)
    sliding_window_size = hf_config.sliding_window
    llm = LLM(model, max_model_len=16384)

    input_len = sliding_window_size - 10
    final_len = sliding_window_size + 10
    dummy_prompt_id = 30
    dummy_prompt_ids = [[dummy_prompt_id] * input_len]

    max_tokens = final_len - input_len
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        ignore_eos=True,
    )

    output = llm.generate(
        prompt_token_ids=dummy_prompt_ids,
        sampling_params=sampling_params,
    )

    for request_output in output:
        for completion in request_output.outputs:
            token_ids = completion.token_ids
            assert len(token_ids) == max_tokens
