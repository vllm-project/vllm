"""Compare the outputs of HF and vLLM for Mistral models using greedy sampling.

Run `pytest tests/models/test_mistral.py`.
"""
import pytest

from ...utils import check_logprobs_close

MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.3",
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    # TODO(sang): Sliding window should be tested separately.
    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            example_prompts, max_tokens, num_logprobs)

    with vllm_runner(model, dtype=dtype,
                     tokenizer_mode="mistral") as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs)

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )


@pytest.mark.parametrize("model", MODELS[1:])
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [5])
def test_mistral_format(
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    with vllm_runner(
            model,
            dtype=dtype,
            tokenizer_mode="auto",
            load_format="safetensors",
            config_format="hf",
    ) as hf_format_model:
        hf_format_outputs = hf_format_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs)

    with vllm_runner(
            model,
            dtype=dtype,
            tokenizer_mode="mistral",
            load_format="mistral",
            config_format="mistral",
    ) as mistral_format_model:
        mistral_format_outputs = mistral_format_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs)

    check_logprobs_close(
        outputs_0_lst=hf_format_outputs,
        outputs_1_lst=mistral_format_outputs,
        name_0="hf",
        name_1="mistral",
    )
