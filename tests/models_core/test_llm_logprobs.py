"""Compare the outputs of HF and vLLM when using greedy sampling.

Because of numerical precision and the fact that we are generating
over so many samples, we look

Run `pytest tests/models/test_models_logprobs.py`.
"""
import pytest

from tests.models.utils import check_logprobs_close
from tests.nm_utils.utils_skip import should_skip_test_group

if should_skip_test_group(group_name="TEST_MODELS_CORE"):
    pytest.skip("TEST_MODELS_CORE=DISABLE, skipping core model test group",
                allow_module_level=True)

MODEL_MAX_LEN = 1024

MODELS = [
    # Llama (8B param variant)
    "meta-llama/Meta-Llama-3-8B-Instruct",
    # Qwen2 (7B param variant)
    "Qwen/Qwen2-7B-Instruct",
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [5])
def test_models(
    vllm_runner_nm,
    hf_runner_nm,
    example_prompts,
    model: str,
    max_tokens: int,
    num_logprobs: int,
) -> None:
    hf_model = hf_runner_nm(model)
    hf_outputs = hf_model.generate_greedy_logprobs_nm(example_prompts,
                                                      max_tokens, num_logprobs)

    del hf_model

    vllm_model = vllm_runner_nm(model, max_model_len=MODEL_MAX_LEN)
    vllm_outputs = vllm_model.generate_greedy_logprobs(example_prompts,
                                                       max_tokens,
                                                       num_logprobs)

    del vllm_model

    # loop through the prompts
    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf_model",
        name_1="vllm_model",
    )
