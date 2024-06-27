"""Compares vllm vs sparseml for compressed-tensors

Note: vllm and sparseml do not have bitwise correctness, 
so in this test, we just confirm that the top selected 
tokens of the are in the top 5 selections of each other.
"""

import pytest

from tests.quantization.utils import is_quant_method_supported

from .utils import check_logprobs_close

MODELS = [
    "nm-testing/Meta-Llama-3-8B-Instruct-W8-Channel-A8-Dynamic-Per-Token-Test",
]

MAX_TOKENS = 32
NUM_LOGPROBS = 5


@pytest.mark.skipif(
    not is_quant_method_supported("compressed-tensors"),
    reason="compressed-tensors is not supported on this machine type.")
@pytest.mark.parametrize("model_name", MODELS)
def test_models(
    vllm_runner,
    hf_runner,
    example_prompts,
    model_name,
) -> None:
    # Run sparseml.
    with hf_runner(model_name=model_name,
                   is_sparseml_model=True) as sparseml_model:

        sparseml_outputs = sparseml_model.generate_greedy_logprobs_limit(
            example_prompts, MAX_TOKENS, NUM_LOGPROBS)

    # Run vllm.
    with vllm_runner(model_name=model_name) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, MAX_TOKENS, NUM_LOGPROBS)

    check_logprobs_close(
        outputs_0_lst=sparseml_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="sparseml",
        name_1="vllm",
    )
