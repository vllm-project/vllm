"""Compare the outputs of HF and vLLM for Granite models using greedy sampling.

Run `pytest tests/models/test_granite.py`.
"""
import importlib.metadata

import pytest

from ...utils import check_logprobs_close

TRANSFORMERS_VERSION = tuple(
    map(int,
        importlib.metadata.version("transformers").split(".")))

MODELS = [
    "ibm/PowerLM-3b",
]


# GraniteForCausalLM will be in transformers >= 4.45
@pytest.mark.skipif(TRANSFORMERS_VERSION < (4, 45),
                    reason="granite model test requires transformers >= 4.45")
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

    with vllm_runner(model, dtype=dtype) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs)
    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )
