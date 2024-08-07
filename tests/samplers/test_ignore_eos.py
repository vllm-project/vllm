"""Make sure ignore_eos works.

Run `pytest tests/samplers/test_ignore_eos.py`.
"""

import pytest

from vllm import SamplingParams

MODELS = ["facebook/opt-125m"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [1024])
def test_beam_search_single_input(
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    example_prompts = "1 + 1 is"

    vllm_model = vllm_runner(model, dtype=dtype)
    sampling_params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
    ignore_eos_output = vllm_model.model.generate(
        example_prompts, sampling_params=sampling_params)
    print(len(ignore_eos_output[0].outputs[0].token_ids))
    assert max_tokens - len(ignore_eos_output[0].outputs[0].token_ids) < 10
    assert max_tokens - len(ignore_eos_output[0].outputs[0].token_ids) >= 0
