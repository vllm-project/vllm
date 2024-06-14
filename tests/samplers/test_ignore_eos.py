"""Make sure ignore_eos works.

Run `pytest tests/samplers/test_ignore_eos.py`.
"""

import pytest

from tests.nm_utils.utils_skip import should_skip_test_group
from vllm import SamplingParams

if should_skip_test_group(group_name="TEST_SAMPLERS"):
    pytest.skip("TEST_SAMPLERS=DISABLE, skipping sampler test group",
                allow_module_level=True)

# We also test with llama because it has generation_config to specify EOS
# (past regression).
MODELS = ["facebook/opt-125m", "meta-llama/Llama-2-7b-hf"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", [512])
def test_ignore_eos(
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
) -> None:
    with vllm_runner(model, dtype=dtype) as vllm_model:
        sampling_params = SamplingParams(max_tokens=max_tokens,
                                         ignore_eos=True)

        for prompt in example_prompts:
            ignore_eos_output = vllm_model.model.generate(
                prompt, sampling_params=sampling_params)
            output_length = len(ignore_eos_output[0].outputs[0].token_ids)
            assert output_length == max_tokens
