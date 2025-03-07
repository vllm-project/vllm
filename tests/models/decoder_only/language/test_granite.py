# SPDX-License-Identifier: Apache-2.0
"""Compare the outputs of HF and vLLM for Granite models using greedy sampling.

Run `pytest tests/models/test_granite.py`.
"""
import pytest

from vllm.platforms import current_platform

from ...utils import check_logprobs_close

MODELS = [
    # TODO(sang): Sliding window should be tested separately.
    "ibm/PowerLM-3b",
    "ibm/PowerMoE-3b",
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [64])
@pytest.mark.parametrize("num_logprobs", [5])
@pytest.mark.parametrize(
    "use_rocm_aiter", [True, False] if current_platform.is_rocm() else [False])
def test_models(hf_runner, vllm_runner, example_prompts, model: str,
                dtype: str, max_tokens: int, num_logprobs: int,
                use_rocm_aiter: bool, monkeypatch) -> None:
    if use_rocm_aiter:
        if monkeypatch.getenv("SKIP_ROCM_ATIER_MODEL_TEST_CASES") == "true":
            pytest.skip("Skipping test suite for ROCM AITER")
        monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")

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
