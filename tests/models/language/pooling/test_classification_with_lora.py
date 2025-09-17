# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest


@pytest.mark.parametrize(
    "model",
    ["jason9693/Qwen2.5-1.5B-apeach"],
)
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
) -> None:

    with vllm_runner(model, max_model_len=512, enable_lora=True) as vllm_model:
        model_config = vllm_model.llm.llm_engine.model_config
        head_dtype = model_config.head_dtype
        dtype = model_config.dtype
        assert head_dtype == dtype
