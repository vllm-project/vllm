# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from transformers import AutoModelForSequenceClassification


@pytest.mark.parametrize(
    "model",
    ["nie3e/sentiment-polish-gpt2-small"],
)
@pytest.mark.parametrize("dtype", ["half"])
def test_classify_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
) -> None:
    with hf_runner(
        model, dtype=dtype, auto_cls=AutoModelForSequenceClassification
    ) as hf_model:
        hf_outputs = hf_model.classify(example_prompts)

    for head_dtype_str in ["float32", "model"]:
        with vllm_runner(
            model,
            max_model_len=512,
            dtype=dtype,
            hf_overrides={"head_dtype": head_dtype_str},
        ) as vllm_model:
            model_config = vllm_model.llm.llm_engine.model_config
            model_dtype = model_config.dtype
            head_dtype = model_config.head_dtype

            if head_dtype_str == "float32":
                assert head_dtype == torch.float32
            elif head_dtype_str == "model":
                assert head_dtype == model_dtype

            vllm_outputs = vllm_model.classify(example_prompts)

        for hf_output, vllm_output in zip(hf_outputs, vllm_outputs):
            hf_output = torch.tensor(hf_output).float()
            vllm_output = torch.tensor(vllm_output).float()

            assert torch.allclose(hf_output, vllm_output, atol=1e-2)
