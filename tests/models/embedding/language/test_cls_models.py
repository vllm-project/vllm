"""Compare the classification outputs of HF and vLLM models.

Run `pytest tests/models/test_cls_models.py`.
"""
import pytest
import torch
from transformers import AutoModelForSequenceClassification


@pytest.mark.parametrize(
    "model",
    [
        pytest.param("jason9693/Qwen2.5-1.5B-apeach",
                     marks=[pytest.mark.core_model, pytest.mark.cpu_model]),
    ],
)
@pytest.mark.parametrize("dtype", ["float"])
def test_classification_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
) -> None:
    with vllm_runner(model, dtype=dtype) as vllm_model:
        vllm_outputs = vllm_model.classify(example_prompts)

        # This test is for verifying whether the model's extra_repr
        # can be printed correctly.
        def print_model(model):
            print(model)

        vllm_model.apply_model(print_model)

    with hf_runner(model,
                   dtype=dtype,
                   auto_cls=AutoModelForSequenceClassification) as hf_model:
        hf_outputs = hf_model.classify(example_prompts)

    # check logits difference
    for hf_output, vllm_output in zip(hf_outputs, vllm_outputs):
        hf_output = torch.tensor(hf_output)
        vllm_output = torch.tensor(vllm_output)

        assert torch.allclose(hf_output, vllm_output, 1e-3)
