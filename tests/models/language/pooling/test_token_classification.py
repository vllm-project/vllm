# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from transformers import AutoModelForTokenClassification

from tests.models.utils import softmax
from vllm.platforms import current_platform


@pytest.mark.parametrize("model", ["boltuix/NeuroBERT-NER"])
# The float32 is required for this tiny model to pass the test.
@pytest.mark.parametrize("dtype", ["float"])
@torch.inference_mode
def test_bert_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
) -> None:
    with vllm_runner(model, max_model_len=None, dtype=dtype) as vllm_model:
        vllm_outputs = vllm_model.token_classify(example_prompts)

    # Use eager attention on ROCm to avoid HF Transformers flash attention
    # accuracy issues: https://github.com/vllm-project/vllm/issues/30167
    hf_model_kwargs = {}
    if current_platform.is_rocm():
        hf_model_kwargs["attn_implementation"] = "eager"

    with hf_runner(
        model,
        dtype=dtype,
        auto_cls=AutoModelForTokenClassification,
        model_kwargs=hf_model_kwargs,
    ) as hf_model:
        tokenizer = hf_model.tokenizer
        hf_outputs = []
        for prompt in example_prompts:
            inputs = tokenizer([prompt], return_tensors="pt")
            inputs = hf_model.wrap_device(inputs)
            output = hf_model.model(**inputs)
            hf_outputs.append(softmax(output.logits[0]))

    # check logits difference
    for hf_output, vllm_output in zip(hf_outputs, vllm_outputs):
        hf_output = hf_output.detach().clone().cpu().float()
        vllm_output = vllm_output.detach().clone().cpu().float()
        torch.testing.assert_close(hf_output, vllm_output, atol=1.2e-2, rtol=1e-3)


@pytest.mark.parametrize("model", ["disham993/electrical-ner-ModernBERT-base"])
@pytest.mark.parametrize("dtype", ["float"])
@torch.inference_mode
def test_modernbert_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
) -> None:
    with vllm_runner(model, max_model_len=None, dtype=dtype) as vllm_model:
        vllm_outputs = vllm_model.token_classify(example_prompts)

    # Use eager attention on ROCm to avoid HF Transformers flash attention
    # accuracy issues: https://github.com/vllm-project/vllm/issues/30167
    hf_model_kwargs = {}
    if current_platform.is_rocm():
        hf_model_kwargs["attn_implementation"] = "eager"

    with hf_runner(
        model,
        dtype=dtype,
        auto_cls=AutoModelForTokenClassification,
        model_kwargs=hf_model_kwargs,
    ) as hf_model:
        tokenizer = hf_model.tokenizer
        hf_outputs = []
        for prompt in example_prompts:
            inputs = tokenizer([prompt], return_tensors="pt")
            inputs = hf_model.wrap_device(inputs)
            output = hf_model.model(**inputs)
            hf_outputs.append(softmax(output.logits[0]))

    # check logits difference
    for hf_output, vllm_output in zip(hf_outputs, vllm_outputs):
        hf_output = hf_output.detach().clone().cpu().float()
        vllm_output = vllm_output.detach().clone().cpu().float()
        torch.testing.assert_close(hf_output, vllm_output, atol=1.2e-2, rtol=1e-3)


@pytest.mark.parametrize("model", ["bd2lcco/Qwen3-0.6B-finetuned"])
@pytest.mark.parametrize("dtype", ["float"])
@torch.inference_mode
def test_auto_conversion(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
) -> None:
    with vllm_runner(model, max_model_len=1024, dtype=dtype) as vllm_model:
        vllm_outputs = vllm_model.token_classify(example_prompts)

    with hf_runner(
        model, dtype=dtype, auto_cls=AutoModelForTokenClassification
    ) as hf_model:
        tokenizer = hf_model.tokenizer
        hf_outputs = []
        for prompt in example_prompts:
            inputs = tokenizer([prompt], return_tensors="pt")
            inputs = hf_model.wrap_device(inputs)
            output = hf_model.model(**inputs)
            hf_outputs.append(softmax(output.logits[0]))

    # check logits difference
    for hf_output, vllm_output in zip(hf_outputs, vllm_outputs):
        hf_output = hf_output.detach().clone().cpu().float()
        vllm_output = vllm_output.detach().clone().cpu().float()
        assert torch.allclose(hf_output, vllm_output, atol=1e-2)
