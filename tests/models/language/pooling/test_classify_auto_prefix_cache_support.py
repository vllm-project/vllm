# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: SIM117
# Keep Decode-only SequenceClassification models support auto prefix cache
import pytest
import torch
from transformers import AutoModelForSequenceClassification


@pytest.mark.parametrize(
    "model",
    ["jason9693/Qwen2.5-1.5B-apeach"],
)
@pytest.mark.parametrize("dtype", ["half"])
def test_decode_only_classify(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    monkeypatch,
) -> None:

    with vllm_runner(model,
                     max_model_len=512,
                     dtype=dtype,
                     enable_prefix_caching=True) as vllm_model:
        vllm_outputs = vllm_model.classify(example_prompts)

    with hf_runner(model,
                   dtype=dtype,
                   auto_cls=AutoModelForSequenceClassification) as hf_model:
        hf_outputs = hf_model.classify(example_prompts)

    for hf_output, vllm_output in zip(hf_outputs, vllm_outputs):
        hf_output = torch.tensor(hf_output)
        vllm_output = torch.tensor(vllm_output)

        assert torch.allclose(hf_output, vllm_output,
                              1e-3 if dtype == "float" else 1e-2)


@pytest.mark.parametrize(
    "model",
    ["Alibaba-NLP/gte-Qwen2-1.5B-instruct"],
)
@pytest.mark.parametrize("dtype", ["half"])
def test_encode_only_classify(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    monkeypatch,
) -> None:
    with pytest.raises(RuntimeError):
        with vllm_runner(model,
                         max_model_len=512,
                         dtype=dtype,
                         enable_prefix_caching=True) as vllm_model:
            vllm_model.classify(example_prompts)
    # Is there any way to capture errors in worker processes?
    # NotImplementedError: Encoder self-attention and encoder/decoder
    # cross-attention are not implemented for FlashAttentionImpl
