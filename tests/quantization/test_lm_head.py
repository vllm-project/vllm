# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests whether gptq models with quantized lm_head can be loaded.

Run `pytest tests/quantization/test_quant_lm_head_true.py --forked`.
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.gptq import GPTQLinearMethod
from vllm.model_executor.layers.quantization.gptq_marlin import GPTQMarlinLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)

PROMPT = "On the surface of Mars, we found"

MODELS_QUANT = [
    ("ModelCloud/Qwen1.5-1.8B-Chat-GPTQ-4bits-dynamic-cfg-with-lm_head", True),
    ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ", False),
]


@pytest.mark.parametrize("model_id, lm_head_quantized", MODELS_QUANT)
def test_lm_head(
    vllm_runner,
    model_id: str,
    lm_head_quantized: bool,
    monkeypatch,
) -> None:
    # `LLM.apply_model` requires pickling a function.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    with vllm_runner(
        model_id, dtype=torch.float16, max_model_len=2048, enforce_eager=True
    ) as vllm_model:

        def check_model(model):
            lm_head_layer = model.lm_head
            if lm_head_quantized:
                assert isinstance(
                    lm_head_layer.quant_method,
                    (GPTQLinearMethod, GPTQMarlinLinearMethod),
                )
            else:
                assert isinstance(
                    lm_head_layer.quant_method, UnquantizedEmbeddingMethod
                )

        vllm_model.apply_model(check_model)

        print(vllm_model.generate_greedy(["Hello my name is"], max_tokens=4)[0][1])
