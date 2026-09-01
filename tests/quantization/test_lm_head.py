# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests whether gptq models with quantized lm_head can be loaded.

Run `pytest tests/quantization/test_quant_lm_head_true.py --forked`.
"""

import pytest
import torch

from tests.quantization.utils import load_model_without_vllm_runner
from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQLinearMethod
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
    model_id: str,
    lm_head_quantized: bool,
    monkeypatch,
    dist_init,
    workspace_init,
) -> None:
    # `LLM.apply_model` requires pickling a function.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    model, _ = load_model_without_vllm_runner(
        model_id,
        dtype=torch.float16,
        model_config_kwargs={
            "max_model_len": 2048,
            "hf_overrides": {"num_hidden_layers": 3},
        },
    )
    lm_head_layer = model.lm_head
    if lm_head_quantized:
        assert isinstance(lm_head_layer.quant_method, AutoGPTQLinearMethod)
    else:
        assert isinstance(lm_head_layer.quant_method, UnquantizedEmbeddingMethod)
