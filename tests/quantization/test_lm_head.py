"""Tests whether gptq models with quantized lm_head can be loaded.

Run `pytest tests/quantization/test_quant_lm_head_true.py --forked`.
"""
from typing import Tuple

import pytest
import torch

from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.gptq import GPTQLinearMethod
from vllm.model_executor.layers.quantization.gptq_marlin import (
    GPTQMarlinLinearMethod)
from vllm.model_executor.layers.quantization.marlin import MarlinLinearMethod

PROMPT = "On the surface of Mars, we found"

MODELS_QUANT = [(
    "LnL-AI/TinyLlama-1.1B-intermediate-step-1341k-3T-autoround-lm_head-symFalse",
    True), ("TheBloke/TinyLlama-1.1B-Chat-v1.0-GPTQ", False),
                ("neuralmagic/Meta-Llama-3-8B-Instruct-FP8", False)]


@pytest.mark.parametrize("model_lm_head_quant", MODELS_QUANT)
def test_lm_head(
    vllm_runner,
    model_lm_head_quant: Tuple[str, bool],
) -> None:
    model, lm_head_quantized = model_lm_head_quant
    vllm_model = vllm_runner(model, dtype=torch.float16, max_model_len=2048)

    lm_head_layer = (vllm_model.model.llm_engine.model_executor.driver_worker.
                     model_runner.model.lm_head)

    if lm_head_quantized:
        assert isinstance(
            lm_head_layer.linear_method,
            (GPTQLinearMethod, GPTQMarlinLinearMethod, MarlinLinearMethod))
    else:
        assert isinstance(lm_head_layer.linear_method, UnquantizedLinearMethod)

    print(
        vllm_model.generate_greedy(prompts=["Hello my name is"],
                                   max_tokens=10)[0][1])
    del vllm_model
