"""Tests whether gptq models with quantized lm_head can be loaded.

Run `pytest tests/quantization/test_quant_lm_head_true.py --forked`.
"""
from typing import Optional, Tuple

import pytest
import torch

from vllm import CompletionOutput, SamplingParams
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.gptq import GPTQLinearMethod
from vllm.model_executor.layers.quantization.gptq_marlin import (
    GPTQMarlinLinearMethod)
from vllm.model_executor.layers.quantization.marlin import MarlinLinearMethod

PROMPT = "On the surface of Mars, we found"

# Model Id // Expected output
MODELS_QUANT_TYPE = [
    ("LnL-AI/TinyLlama-1.1B-intermediate-step-1341k-3T-autoround-lm_head-symFalse",
     " a lot of water, but we didn't find any life."),
    ("LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit",
     " a rocky terrain that is similar to the Martian surface."),
    # test non-quantized model as baseline to ensure OPT works
    ("facebook/opt-125m",
     " a lot of evidence that the planet is a very large, rocky planet."),
    ("LnL-AI/opt-125M-autoround-lm_head-true-symTrue",
     " a lot of evidence for the existence of life on Earth."),
    ("LnL-AI/opt-125M-autoround-lm_head-false-symTrue",
     " a lot of evidence of life. But the evidence is not conclusive."),
]

MAX_TOKENS = 20


@pytest.mark.parametrize("model_quant_type", MODELS_QUANT_TYPE)
def test_lm_head(
    vllm_runner,
    model_quant_type: Tuple[str, str],
) -> None:
    model, expected_output = model_quant_type
    vllm_model = vllm_runner(model,
                             dtype=torch.float16,
                             enforce_eager=True,
                             enable_prefix_caching=True,
                             gpu_memory_utilization=0.5,
                             max_model_len=2048,
                             seed=898,
                             tensor_parallel_size=1)
    llm_engine = vllm_model.model.llm_engine

    lm_head_quantized = False

    quantization_config = llm_engine.model_config.hf_config if hasattr(
        llm_engine.model_config.hf_config, "quantization_config") else None
    if quantization_config is not None:
        quantization_config = (
            llm_engine.model_config.hf_config.quantization_config)

        if quantization_config.get("lm_head"):
            lm_head_quantized = True

    lm_head_layer = (vllm_model.model.llm_engine.model_executor.driver_worker.
                     model_runner.model.lm_head)

    if lm_head_quantized:
        assert isinstance(
            lm_head_layer.linear_method,
            (GPTQLinearMethod, GPTQMarlinLinearMethod, MarlinLinearMethod))

    else:
        assert isinstance(lm_head_layer.linear_method, UnquantizedLinearMethod)

    llm_engine.add_request(
        "id", PROMPT, SamplingParams(
            temperature=0.0,
            max_tokens=MAX_TOKENS,
        ), None)

    output: Optional[CompletionOutput] = None
    output_text = ""
    while llm_engine.has_unfinished_requests():
        (request_output, ) = llm_engine.step()
        (output, ) = request_output.outputs

        # Ensure we don't backtrack
        assert output.text.startswith(output_text)
        output_text = output.text

    print(f"output: {output_text}")
    assert output is not None
    assert output_text.startswith(expected_output)
