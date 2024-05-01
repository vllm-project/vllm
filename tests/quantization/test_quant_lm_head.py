"""Tests whether gptq models with quantized lm_head can be loaded.

Run `pytest tests/quantization/test_quant_lm_head_true.py --forked`.
"""
from typing import Optional

import pytest
import torch

from vllm import CompletionOutput, SamplingParams
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.gptq import GPTQLinearMethod

# Model Id // Expected output
MODELS_QUANT_TYPE = [
    ("LnL-AI/TinyLlama-1.1B-intermediate-step-1341k-3T-autoround-lm_head-symFalse",
     "On the surface of Mars, we found",
     " a lot of water, but we didn't find any life."),
    ("LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit",
     "On the surface of Mars, we found",
     " a rocky terrain that is similar to the Martian surface."),
    ("LnL-AI/opt-125M-autoround-lm_head-true-symTrue",
     "On the surface of Mars, we found",
     " a lot of evidence for the existence of life on Earth."),
    ("LnL-AI/opt-125M-autoround-lm_head-false-symTrue",
     "On the surface of Mars, we found",
     " a lot of evidence of life. But the evidence is not conclusive."),
]

MAX_TOKENS = 20


@pytest.mark.parametrize("model_quant_type", MODELS_QUANT_TYPE)
def test_lm_head(
    vllm_runner,
    model_quant_type: str,
) -> None:
    model, prompt, expected_output = model_quant_type
    vllm_model = vllm_runner(model,
                             dtype=torch.float16,
                             enforce_eager=True,
                             enable_prefix_caching=True,
                             gpu_memory_utilization=0.5,
                             max_model_len=2048,
                             seed=898,
                             tensor_parallel_size=1)
    llm_engine = vllm_model.model.llm_engine

    quantization_config = llm_engine.model_config.hf_config.quantization_config
    quant_lm_head = False
    if quantization_config.get("lm_head"):
        quant_lm_head = True

    lm_head_layer = (vllm_model.model.llm_engine.model_executor.driver_worker.
                     model_runner.model.lm_head)

    if quant_lm_head:
        assert isinstance(lm_head_layer.linear_method, GPTQLinearMethod)
    else:
        assert isinstance(lm_head_layer.linear_method, UnquantizedLinearMethod)

    llm_engine.add_request(
        "id", prompt, SamplingParams(
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
