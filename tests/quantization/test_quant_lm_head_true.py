"""Tests whether gptq models with quantized lm_head can be loaded.

Run `pytest tests/quantization/test_quant_lm_head_true.py --forked`.
"""
from typing import Optional

import pytest

from vllm import CompletionOutput, SamplingParams

MODEL = ("LnL-AI/TinyLlama-1.1B-intermediate-step-1341k-3T-autoround-lm_head-"
         "symFalse")
MAX_TOKENS = 20


@pytest.fixture
def vllm_model(vllm_runner):
    return vllm_runner(
        MODEL,
        enforce_eager=True,
    )


def test_lm_head_true(vllm_model):
    llm_engine = vllm_model.model.llm_engine

    quantization_config = llm_engine.model_config.hf_config.quantization_config
    assert quantization_config.get("lm_head")

    llm_engine.add_request(
        "id", "A story about vLLM:\n",
        SamplingParams(
            temperature=0.0,
            max_tokens=MAX_TOKENS,
        ), None)

    expected_output = "VLLM is a very popular and successful program"

    output: Optional[CompletionOutput] = None
    output_text = ""
    while llm_engine.has_unfinished_requests():
        (request_output, ) = llm_engine.step()
        (output, ) = request_output.outputs

        # Ensure we don't backtrack
        assert output.text.startswith(output_text)
        output_text = output.text

    assert output is not None
    assert output_text.startswith(expected_output)
    