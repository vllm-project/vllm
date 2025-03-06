# SPDX-License-Identifier: Apache-2.0
"""Test model set-up and inference for TorchAO quantized HF models supported
 on the CPU/GPU backend.

 Validating the configuration and printing results for manual checking.

 Run `pytest tests/quantization/test_torchao_quant.py`.
"""
import pytest

from vllm.config import CompilationLevel

DTYPE = ["bfloat16"]


def test_pre_quantized_model(vllm_runner):
    with vllm_runner("drisspg/float8_dynamic_act_float8_weight-opt-125m",
                     quantization="torchao",
                     dtype="bfloat16",
                     compilation_config=CompilationLevel.PIECEWISE) as llm:
        output = llm.generate_greedy(["The capital of France is"],
                                     max_tokens=32)
    assert output
    print(output)


if __name__ == "__main__":
    pytest.main([__file__])
