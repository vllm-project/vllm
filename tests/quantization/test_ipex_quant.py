# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test model set-up and inference for quantized HF models supported
on the CPU/GPU backend using IPEX (including AWQ/GPTQ).

Validating the configuration and printing results for manual checking.

Run `pytest tests/quantization/test_ipex_quant.py`.
"""

import pytest

from vllm.platforms import current_platform

MODELS = [
    "AMead10/Llama-3.2-1B-Instruct-AWQ",
    "shuyuej/Llama-3.2-1B-Instruct-GPTQ",  # with g_idx
]
DTYPE = ["bfloat16"]


@pytest.mark.skipif(
    not current_platform.is_cpu() and not current_platform.is_xpu(),
    reason="only supports Intel CPU/XPU backend.",
)
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", DTYPE)
def test_ipex_quant(vllm_runner, model, dtype):
    with vllm_runner(model, dtype=dtype) as llm:
        output = llm.generate_greedy(["The capital of France is"], max_tokens=32)
    assert output
    print(output)
