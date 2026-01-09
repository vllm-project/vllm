# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test model set-up and inference for quantized HF models supported
on the AutoRound.

Validating the configuration and printing results for manual checking.

Run `pytest tests/quantization/test_auto_round.py`.
"""

import pytest

from vllm.platforms import current_platform

MODELS = [
    "OPEA/Qwen2.5-0.5B-Instruct-int4-sym-inc",  ##auto_round:auto_gptq
    "Intel/Qwen2-0.5B-Instruct-int4-sym-AutoRound",  ##auto_round:auto_awq
]


@pytest.mark.skipif(
    not current_platform.is_cpu()
    and not current_platform.is_xpu()
    and not current_platform.is_cuda(),
    reason="only supports CPU/XPU/CUDA backend.",
)
@pytest.mark.parametrize("model", MODELS)
def test_auto_round(vllm_runner, model):
    with vllm_runner(
        model, enforce_eager=True, allow_deprecated_quantization=True
    ) as llm:
        output = llm.generate_greedy(["The capital of France is"], max_tokens=8)
    assert output
    print(f"{output[0][1]}")
