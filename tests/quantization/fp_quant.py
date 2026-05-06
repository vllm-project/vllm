# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test model set-up and inference for quantized HF models supported
on the GPU backend using FPQuant.

Validating the configuration and printing results for manual checking.

Run `pytest tests/quantization/test_fp_quant.py`.
"""

import pytest

from tests.quantization.utils import is_quant_method_supported

MODELS = [
    "ISTA-DASLab/Qwen3-0.6B-RTN-NVFP4",
    "ISTA-DASLab/Qwen3-0.6B-RTN-MXFP4",
]
DTYPE = ["bfloat16"]
EAGER = [True, False]


@pytest.mark.skipif(
    not is_quant_method_supported("fp_quant"),
    reason="FPQuant is not supported on this GPU type.",
)
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("eager", EAGER)
def test_fpquant(vllm_runner, model, eager):
    with vllm_runner(model, enforce_eager=eager) as llm:
        output = llm.generate_greedy(["1 2 3 4 5"], max_tokens=2)
    assert output[0][1] == "1 2 3 4 5 6"
