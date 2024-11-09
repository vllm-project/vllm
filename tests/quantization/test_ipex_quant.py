"""Test model set-up and inference for quantized HF models supported
 on the CPU backend using IPEX (including AWQ).
 
 Validating the configuration and printing results for manual checking.

 Run `pytest tests/quantization/test_ipex_quant.py`.
"""

import pytest

from vllm.platforms import current_platform

MODELS = [
    "casperhansen/llama-3-8b-instruct-awq",
]
DTYPE = ["bfloat16"]


@pytest.mark.skipif(not current_platform.is_cpu(),
                    reason="only supports the CPU backend.")
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", DTYPE)
def test_ipex_quant(vllm_runner, model, dtype):
    with vllm_runner(model, dtype=dtype) as llm:
        output = llm.generate_greedy(["The capital of France is"],
                                     max_tokens=32)
    assert output
    print(output)
