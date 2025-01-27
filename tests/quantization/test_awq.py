"""Test model set-up and inference for quantized HF models supported
 on the HPU backend using AutoAWQ.
 
 Validating the configuration and printing results for manual checking.

 Run `pytest tests/quantization/test_awq.py`.
"""

import pytest

from vllm.platforms import current_platform

MODELS = [
    "TheBloke/Llama-2-7B-Chat-AWQ",
]
DTYPE = ["bfloat16"]


@pytest.mark.skipif(not current_platform.is_hpu(),
                    reason="only supports Intel HPU backend.")
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", DTYPE)
def test_awq(vllm_runner, model, dtype):
    with vllm_runner(model, dtype=dtype, quantization='awq_hpu') as llm:
        output = llm.generate_greedy(["The capital of France is"],
                                     max_tokens=32)
    assert output
    print(output)
