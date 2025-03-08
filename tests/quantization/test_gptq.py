"""Test model set-up and inference for quantized HF models supported
 on the HPU backend using AutoGPTQ.
 
 Validating the configuration and printing results for manual checking.

 Run `pytest tests/quantization/test_gptq.py`.
"""

import pytest

from vllm.platforms import current_platform

MODELS = [
    "TheBloke/Llama-2-7B-Chat-GPTQ",
]
DTYPE = ["bfloat16"]


@pytest.mark.skipif(not current_platform.is_hpu(),
                    reason="only supports Intel HPU backend.")
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", DTYPE)
def test_gptq(vllm_runner, model, dtype):
    with vllm_runner(model, dtype=dtype, quantization='gptq_hpu') as llm:
        output = llm.generate_greedy(["The capital of France is"],
                                     max_tokens=32)
    assert output
    print(output)