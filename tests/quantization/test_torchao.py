# SPDX-License-Identifier: Apache-2.0
"""Test model set-up and inference for quantized HF models supported
 on the CPU/GPU backend using IPEX (including AWQ/GPTQ).

 Validating the configuration and printing results for manual checking.

 Run `pytest tests/quantization/test_ipex_quant.py`.
"""

import pytest

from tests.quantization.utils import is_quant_method_supported
from tests.utils import fork_new_process_for_each_test

MODELS = ["facebook/opt-125m"]
DTYPE = ["bfloat16"]


@pytest.mark.skipif(not is_quant_method_supported("torchao"),
                    reason="torchao is not supported on this GPU type.")
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", DTYPE)
@fork_new_process_for_each_test
def test_torchao(vllm_runner, model, dtype):
    with vllm_runner(model, quantization="torchao", dtype=dtype) as llm:
        output = llm.generate_greedy(["The capital of France is"],
                                     max_tokens=32)
    assert output
    print(output)
