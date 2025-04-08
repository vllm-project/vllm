# SPDX-License-Identifier: Apache-2.0
import importlib.metadata
import importlib.util

import pytest

DTYPE = ["bfloat16"]

TORCHAO_AVAILABLE = importlib.util.find_spec("torchao") is not None


@pytest.mark.skipif(not TORCHAO_AVAILABLE, reason="torchao is not available")
def test_pre_quantized_model(vllm_runner):
    with vllm_runner("drisspg/float8_dynamic_act_float8_weight-opt-125m",
                     quantization="torchao",
                     dtype="bfloat16",
                     enforce_eager=True) as llm:
        output = llm.generate_greedy(["The capital of France is"],
                                     max_tokens=32)
    assert output
    print(output)


if __name__ == "__main__":
    pytest.main([__file__])
