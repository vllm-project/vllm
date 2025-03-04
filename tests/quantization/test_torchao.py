# SPDX-License-Identifier: Apache-2.0
"""Test model set-up and inference for TorchAO quantized HF models supported
 on the CPU/GPU backend.

 Validating the configuration and printing results for manual checking.

 Run `pytest tests/quantization/test_torchao_quant.py`.
"""
import pytest
from torchao.quantization.quant_api import (
    Float8DynamicActivationFloat8WeightConfig, Int4WeightOnlyConfig,
    Int8DynamicActivationInt8WeightConfig, PerRow)

MODELS = ["facebook/opt-125m"]
DTYPE = ["bfloat16"]
CONFIGS = [
    Int4WeightOnlyConfig(),
    Int8DynamicActivationInt8WeightConfig(),
    Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()),
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", DTYPE)
@pytest.mark.parametrize("config", CONFIGS)
# @fork_new_process_for_each_test
def test_torchao(vllm_runner, model, dtype, config):
    with vllm_runner(
            model,
            quantization="torchao",
            dtype=dtype,
            # quantization_config={"config": config}
    ) as llm:
        output = llm.generate_greedy(["The capital of France is"],
                                     max_tokens=32)
    assert output
    print(output)
