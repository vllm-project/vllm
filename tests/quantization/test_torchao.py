# SPDX-License-Identifier: Apache-2.0
"""Test model set-up and inference for TorchAO quantized HF models supported
 on the CPU/GPU backend.

 Validating the configuration and printing results for manual checking.

 Run `pytest tests/quantization/test_torchao_quant.py`.
"""
import pytest
from torchao.core.config import config_to_dict
from torchao.quantization.quant_api import (
    AffineQuantizedTensor, Float8DynamicActivationFloat8WeightConfig,
    Int4WeightOnlyConfig, Int8DynamicActivationInt8WeightConfig, PerRow)

from vllm.config import CompilationLevel

MODELS = ["TinyLlama/TinyLlama_v1.1"]
DTYPE = ["bfloat16"]
CONFIGS = [
    Int4WeightOnlyConfig(),
    Int8DynamicActivationInt8WeightConfig(),
    Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()),
]


def check_all_weights_quantized(model):
    """
    Check if all major linear components in a LLaMA model have weights that are 
    instances of AffineQuantizedTensor
    """
    for layer_idx, layer in enumerate(model.model.layers):
        # Key linear components to check
        components = {
            f"layer{layer_idx}.self_attn.qkv_proj": layer.self_attn.qkv_proj,
            f"layer{layer_idx}.self_attn.o_proj": layer.self_attn.o_proj,
            f"layer{layer_idx}.mlp.gate_up_proj": layer.mlp.gate_up_proj,
            f"layer{layer_idx}.mlp.down_proj": layer.mlp.down_proj
        }

        for name, component in components.items():
            # Check if weight is an AffineQuantizedTensor
            if not hasattr(component, 'weight'):
                assert isinstance(
                    component,
                    AffineQuantizedTensor), f"{name} is not quantized"


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", DTYPE)
@pytest.mark.parametrize("config", CONFIGS)
def test_torchao(vllm_runner, model, dtype, config):
    hf_override_dict = {
        "quant_method": "torchao",
        "quant_type": {
            "default": config_to_dict(config)
        },
        "runtime_quant": True
    }
    with vllm_runner(model,
                     quantization="torchao",
                     dtype=dtype,
                     hf_overrides={"quantization_config": hf_override_dict},
                     enforce_eager=True) as llm:
        output = llm.generate_greedy(["The capital of France is"],
                                     max_tokens=32)

        llm.apply_model(check_all_weights_quantized)
        print(output)


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
