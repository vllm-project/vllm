# SPDX-License-Identifier: Apache-2.0
"""Test model set-up and inference for TorchAO quantized HF models supported
 on the CPU/GPU backend.

 Validating the configuration and printing results for manual checking.

 Run `pytest tests/quantization/test_torchao_quant.py`.
"""
import pytest
import torch
from torchao.quantization.quant_api import LinearActivationQuantizedTensor

torch.serialization.add_safe_globals([set])


def check_all_weights_quantized(model):
    """
    Check if all major linear components in a LLaMA model have weights that are 
    instances of AffineQuantizedTensor
    """
    for layer_idx, layer in enumerate(model.model.decoder.layers):
        # Key linear components to check
        components = {
            f"layer{layer_idx}.self_attn.qkv_proj": layer.self_attn.qkv_proj,
            f"layer{layer_idx}.self_attn.out_proj": layer.self_attn.out_proj,
            f"layer{layer_idx}.mlp.gate_up_proj": layer.fc1.weight,
            f"layer{layer_idx}.mlp.down_proj": layer.fc2.weight
        }

        for name, component in components.items():
            # Check if weight is an AffineQuantizedTensor
            if not hasattr(component, 'weight'):
                assert isinstance(component, LinearActivationQuantizedTensor
                                  ), f"{name} is not quantized"


def test_pre_quantized_model(vllm_runner):
    with vllm_runner(
            "/home/drisspg/meta/scripts/data/f8a8-opt-125m_2",
            # quantization="torchao",
            dtype="bfloat16",
    ) as llm:
        output = llm.generate_greedy(["The capital of France is"],
                                     max_tokens=32)
        llm.apply_model(check_all_weights_quantized)
    assert output
    print(output)


if __name__ == "__main__":
    pytest.main([__file__])
