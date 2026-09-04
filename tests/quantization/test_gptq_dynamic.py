# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests whether gptq models with dynamic quantized can be loaded.

Run `pytest tests/quantization/test_gptq_dynamic.py --forked`.

Note: Only symmetric GPTQ models are supported after consolidation to Marlin.
"""

import pytest
import torch

from tests.quantization.utils import load_model_without_vllm_runner
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQLinearMethod
from vllm.model_executor.layers.quantization.utils.gptq_utils import (
    get_dynamic_override,
)

PROMPT = "On the surface of Mars, we found"

# The first layer is quantized using bits=4, group_size=128
# The second layer is quantized using bits=8, group_size=32
# All other layers (layer index >= 2) are not quantized
# Note: Only symmetric models are supported with Marlin kernels
MODELS = [
    "ModelCloud/Qwen1.5-1.8B-Chat-GPTQ-4bits-dynamic-cfg-with-lm_head-symTrue",
]


@pytest.mark.parametrize("model_id", MODELS)
def test_gptq_with_dynamic(model_id: str, monkeypatch, dist_init, workspace_init):
    # `LLM.apply_model` requires pickling a function.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    linear_method_cls = AutoGPTQLinearMethod

    model, _ = load_model_without_vllm_runner(
        model_id,
        dtype=torch.float16,
        model_config_kwargs={
            "max_model_len": 2048,
            "hf_overrides": {"num_hidden_layers": 3},
        },
    )

    for name, submodule in model.named_modules():
        if name == "lm_head":
            assert isinstance(submodule.quant_method, linear_method_cls)
        elif name == "model.layers.0.self_attn.qkv_proj":
            assert isinstance(submodule.quant_method, linear_method_cls)
            config = submodule.quant_method.quant_config
            assert config.weight_bits == 4
            assert config.group_size == 128
            assert config.desc_act
        elif name == "model.layers.1.self_attn.qkv_proj":
            assert isinstance(submodule.quant_method, linear_method_cls)
            config = submodule.quant_method.quant_config
            assert get_dynamic_override(config, layer_name=name, key="bits") == 8
            assert get_dynamic_override(config, layer_name=name, key="group_size") == 32
            assert not get_dynamic_override(config, layer_name=name, key="desc_act")
        elif name in (
            "model.layers.2.self_attn.qkv_proj",
            "model.layers.2.mlp.gate_up_proj",
        ):
            assert isinstance(submodule.quant_method, UnquantizedLinearMethod)
