# SPDX-License-Identifier: Apache-2.0
"""Tests whether gptq models with dynamic quantized can be loaded.

Run `pytest tests/quantization/test_gptq_dynamic.py --forked`.
"""

import pytest
import torch

from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.gptq import GPTQLinearMethod
from vllm.model_executor.layers.quantization.gptq_marlin import (
    GPTQMarlinLinearMethod)
from vllm.model_executor.layers.quantization.utils.gptq_utils import (
    get_dynamic_override)

PROMPT = "On the surface of Mars, we found"

# The first layer is quantized using bits=4, group_size=128
# The second layer is quantized using bits=8, group_size=32
# All other layers (layer index >= 2) are not quantized
MODEL_QUANT = [
    ("ModelCloud/Qwen1.5-1.8B-Chat-GPTQ-4bits-dynamic-cfg-with-lm_head-symTrue",
     True),
    ("ModelCloud/Qwen1.5-1.8B-Chat-GPTQ-4bits-dynamic-cfg-with-lm_head-symFalse",
     False),
]


@pytest.mark.parametrize("model_id, use_marlin_kernel", MODEL_QUANT)
def test_gptq_with_dynamic(vllm_runner, model_id: str,
                           use_marlin_kernel: bool):

    vllm_model = vllm_runner(model_id, dtype=torch.float16, max_model_len=2048)

    linear_method_cls = GPTQMarlinLinearMethod if use_marlin_kernel else (
        GPTQLinearMethod)

    for name, submodule in (vllm_model.model.llm_engine.model_executor.
                            driver_worker.model_runner.model.named_modules()):
        if name == "lm_head":
            assert isinstance(submodule.quant_method, linear_method_cls)
        elif name == 'model.layers.0.self_attn.qkv_proj':
            # The first layer is quantized using bits=4, group_size=128
            # desc_act=True
            assert isinstance(submodule.quant_method, linear_method_cls)
            config = submodule.quant_method.quant_config
            assert config.weight_bits == 4
            assert config.group_size == 128
            assert config.desc_act
        elif name == 'model.layers.1.self_attn.qkv_proj':
            # The second layer is quantized using bits=8, group_size=32
            # desc_act=False
            assert isinstance(submodule.quant_method, linear_method_cls)
            config = submodule.quant_method.quant_config
            assert get_dynamic_override(config, layer_name=name,
                                        key="bits") == 8
            assert get_dynamic_override(config,
                                        layer_name=name,
                                        key="group_size") == 32
            assert not get_dynamic_override(
                config, layer_name=name, key="desc_act")
        elif (name == 'model.layers.2.self_attn.qkv_proj'
              or name == 'model.layers.2.mlp.gate_up_proj'):
            # All other layers (layer index >= 2) are not quantized
            assert isinstance(submodule.quant_method, UnquantizedLinearMethod)

    del vllm_model
