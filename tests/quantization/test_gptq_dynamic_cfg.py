"""Tests whether gptq models with dynamic_cfg quantized can be loaded.

Run `pytest tests/quantization/test_gptq_dynamic_cfg.py --forked`.
"""

import pytest
import torch

from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.gptq_marlin import (
    GPTQMarlinLinearMethod)

PROMPT = "On the surface of Mars, we found"

# The first layer is quantized using bits=4, group_size=128
# The second layer is quantized using bits=8, group_size=32
# All other layers (layer index >= 2) are not quantized
MODEL_QUANT = ["ModelCloud/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bits-dynamic-cfg"]


@pytest.mark.parametrize("model_id", MODEL_QUANT)
def test_gptq_with_dynamic_cfg(vllm_runner, model_id: str):
    vllm_model = vllm_runner(model_id, dtype=torch.float16, max_model_len=2048)

    for name, submodule in (vllm_model.model.llm_engine.model_executor.
                            driver_worker.model_runner.model.named_modules()):
        if name == 'model.model.layers.0.self_attn.qkv_proj':
            # The first layer is quantized using bits=4, group_size=128
            # desc_act=True
            assert isinstance(submodule, GPTQMarlinLinearMethod)
            assert submodule.quant_config.bits == 4
            assert submodule.quant_config.group_size == 128
            assert submodule.quant_config.desc_act
        elif name == 'model.model.layers.1.self_attn.qkv_proj':
            # The second layer is quantized using bits=8, group_size=32
            # desc_act=False
            assert isinstance(submodule, GPTQMarlinLinearMethod)
            assert submodule.quant_config.bits == 8
            assert submodule.quant_config.group_size == 32
            assert not submodule.quant_config.desc_act
        elif (name == 'model.model.layers.2.self_attn.qkv_proj'
              or name == 'model.model.layers.2.mlp.gate_up_proj'):
            # All other layers (layer index >= 2) are not quantized
            assert isinstance(submodule, UnquantizedLinearMethod)

    print(vllm_model.generate_greedy(prompts=[PROMPT], max_tokens=10)[0][1])
    del vllm_model
