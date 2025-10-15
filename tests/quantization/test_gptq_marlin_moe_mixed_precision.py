# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy
from unittest.mock import MagicMock

import pytest
import torch

from vllm.model_executor.layers.quantization.gptq_marlin import (
    GPTQMarlinConfig,
    GPTQMarlinMoEMethod,
)


def _make_marlin_config() -> GPTQMarlinConfig:
    base_config = {
        "bits": 4,
        "group_size": 128,
        "desc_act": False,
        "sym": True,
        "dynamic": {
            "+:model[.]layers[.]([0-2])[.].*": {"bits": 8},
        },
    }
    return GPTQMarlinConfig.from_config(base_config)


def _with_default_cuda_device() -> tuple[str, bool]:
    """Set default device to CUDA:0 if available, returning the previous device."""
    if not torch.cuda.is_available():
        return "", False
    previous = torch.get_default_device()
    torch.set_default_device("cuda:0")
    return previous, True


class _DummyMoELayer(torch.nn.Module):
    pass


def test_gptq_marlin_moe_method_uses_dynamic_bits_override():
    config = _make_marlin_config()

    eight_bit_config = deepcopy(config)
    setattr(eight_bit_config, "_layer_prefix", "model.layers.1.mlp.experts")
    eight_bit_method = GPTQMarlinMoEMethod(eight_bit_config, MagicMock())

    assert eight_bit_method.quant_metadata["bits"] == 8
    assert eight_bit_method.quant_config.pack_factor == 4
    assert eight_bit_method.quant_type.size_bits == 8

    layer = _DummyMoELayer()
    previous_device, has_cuda = _with_default_cuda_device()
    if not has_cuda:
        pytest.skip("CUDA device required for GPTQ Marlin MoE unit test")
    try:
        eight_bit_method.create_weights(
            layer=layer,
            num_experts=1,
            hidden_size=128,
            intermediate_size_per_partition=64,
            params_dtype=torch.float16,
            intermediate_size_full=64,
        )
    finally:
        torch.set_default_device(previous_device or "cpu")

    assert layer.w2_qweight.shape == (1, 16, 128)  # 64 // 4 = 16


def test_gptq_marlin_moe_method_retains_base_bits_without_override():
    config = _make_marlin_config()

    four_bit_config = deepcopy(config)
    setattr(four_bit_config, "_layer_prefix", "model.layers.5.mlp.experts")
    four_bit_method = GPTQMarlinMoEMethod(four_bit_config, MagicMock())

    assert four_bit_method.quant_metadata["bits"] == 4
    assert four_bit_method.quant_config.pack_factor == 8
    assert four_bit_method.quant_type.size_bits == 4

    layer = _DummyMoELayer()
    previous_device, has_cuda = _with_default_cuda_device()
    if not has_cuda:
        pytest.skip("CUDA device required for GPTQ Marlin MoE unit test")
    try:
        four_bit_method.create_weights(
            layer=layer,
            num_experts=1,
            hidden_size=128,
            intermediate_size_per_partition=64,
            params_dtype=torch.float16,
            intermediate_size_full=64,
        )
    finally:
        torch.set_default_device(previous_device or "cpu")

    assert layer.w2_qweight.shape == (1, 8, 128)  # 64 // 8 = 8
