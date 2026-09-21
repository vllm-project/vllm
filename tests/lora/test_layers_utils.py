# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from torch import nn

from vllm.lora.layers.utils import _get_lora_device

pytestmark = pytest.mark.skip_global_cleanup


def _param() -> nn.Parameter:
    return nn.Parameter(torch.empty(1), requires_grad=False)


def test_get_lora_device_unquantized():
    base_layer = nn.Module()
    base_layer.weight = _param()
    assert _get_lora_device(base_layer) == base_layer.weight.device


def test_get_lora_device_gptq_awq():
    base_layer = nn.Module()
    base_layer.qweight = _param()
    assert _get_lora_device(base_layer) == base_layer.qweight.device


def test_get_lora_device_ark_linear():
    base_layer = nn.Module()
    base_layer.ark_linear = nn.Module()
    base_layer.ark_linear.qweight = _param()
    assert _get_lora_device(base_layer) == base_layer.ark_linear.qweight.device


def test_get_lora_device_unsupported_raises():
    base_layer = nn.Module()
    with pytest.raises(ValueError, match="Unsupported base layer"):
        _get_lora_device(base_layer)
