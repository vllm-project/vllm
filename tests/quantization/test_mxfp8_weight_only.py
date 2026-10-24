# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for MXFP8 MoE weight-only checkpoint handling.

When a compressed-tensors checkpoint declares MXFP8 weights but sets
input_activations to null (weight-only / W8A16), the MoE dispatch path
must respect that declaration instead of unconditionally applying W8A8
dynamic activation quantization.

Run: pytest tests/quantization/test_mxfp8_weight_only.py -v
"""
from types import SimpleNamespace

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
)

MXFP8_WEIGHT_CONFIG = {
    "num_bits": 8,
    "type": "float",
    "strategy": "group",
    "group_size": 32,
    "symmetric": True,
    "dynamic": False,
    "scale_dtype": "uint8",
}


def _make_mxfp8_moe_layer(monkeypatch, input_activations):
    """Build a RoutedExperts with MXFP8 weights and the given activations."""
    from tests.kernels.moe.utils import make_dummy_moe_config
    from vllm.model_executor.layers.fused_moe.oracle import mxfp8
    from vllm.model_executor.layers.fused_moe.routed_experts import (
        RoutedExperts,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
        compressed_tensors_moe_w8a8_mxfp8 as ct_mxfp8_mod,
    )

    stub = lambda config, use_a16=False: (SimpleNamespace(value="stub"), None)
    monkeypatch.setattr(mxfp8, "select_mxfp8_moe_backend", stub)
    monkeypatch.setattr(ct_mxfp8_mod, "select_mxfp8_moe_backend", stub)

    quant_config = CompressedTensorsConfig.from_config(
        {
            "format": "dense",
            "config_groups": {
                "group_0": {
                    "targets": [r"re:.*experts\.\d+\.w\d+$"],
                    "weights": MXFP8_WEIGHT_CONFIG,
                    "input_activations": input_activations,
                }
            },
        }
    )
    return RoutedExperts(
        "model.layers.0.ffn.experts",
        torch.bfloat16,
        make_dummy_moe_config(
            num_experts=2, hidden_dim=64, intermediate_size=32
        ),
        quant_config,
        expert_map_manager=SimpleNamespace(
            local_num_experts=2,
            placement_strategy="linear",
            expert_map=None,
            expert_mask=None,
            routing_tables=None,
            map_global_to_local=lambda index: index,
        ),
        ckpt_gate_proj_name="w1",
        ckpt_down_proj_name="w2",
        ckpt_up_proj_name="w3",
    )


def test_moe_weight_only_mxfp8_sets_use_a16(monkeypatch):
    """Weight-only MXFP8 (input_activations=null) must set use_a16=True."""
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_mxfp8 import (  # noqa: E501
        CompressedTensorsW8A8Mxfp8MoEMethod,
    )

    with set_current_vllm_config(VllmConfig()):
        layer = _make_mxfp8_moe_layer(monkeypatch, input_activations=None)

    assert isinstance(layer.quant_method, CompressedTensorsW8A8Mxfp8MoEMethod)
    assert layer.quant_method.use_a16


def test_moe_w8a8_mxfp8_does_not_set_use_a16(monkeypatch):
    """Explicit MXFP8 activations should set use_a16=False."""
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_mxfp8 import (  # noqa: E501
        CompressedTensorsW8A8Mxfp8MoEMethod,
    )

    mxfp8_input = {
        "num_bits": 8,
        "type": "float",
        "strategy": "group",
        "group_size": 32,
        "symmetric": True,
        "dynamic": True,
        "scale_dtype": "uint8",
    }
    with set_current_vllm_config(VllmConfig()):
        layer = _make_mxfp8_moe_layer(
            monkeypatch, input_activations=mxfp8_input
        )

    assert isinstance(layer.quant_method, CompressedTensorsW8A8Mxfp8MoEMethod)
    assert not layer.quant_method.use_a16


def test_moe_mxfp8_weights_with_invalid_input_raises(monkeypatch):
    """MXFP8 weights with non-MXFP8 input quant should raise."""
    fp8_input = {
        "num_bits": 8,
        "type": "float",
        "strategy": "token",
        "symmetric": True,
        "dynamic": True,
    }
    with set_current_vllm_config(VllmConfig()):
        with pytest.raises(ValueError, match="MXFP8 format or None"):
            _make_mxfp8_moe_layer(monkeypatch, input_activations=fp8_input)
