# SPDX-License-Identifier: Apache-2.0
"""Unit tests for DeepSeek-V4.1 quant method routing and shared experts exclusion."""

from types import SimpleNamespace
import pytest
import torch

from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from vllm.model_executor.layers.quantization.modelopt import ModelOptLinearMethod
from vllm.models.deepseek_v41.quant_config import DeepseekV4FP8Config


class DummyLinear(LinearBase):
    """Dummy linear layer for testing quant method resolution."""

    def __init__(self):
        super().__init__(input_size=128, output_size=128, tp_rank=0, tp_size=1)


@pytest.fixture
def vllm_config_context():
    """Mock vllm_config context required for quant method instantiation."""
    mock_config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16, is_multimodal_model=False),
        kernel_config=SimpleNamespace(linear_backend="auto"),
    )
    with set_current_vllm_config(mock_config):
        yield


def test_shared_experts_excluded_from_modelopt_linear_method(vllm_config_context):
    """Verify shared_experts falls back to Fp8LinearMethod under [32, 32] block size."""
    cfg = DeepseekV4FP8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[32, 32],
    )
    # Set resolved expert dtype to fp4 to trigger is_scale_e8m0 = True
    cfg._resolved_expert_dtype = "fp4"

    layer = DummyLinear()
    method_gate = cfg.get_quant_method(layer, prefix="model.layers.0.mlp.shared_experts.gate_proj")
    method_down = cfg.get_quant_method(layer, prefix="model.layers.0.mlp.shared_experts.down_proj")

    # Shared experts must NOT use ModelOptLinearMethod
    assert not isinstance(method_gate, ModelOptLinearMethod)
    assert isinstance(method_gate, Fp8LinearMethod)

    assert not isinstance(method_down, ModelOptLinearMethod)
    assert isinstance(method_down, Fp8LinearMethod)


def test_dense_and_attn_linear_use_modelopt_linear_method(vllm_config_context):
    """Verify attention and dense linear layers use ModelOptLinearMethod under [32, 32]."""
    cfg = DeepseekV4FP8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[32, 32],
    )
    cfg._resolved_expert_dtype = "fp4"

    layer = DummyLinear()
    method_q = cfg.get_quant_method(layer, prefix="model.layers.0.self_attn.q_proj")
    method_o = cfg.get_quant_method(layer, prefix="model.layers.0.self_attn.o_proj")

    assert isinstance(method_q, ModelOptLinearMethod)
    assert isinstance(method_o, ModelOptLinearMethod)


def test_quant_method_fallback_for_128x128_block_size(vllm_config_context):
    """Verify fallback to standard Fp8LinearMethod for standard [128, 128] block size."""
    cfg = DeepseekV4FP8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[128, 128],
    )
    cfg._resolved_expert_dtype = "fp4"

    layer = DummyLinear()
    method_attn = cfg.get_quant_method(layer, prefix="model.layers.0.self_attn.q_proj")
    method_shared = cfg.get_quant_method(layer, prefix="model.layers.0.mlp.shared_experts.gate_proj")

    # When block size is not [32, 32], standard Fp8LinearMethod is used for both
    assert isinstance(method_attn, Fp8LinearMethod)
    assert isinstance(method_shared, Fp8LinearMethod)


def test_deepseek_v4_quark_block_size_discovery():
    """Verify deepseek_v4 dynamic block size extraction from Quark configs."""
    from vllm.models.deepseek_v4.quant_config import (
        DeepseekV4FP8Config as DeepseekV4CoreFP8Config,
    )

    # 1. Top-level weight_block_size
    cfg1 = DeepseekV4CoreFP8Config.from_config({
        "quant_method": "quark",
        "weight_block_size": [32, 32],
    })
    assert cfg1.weight_block_size == [32, 32]

    # 2. Nested in layer_quant_config
    cfg2 = DeepseekV4CoreFP8Config.from_config({
        "quant_method": "quark",
        "layer_quant_config": {
            "model.layers.0.self_attn.q_proj": {
                "weight": {"block_size": [32, 32]}
            }
        },
    })
    assert cfg2.weight_block_size == [32, 32]

    # 3. Default fallback to [128, 128]
    cfg3 = DeepseekV4CoreFP8Config.from_config({
        "quant_method": "quark",
    })
    assert cfg3.weight_block_size == [128, 128]

