# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test that INT8 GPTQ MoE layers fall back to MoeWNA16 kernels.

MarlinExperts does not support INT8 MoE weights. When a GPTQ model has
8-bit MoE expert layers (e.g., mixed INT4/INT8 models), the quantization
dispatch must route those layers to MoeWNA16Method which handles INT8 via
Triton fused_experts kernels.

See: https://github.com/vllm-project/vllm/issues/41955
"""

from copy import deepcopy
from unittest.mock import MagicMock, patch

import pytest

from vllm.model_executor.layers.quantization.gptq_marlin import (
    GPTQMarlinConfig,
    get_moe_quant_method,
)
from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Method
from vllm.platforms import current_platform


def _make_gptq_config(weight_bits: int = 4, group_size: int = 128) -> GPTQMarlinConfig:
    """Create a GPTQMarlinConfig for testing."""
    full_config = {
        "quant_method": "gptq",
        "bits": weight_bits,
        "group_size": group_size,
        "sym": True,
        "desc_act": False,
        "lm_head": False,
        "dynamic": {},
    }
    return GPTQMarlinConfig(
        weight_bits=weight_bits,
        group_size=group_size,
        desc_act=False,
        is_sym=True,
        lm_head_quantized=False,
        dynamic={},
        full_config=full_config,
    )


def _make_gptq_config_with_dynamic(
    base_bits: int = 4, override_bits: int = 8, group_size: int = 128
) -> GPTQMarlinConfig:
    """Create a GPTQMarlinConfig with dynamic per-layer overrides."""
    dynamic = {
        r"+:.*\.moe\..*": {"bits": override_bits},
    }
    full_config = {
        "quant_method": "gptq",
        "bits": base_bits,
        "group_size": group_size,
        "sym": True,
        "desc_act": False,
        "lm_head": False,
        "dynamic": dynamic,
    }
    return GPTQMarlinConfig(
        weight_bits=base_bits,
        group_size=group_size,
        desc_act=False,
        is_sym=True,
        lm_head_quantized=False,
        dynamic=dynamic,
        full_config=full_config,
    )


def _make_mock_fused_moe_layer():
    """Create a mock FusedMoE layer with valid shapes for Marlin."""
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    mock_layer = MagicMock(spec=FusedMoE)
    mock_layer.hidden_size = 4096  # divisible by 128
    mock_layer.intermediate_size_per_partition = 11008  # typical MoE size
    mock_layer.apply_router_weight_on_input = False
    mock_layer.moe_config = MagicMock()
    return mock_layer


@pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="GPTQ Marlin requires CUDA.",
)
class TestInt8MoEFallback:
    """Test that INT8 MoE layers are routed to MoeWNA16 kernels."""

    def test_int8_base_config_falls_back_to_wna16(self):
        """When base weight_bits=8, MoE layers should use MoeWNA16Method."""
        config = _make_gptq_config(weight_bits=8)
        layer = _make_mock_fused_moe_layer()

        from vllm.model_executor.layers.quantization.gptq_marlin import (
            GPTQMarlinMoEMethod,
        )

        result = get_moe_quant_method(
            config, layer, "model.layers.0.moe.experts", GPTQMarlinMoEMethod
        )

        assert isinstance(result, MoeWNA16Method), (
            f"Expected MoeWNA16Method for INT8 MoE, got {type(result).__name__}"
        )

    def test_int4_base_config_uses_marlin(self):
        """When base weight_bits=4, MoE layers should use GPTQMarlinMoEMethod."""
        config = _make_gptq_config(weight_bits=4)
        layer = _make_mock_fused_moe_layer()

        from vllm.model_executor.layers.quantization.gptq_marlin import (
            GPTQMarlinMoEMethod,
        )

        result = get_moe_quant_method(
            config, layer, "model.layers.0.moe.experts", GPTQMarlinMoEMethod
        )

        assert isinstance(result, GPTQMarlinMoEMethod), (
            f"Expected GPTQMarlinMoEMethod for INT4 MoE, got {type(result).__name__}"
        )

    def test_dynamic_override_to_int8_falls_back(self):
        """When dynamic override sets a MoE layer to 8-bit, it should
        fall back to MoeWNA16Method."""
        config = _make_gptq_config_with_dynamic(base_bits=4, override_bits=8)
        layer = _make_mock_fused_moe_layer()

        from vllm.model_executor.layers.quantization.gptq_marlin import (
            GPTQMarlinMoEMethod,
        )

        # The prefix matches the dynamic rule "+:.*\.moe\..*"
        result = get_moe_quant_method(
            config, layer, "model.layers.5.moe.experts", GPTQMarlinMoEMethod
        )

        assert isinstance(result, MoeWNA16Method), (
            f"Expected MoeWNA16Method for dynamically overridden INT8 MoE, "
            f"got {type(result).__name__}"
        )

    def test_dynamic_override_non_matching_stays_int4(self):
        """When dynamic override exists but doesn't match the prefix,
        INT4 base config should use GPTQMarlinMoEMethod."""
        config = _make_gptq_config_with_dynamic(base_bits=4, override_bits=8)
        layer = _make_mock_fused_moe_layer()

        from vllm.model_executor.layers.quantization.gptq_marlin import (
            GPTQMarlinMoEMethod,
        )

        # This prefix does NOT match "+:.*\.moe\..*"
        result = get_moe_quant_method(
            config, layer, "model.layers.5.mlp.experts", GPTQMarlinMoEMethod
        )

        assert isinstance(result, GPTQMarlinMoEMethod), (
            f"Expected GPTQMarlinMoEMethod for non-matching INT4 MoE, "
            f"got {type(result).__name__}"
        )

    def test_int8_fallback_propagates_config_values(self):
        """Verify that group_size and other config fields propagate
        correctly through the INT8 fallback path."""
        config = _make_gptq_config(weight_bits=8, group_size=64)
        layer = _make_mock_fused_moe_layer()

        from vllm.model_executor.layers.quantization.gptq_marlin import (
            GPTQMarlinMoEMethod,
        )

        result = get_moe_quant_method(
            config, layer, "model.layers.0.moe.experts", GPTQMarlinMoEMethod
        )

        assert isinstance(result, MoeWNA16Method)
        assert result.quant_config.weight_bits == 8
        assert result.quant_config.group_size == 64

    def test_int8_with_desc_act_raises(self):
        """INT8 MoE with desc_act=True should raise ValueError."""
        full_config = {
            "quant_method": "gptq",
            "bits": 8,
            "group_size": 128,
            "sym": True,
            "desc_act": True,
            "lm_head": False,
            "dynamic": {},
        }
        config = GPTQMarlinConfig(
            weight_bits=8,
            group_size=128,
            desc_act=True,
            is_sym=True,
            lm_head_quantized=False,
            dynamic={},
            full_config=full_config,
        )
        layer = _make_mock_fused_moe_layer()

        from vllm.model_executor.layers.quantization.gptq_marlin import (
            GPTQMarlinMoEMethod,
        )

        with pytest.raises(ValueError, match="desc_act=True.*not supported"):
            get_moe_quant_method(
                config, layer, "model.layers.0.moe.experts", GPTQMarlinMoEMethod
            )
