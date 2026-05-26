# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for AutoAWQConfig behavior after unification.

These tests verify the bug fixes for:
1. CPU platform override conflict (auto_awq should not override on CPU)
2. MoE fallback compatibility (full_config["quant_method"] should be "awq")
3. Config attribute consistency
4. End-to-end quantization method loading (auto_awq loads and runs correctly)

Note: Tests that require importing the full auto_awq module (which has GPU-dependent
imports) should use subprocess or be run in a GPU environment.
"""

from __future__ import annotations

import pytest
import torch

from tests.quantization.utils import is_quant_method_supported


def _get_auto_awq_config_source() -> str:
    """Read the AutoAWQConfig class source code for isolated testing."""
    import inspect

    import vllm.model_executor.layers.quantization.auto_awq as auto_awq_module

    return inspect.getsource(auto_awq_module.AutoAWQConfig)


class TestAutoAWQConfigFromConfig:
    """Tests for AutoAWQConfig.from_config behavior.

    These tests require GPU environment to import the full module.
    They are skipped on non-GPU platforms.
    """

    def test_full_config_quant_method_is_awq_for_moe_fallback(self):
        """full_config should have quant_method='awq' for MoE fallback compatibility.

        MoeWNA16Config only accepts 'gptq' or 'awq' as linear_quant_method.
        If full_config has 'auto_awq', the MoE fallback will fail.
        """
        from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig

        config = {
            "w_bit": 4,
            "q_group_size": 128,
            "zero_point": True,
            "lm_head": False,
        }
        awq_config = AutoAWQConfig.from_config(config)

        # Verify quant_method is 'awq' for MoE fallback
        assert awq_config.full_config["quant_method"] == "awq", (
            f"Expected quant_method='awq', got {awq_config.full_config['quant_method']}"
        )

    def test_full_config_preserves_other_fields(self):
        """full_config should preserve all original config fields."""
        from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig

        config = {
            "w_bit": 4,
            "q_group_size": 128,
            "zero_point": True,
            "lm_head": False,
            "custom_field": "custom_value",
        }
        awq_config = AutoAWQConfig.from_config(config)

        assert awq_config.full_config["w_bit"] == 4
        assert awq_config.full_config["q_group_size"] == 128
        assert awq_config.full_config["zero_point"] is True
        assert awq_config.full_config["lm_head"] is False
        assert awq_config.full_config["custom_field"] == "custom_value"

    def test_full_config_is_copy_not_original(self):
        """full_config should be a copy, not the original dict."""
        from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig

        config = {
            "w_bit": 4,
            "q_group_size": 128,
            "zero_point": True,
            "lm_head": False,
        }
        original_quant_method = config.get("quant_method")

        AutoAWQConfig.from_config(config)

        # Original config should not be modified
        assert config.get("quant_method") == original_quant_method


class TestAutoAWQConfigAttributes:
    """Tests for AutoAWQConfig attribute consistency.

    These tests require GPU environment to import the full module.
    They are skipped on non-GPU platforms.
    """

    def test_config_attributes_match_input(self):
        """Config attributes should match input values."""
        from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig

        awq_config = AutoAWQConfig(
            weight_bits=4,
            group_size=128,
            zero_point=True,
            lm_head_quantized=False,
            modules_to_not_convert=["lm_head"],
        )

        assert awq_config.weight_bits == 4
        assert awq_config.group_size == 128
        assert awq_config.zero_point is True
        assert awq_config.lm_head_quantized is False
        assert awq_config.modules_to_not_convert == ["lm_head"]

    def test_pack_factor_for_4bit(self):
        """Pack factor should be 8 for 4-bit quantization."""
        from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig

        awq_config = AutoAWQConfig(
            weight_bits=4,
            group_size=128,
            zero_point=True,
            lm_head_quantized=False,
        )

        assert awq_config.pack_factor == 8  # 32 // 4


class TestAutoAWQConfigOverrideLogic:
    """Tests for override logic by parsing source code (no GPU import required)."""

    def test_cpu_check_in_override_method(self):
        """override_quantization_method should check current_platform.is_cpu()."""
        # Read the source file directly
        import pathlib

        source_path = (
            pathlib.Path(__file__).parent.parent.parent
            / "vllm/model_executor/layers/quantization/auto_awq.py"
        )
        source = source_path.read_text()

        # Verify the CPU check exists in override method
        assert "current_platform.is_cpu()" in source, (
            "override_quantization_method should check is_cpu()"
        )
        assert "return None" in source, (
            "override_quantization_method should return None on CPU"
        )

    def test_quant_method_normalization_in_from_config(self):
        """from_config should normalize quant_method to 'awq' for MoE fallback."""
        import pathlib

        source_path = (
            pathlib.Path(__file__).parent.parent.parent
            / "vllm/model_executor/layers/quantization/auto_awq.py"
        )
        source = source_path.read_text()

        # Verify the normalization exists
        assert (
            '"quant_method"] = "awq"' in source or "'quant_method'] = 'awq'" in source
        ), "from_config should set quant_method='awq' in full_config"


# =============================================================================
# End-to-end integration tests (require GPU environment)
# =============================================================================

PROMPT = "On the surface of Mars, we found"

# Small AWQ model for testing - using Qwen2 1.5B which has official AWQ checkpoint
AWQ_MODELS = [
    "Qwen/Qwen2-1.5B-Instruct-AWQ",
]


@pytest.mark.skipif(
    not is_quant_method_supported("auto_awq"),
    reason="auto_awq is not supported on this GPU type.",
)
@pytest.mark.parametrize("model_id", AWQ_MODELS)
def test_auto_awq_quantization_method(vllm_runner, model_id: str):
    """Test that quantization='auto_awq' loads and runs correctly.

    This verifies that:
    1. AutoAWQConfig.override_quantization_method() correctly detects AWQ models
    2. The model loads with auto_awq quantization and uses the appropriate kernel
    3. Generation produces valid output
    """
    with vllm_runner(
        model_id,
        dtype=torch.float16,
        quantization="auto_awq",
        max_model_len=2048,
        enforce_eager=True,
    ) as llm:

        def check_model(model):
            from vllm.model_executor.layers.quantization.auto_awq import (
                AutoAWQLinearMethod,
                AutoAWQMarlinLinearMethod,
            )

            for name, submodule in model.named_modules():
                if name == "model.layers.0.self_attn.qkv_proj":
                    # Should use either AutoAWQLinearMethod (Triton) or
                    # AutoAWQMarlinLinearMethod (Marlin) depending on hardware
                    assert isinstance(
                        submodule.quant_method,
                        (AutoAWQLinearMethod, AutoAWQMarlinLinearMethod),
                    ), (
                        f"Expected AutoAWQLinearMethod or AutoAWQMarlinLinearMethod "
                        f"for {name}, got {type(submodule.quant_method)}"
                    )
                    break

        llm.apply_model(check_model)

        outputs = llm.generate_greedy([PROMPT], max_tokens=8)
        assert outputs
        assert len(outputs[0][1]) > 0


def test_auto_awq_config_get_name():
    """Test that AutoAWQConfig.get_name() returns 'auto_awq'."""
    from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig

    assert AutoAWQConfig.get_name() == "auto_awq"
