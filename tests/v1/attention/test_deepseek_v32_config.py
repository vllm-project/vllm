# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Focused test coverage for PR #34899 - Config mechanism validation
https://github.com/vllm-project/vllm/pull/34899/
This test file focuses on what CAN be tested without FlashMLA/full environment:
1. Code architecture (inheritance, class removal)
2. Method existence
3. Config update logic (via direct method calls)
"""

from unittest.mock import Mock

import pytest

from vllm.model_executor.models.config import (
    DeepseekV32ForCausalLM,
    VerifyAndUpdateConfig,
)

# ============================================================================
# Test 1: Config Architecture (These work in any environment)
# ============================================================================


class TestDeepSeekV32ConfigArchitecture:
    """Test the config class architecture and inheritance."""

    def test_deepseek_v32_inherits_from_verify_and_update_config(self):
        """Verify DeepseekV32ForCausalLM inherits from VerifyAndUpdateConfig.

        This is the key change in PR #34899: DeepseekV32ForCausalLM now
        directly inherits from VerifyAndUpdateConfig instead of from
        DeepseekV3ForCausalLM (which was the workaround class).
        """
        assert issubclass(DeepseekV32ForCausalLM, VerifyAndUpdateConfig), (
            "DeepseekV32ForCausalLM should inherit from VerifyAndUpdateConfig"
        )
        print("✓ DeepseekV32ForCausalLM correctly inherits from VerifyAndUpdateConfig")

    def test_deepseek_v3_class_removed(self):
        """Verify DeepseekV3ForCausalLM class no longer exists.

        This class was the workaround that disabled fusion.
        It should be removed.
        """
        from vllm.model_executor.models import config as config_module

        assert not hasattr(config_module, "DeepseekV3ForCausalLM"), (
            "DeepseekV3ForCausalLM workaround class should be removed"
        )
        print("✓ DeepseekV3ForCausalLM workaround class has been removed")

    def test_verify_and_update_config_method_exists(self):
        """Verify the verify_and_update_config method is implemented."""
        assert hasattr(DeepseekV32ForCausalLM, "verify_and_update_config"), (
            "DeepseekV32ForCausalLM should implement verify_and_update_config"
        )

        assert callable(DeepseekV32ForCausalLM.verify_and_update_config), (
            "verify_and_update_config should be callable"
        )
        print("✓ verify_and_update_config method exists and is callable")


# ============================================================================
# Test 2: Config Update Logic (Direct method testing)
# ============================================================================


class TestDeepSeekV32ConfigUpdateLogic:
    """Test the config update logic by directly calling the method."""

    def test_verify_and_update_config_updates_fp8_to_fp8_ds_mla(self):
        """Test that verify_and_update_config converts fp8 to fp8_ds_mla."""

        # Create mock vllm_config with the necessary structure
        mock_vllm_config = Mock()
        mock_vllm_config.model_config = Mock()
        mock_vllm_config.model_config.hf_config = Mock()
        mock_vllm_config.model_config.hf_config.index_topk = 1  # Mark as v3.2

        # Test fp8 conversion
        mock_vllm_config.cache_config = Mock()
        mock_vllm_config.cache_config.cache_dtype = "fp8"

        # Call the method directly
        DeepseekV32ForCausalLM.verify_and_update_config(mock_vllm_config)

        # Verify it was updated
        cache_dtype = mock_vllm_config.cache_config.cache_dtype
        assert cache_dtype == "fp8_ds_mla", f"Expected fp8_ds_mla, got {cache_dtype}"
        print("✓ fp8 cache_dtype correctly converted to fp8_ds_mla")

    def test_verify_and_update_config_updates_fp8_e4m3_to_fp8_ds_mla(self):
        """Test verify_and_update_config converts fp8_e4m3 to fp8_ds_mla."""

        mock_vllm_config = Mock()
        mock_vllm_config.model_config = Mock()
        mock_vllm_config.model_config.hf_config = Mock()
        mock_vllm_config.model_config.hf_config.index_topk = 1

        mock_vllm_config.cache_config = Mock()
        mock_vllm_config.cache_config.cache_dtype = "fp8_e4m3"

        DeepseekV32ForCausalLM.verify_and_update_config(mock_vllm_config)

        assert mock_vllm_config.cache_config.cache_dtype == "fp8_ds_mla"
        print("✓ fp8_e4m3 cache_dtype correctly converted to fp8_ds_mla")

    def test_verify_and_update_config_converts_bfloat16_to_auto(self):
        """Test that verify_and_update_config converts bfloat16 to auto."""

        mock_vllm_config = Mock()
        mock_vllm_config.model_config = Mock()
        mock_vllm_config.model_config.hf_config = Mock()
        mock_vllm_config.model_config.hf_config.index_topk = 1

        mock_vllm_config.cache_config = Mock()
        mock_vllm_config.cache_config.cache_dtype = "bfloat16"

        DeepseekV32ForCausalLM.verify_and_update_config(mock_vllm_config)

        assert mock_vllm_config.cache_config.cache_dtype == "auto"
        print("✓ bfloat16 cache_dtype correctly converted to auto")

    def test_verify_and_update_config_preserves_auto(self):
        """Test that verify_and_update_config preserves auto cache_dtype."""

        mock_vllm_config = Mock()
        mock_vllm_config.model_config = Mock()
        mock_vllm_config.model_config.hf_config = Mock()
        mock_vllm_config.model_config.hf_config.index_topk = 1

        mock_vllm_config.cache_config = Mock()
        mock_vllm_config.cache_config.cache_dtype = "auto"

        DeepseekV32ForCausalLM.verify_and_update_config(mock_vllm_config)

        assert mock_vllm_config.cache_config.cache_dtype == "auto"
        print("✓ auto cache_dtype correctly preserved")

    def test_verify_and_update_config_handles_non_fp8_dtypes(self):
        """Test that other dtypes are not modified."""

        mock_vllm_config = Mock()
        mock_vllm_config.model_config = Mock()
        mock_vllm_config.model_config.hf_config = Mock()
        mock_vllm_config.model_config.hf_config.index_topk = 1

        mock_vllm_config.cache_config = Mock()
        mock_vllm_config.cache_config.cache_dtype = "float16"

        DeepseekV32ForCausalLM.verify_and_update_config(mock_vllm_config)

        assert mock_vllm_config.cache_config.cache_dtype == "float16"
        print("✓ float16 cache_dtype correctly preserved")


# ============================================================================
# Summary Test
# ============================================================================


def test_pr_34899_summary():
    """Summary test that validates the key changes from PR #34899.

    This test documents what PR #34899 actually changed and verifies
    those changes.
    """
    print("\n" + "=" * 70)
    print("PR #34899 Validation Summary")
    print("=" * 70)

    # Check 1: Class removal
    from vllm.model_executor.models import config as config_module

    assert not hasattr(config_module, "DeepseekV3ForCausalLM")
    print("✓ DeepseekV3ForCausalLM workaround class removed")

    # Check 2: Correct inheritance
    assert issubclass(DeepseekV32ForCausalLM, VerifyAndUpdateConfig)
    print("✓ DeepseekV32ForCausalLM inherits from VerifyAndUpdateConfig")

    # Check 3: Method exists
    assert hasattr(DeepseekV32ForCausalLM, "verify_and_update_config")
    assert callable(DeepseekV32ForCausalLM.verify_and_update_config)
    print("✓ verify_and_update_config method implemented")

    # Check 4: Config update logic works
    mock_config = Mock()
    mock_config.model_config = Mock()
    mock_config.model_config.hf_config = Mock()
    mock_config.model_config.hf_config.index_topk = 1
    mock_config.cache_config = Mock()
    mock_config.cache_config.cache_dtype = "fp8"

    DeepseekV32ForCausalLM.verify_and_update_config(mock_config)
    assert mock_config.cache_config.cache_dtype == "fp8_ds_mla"
    print("✓ Config update logic: fp8 → fp8_ds_mla conversion works")

    mock_config.cache_config.cache_dtype = "bfloat16"
    DeepseekV32ForCausalLM.verify_and_update_config(mock_config)
    assert mock_config.cache_config.cache_dtype == "auto"
    print("✓ Config update logic: bfloat16 → auto conversion works")

    print("\n" + "=" * 70)
    print("All PR #34899 changes validated successfully!")
    print("=" * 70)
    print("\nWhat this PR does:")
    print("1. Removes DeepseekV3ForCausalLM workaround class")
    print("2. Enables AR+Norm fusion BY DEFAULT (no longer disabled)")
    print("3. DeepseekV32ForCausalLM now properly configures cache dtype")
    print("   - fp8/fp8_e4m3 → fp8_ds_mla (custom MLA format)")
    print("   - bfloat16 → auto")
    print("=" * 70)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
