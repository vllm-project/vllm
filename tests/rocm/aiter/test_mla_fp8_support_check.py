# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for AITER MLA FP8 support detection.

These tests verify that the _check_aiter_mla_fp8_support() function
correctly handles various error conditions without crashing.
"""

from unittest.mock import patch

import pytest


class TestAiterMlaFp8SupportCheck:
    """Test cases for _check_aiter_mla_fp8_support() function."""

    def setup_method(self):
        """Reset the global cache before each test."""
        import vllm._aiter_ops as aiter_ops

        aiter_ops._AITER_MLA_SUPPORTS_FP8 = None

    @patch("vllm._aiter_ops.is_aiter_found_and_supported", return_value=True)
    def test_import_error_handling(self, mock_supported):
        """Test that ImportError is handled gracefully."""
        import vllm._aiter_ops as aiter_ops
        from vllm._aiter_ops import _check_aiter_mla_fp8_support

        aiter_ops._AITER_MLA_SUPPORTS_FP8 = None

        # Should return False without raising
        with patch(
            "vllm._aiter_ops.inspect.signature",
            side_effect=ImportError("No module"),
        ):
            result = _check_aiter_mla_fp8_support()
            assert result is False

    @patch("vllm._aiter_ops.is_aiter_found_and_supported", return_value=True)
    def test_module_not_found_error_handling(self, mock_supported):
        """Test that ModuleNotFoundError is handled gracefully."""
        import vllm._aiter_ops as aiter_ops
        from vllm._aiter_ops import _check_aiter_mla_fp8_support

        aiter_ops._AITER_MLA_SUPPORTS_FP8 = None

        with patch(
            "vllm._aiter_ops.inspect.signature",
            side_effect=ModuleNotFoundError("Module not found"),
        ):
            # Should return False without raising
            assert _check_aiter_mla_fp8_support() is False
            # Cache should be set to False
            assert aiter_ops._AITER_MLA_SUPPORTS_FP8 is False

    @patch("vllm._aiter_ops.is_aiter_found_and_supported", return_value=True)
    def test_attribute_error_handling(self, mock_supported):
        """Test that AttributeError is handled gracefully."""
        import vllm._aiter_ops as aiter_ops
        from vllm._aiter_ops import _check_aiter_mla_fp8_support

        aiter_ops._AITER_MLA_SUPPORTS_FP8 = None

        with patch(
            "vllm._aiter_ops.inspect.signature",
            side_effect=AttributeError("No attribute"),
        ):
            assert _check_aiter_mla_fp8_support() is False
            assert aiter_ops._AITER_MLA_SUPPORTS_FP8 is False

    @patch("vllm._aiter_ops.is_aiter_found_and_supported", return_value=True)
    def test_value_error_handling(self, mock_supported):
        """Test that ValueError is handled gracefully (no signature)."""
        import vllm._aiter_ops as aiter_ops
        from vllm._aiter_ops import _check_aiter_mla_fp8_support

        aiter_ops._AITER_MLA_SUPPORTS_FP8 = None

        with patch(
            "vllm._aiter_ops.inspect.signature",
            side_effect=ValueError("No signature"),
        ):
            assert _check_aiter_mla_fp8_support() is False
            assert aiter_ops._AITER_MLA_SUPPORTS_FP8 is False

    @patch("vllm._aiter_ops.is_aiter_found_and_supported", return_value=True)
    def test_type_error_handling(self, mock_supported):
        """Test that TypeError is handled gracefully (not callable)."""
        import vllm._aiter_ops as aiter_ops
        from vllm._aiter_ops import _check_aiter_mla_fp8_support

        aiter_ops._AITER_MLA_SUPPORTS_FP8 = None

        with patch(
            "vllm._aiter_ops.inspect.signature",
            side_effect=TypeError("Not a callable"),
        ):
            assert _check_aiter_mla_fp8_support() is False
            assert aiter_ops._AITER_MLA_SUPPORTS_FP8 is False

    @patch("vllm._aiter_ops.is_aiter_found_and_supported", return_value=True)
    def test_result_caching(self, mock_supported):
        """Test that the result is cached after first check."""
        import vllm._aiter_ops as aiter_ops

        # Set cache to True
        aiter_ops._AITER_MLA_SUPPORTS_FP8 = True

        from vllm._aiter_ops import _check_aiter_mla_fp8_support

        # Should return cached value without re-checking
        result = _check_aiter_mla_fp8_support()
        assert result is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
