# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for the get_fp8_min_max() helper function.

These tests verify the FP8 min/max value logic for both standard
and fnuz (ROCm MI300) dtype handling.
"""

from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)


class TestGetFp8MinMax:
    """Test cases for get_fp8_min_max() function."""

    @patch("vllm.model_executor.layers.quantization.utils.quant_utils.current_platform")
    def test_standard_fp8_platform(self, mock_platform):
        """Test that standard FP8 platform uses PyTorch's finfo values."""
        mock_platform.is_fp8_fnuz.return_value = False
        mock_platform.fp8_dtype.return_value = torch.float8_e4m3fn

        fp8_min, fp8_max = get_fp8_min_max()
        finfo = torch.finfo(torch.float8_e4m3fn)

        # Standard FP8 max is 448.0 for e4m3fn
        assert fp8_max == finfo.max, f"Expected finfo.max={finfo.max}, got {fp8_max}"
        assert fp8_min == finfo.min, f"Expected finfo.min={finfo.min}, got {fp8_min}"

    @patch("vllm.model_executor.layers.quantization.utils.quant_utils.current_platform")
    def test_fnuz_platform_returns_224(self, mock_platform):
        """Test that fnuz platform returns 224.0."""
        mock_platform.is_fp8_fnuz.return_value = True

        fp8_min, fp8_max = get_fp8_min_max()

        # fnuz on ROCm MI300 should return 224.0, not 240.0
        assert fp8_max == 224.0, f"Expected 224.0 for fnuz platform, got {fp8_max}"
        assert fp8_min == -224.0, f"Expected -224.0 for fnuz platform, got {fp8_min}"

    @patch("vllm.model_executor.layers.quantization.utils.quant_utils.current_platform")
    def test_non_fnuz_platform_uses_finfo(self, mock_platform):
        """Test that non-fnuz platform uses finfo values."""
        mock_platform.is_fp8_fnuz.return_value = False
        mock_platform.fp8_dtype.return_value = torch.float8_e4m3fn

        fp8_min, fp8_max = get_fp8_min_max()
        finfo = torch.finfo(torch.float8_e4m3fn)

        assert (
            fp8_max == finfo.max
        ), f"Non-fnuz platform should use finfo.max={finfo.max}, got {fp8_max}"
        assert (
            fp8_min == finfo.min
        ), f"Non-fnuz platform should use finfo.min={finfo.min}, got {fp8_min}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
