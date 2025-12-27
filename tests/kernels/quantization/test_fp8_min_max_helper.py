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


class TestGetFp8MinMax:
    """Test cases for get_fp8_min_max() function."""

    def test_standard_fp8_dtype(self):
        """Test that standard FP8 dtype uses PyTorch's finfo values."""
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            get_fp8_min_max,
        )

        # For standard float8_e4m3fn, should return finfo values
        fp8_min, fp8_max = get_fp8_min_max(torch.float8_e4m3fn)
        finfo = torch.finfo(torch.float8_e4m3fn)

        # Standard FP8 max is 448.0 for e4m3fn
        assert fp8_max == finfo.max, f"Expected finfo.max={finfo.max}, got {fp8_max}"
        assert fp8_min == finfo.min, f"Expected finfo.min={finfo.min}, got {fp8_min}"

    @patch("vllm.model_executor.layers.quantization.utils.quant_utils.current_platform")
    def test_fnuz_fp8_dtype_on_fnuz_platform(self, mock_platform):
        """Test that fnuz dtype on fnuz platform returns 224.0."""
        mock_platform.is_fp8_fnuz.return_value = True
        mock_platform.fp8_dtype.return_value = torch.float8_e4m3fnuz

        # Re-import to use mocked platform
        from importlib import reload

        import vllm.model_executor.layers.quantization.utils.quant_utils as qu

        reload(qu)

        fp8_min, fp8_max = qu.get_fp8_min_max(torch.float8_e4m3fnuz)

        # fnuz on ROCm MI300 should return 224.0, not 240.0
        assert fp8_max == 224.0, (
            f"Expected 224.0 for fnuz on fnuz platform, got {fp8_max}"
        )
        assert fp8_min == -224.0, (
            f"Expected -224.0 for fnuz on fnuz platform, got {fp8_min}"
        )

    @patch("vllm.model_executor.layers.quantization.utils.quant_utils.current_platform")
    def test_standard_dtype_on_fnuz_platform(self, mock_platform):
        """Test that standard dtype on fnuz platform uses finfo values."""
        mock_platform.is_fp8_fnuz.return_value = True

        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            get_fp8_min_max,
        )

        # Standard e4m3fn dtype should use finfo even on fnuz platform
        fp8_min, fp8_max = get_fp8_min_max(torch.float8_e4m3fn)
        finfo = torch.finfo(torch.float8_e4m3fn)

        assert fp8_max == finfo.max, (
            f"Standard dtype should use finfo.max={finfo.max}, got {fp8_max}"
        )

    @patch("vllm.model_executor.layers.quantization.utils.quant_utils.current_platform")
    def test_fnuz_dtype_on_non_fnuz_platform(self, mock_platform):
        """Test that fnuz dtype on non-fnuz platform uses finfo values."""
        mock_platform.is_fp8_fnuz.return_value = False

        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            get_fp8_min_max,
        )

        # fnuz dtype on non-fnuz platform should use finfo
        fp8_min, fp8_max = get_fp8_min_max(torch.float8_e4m3fnuz)
        finfo = torch.finfo(torch.float8_e4m3fnuz)

        # Should be 240.0, not 224.0 (non-fnuz platform)
        assert fp8_max == finfo.max, (
            f"Non-fnuz platform should use finfo.max={finfo.max}, got {fp8_max}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
