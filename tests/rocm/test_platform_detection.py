# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for ROCm platform detection functions.

These tests mock torch.cuda.get_device_properties() to verify
platform detection logic without requiring actual hardware.
"""

from unittest.mock import patch

import pytest


class MockDeviceProperties:
    """Mock class for torch.cuda.get_device_properties() return value."""

    def __init__(self, gcn_arch_name: str):
        self.gcnArchName = gcn_arch_name


class TestRocmFP8Support:
    """Test cases for supports_fp8() function."""

    @pytest.mark.parametrize(
        "gcn_arch,expected",
        [
            # CDNA architectures (MI300/MI350 series) - should support FP8
            ("gfx942:sramecc+:xnack-", True),  # MI300X
            ("gfx940:sramecc+:xnack-", True),  # MI300A
            ("gfx950:sramecc+:xnack-", True),  # MI350
            # RDNA 3/3.5 architectures - should support FP8 (gfx11x)
            ("gfx1100", True),  # RDNA 3 (RX 7900 XTX)
            ("gfx1101", True),  # RDNA 3
            ("gfx1150:sramecc-:xnack-", True),  # RDNA 3.5
            ("gfx1151:sramecc-:xnack-", True),  # RDNA 3.5 (Strix Halo)
            # RDNA 4 architectures - should support FP8 (gfx12x)
            ("gfx1200", True),  # RDNA 4
            ("gfx1201:sramecc-:xnack-", True),  # RDNA 4
            # Older architectures - should NOT support FP8
            ("gfx90a:sramecc+:xnack-", False),  # MI200 series
            ("gfx908:sramecc+:xnack-", False),  # MI100
            ("gfx1030", False),  # RDNA 2 (RX 6800)
            ("gfx1010", False),  # RDNA 1
        ],
    )
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.is_available", return_value=True)
    def test_supports_fp8_architectures(
        self, mock_cuda_available, mock_get_props, gcn_arch, expected
    ):
        """Test FP8 support detection for various GPU architectures."""
        mock_get_props.return_value = MockDeviceProperties(gcn_arch)

        # Import after patching to ensure the mock is used
        from vllm.platforms.rocm import RocmPlatform

        result = RocmPlatform.supports_fp8()
        assert result == expected, (
            f"supports_fp8() returned {result} for {gcn_arch}, expected {expected}"
        )


class TestRocmFP8Fnuz:
    """Test cases for is_fp8_fnuz() function."""

    @pytest.mark.parametrize(
        "gcn_arch,expected",
        [
            # MI300 series uses fnuz FP8 format
            ("gfx942:sramecc+:xnack-", True),  # MI300X
            ("gfx940:sramecc+:xnack-", True),  # MI300A
            # Other architectures use standard FP8 format
            ("gfx950:sramecc+:xnack-", False),  # MI350
            ("gfx1151:sramecc-:xnack-", False),  # Strix Halo
            ("gfx1100", False),  # RDNA 3
            ("gfx90a:sramecc+:xnack-", False),  # MI200
        ],
    )
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.is_available", return_value=True)
    def test_is_fp8_fnuz_architectures(
        self, mock_cuda_available, mock_get_props, gcn_arch, expected
    ):
        """Test fnuz FP8 format detection for various GPU architectures."""
        mock_get_props.return_value = MockDeviceProperties(gcn_arch)

        from vllm.platforms.rocm import RocmPlatform

        result = RocmPlatform.is_fp8_fnuz()
        assert result == expected, (
            f"is_fp8_fnuz() returned {result} for {gcn_arch}, expected {expected}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
