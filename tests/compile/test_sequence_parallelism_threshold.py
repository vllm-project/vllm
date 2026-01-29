# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch

import pytest

from vllm.compilation.sequence_parallelism import (
    SP_MIN_HIDDEN_SIZE,
    SP_MIN_PER_GPU_SIZE_MB,
    get_sequence_parallelism_threshold,
)
from vllm.platforms.interface import DeviceCapability


class TestGetSequenceParallelismThreshold:
    """Tests for get_sequence_parallelism_threshold function."""

    def test_non_cuda_returns_none(self):
        """Non-CUDA platforms should return None."""
        mock_platform = MagicMock()
        mock_platform.is_cuda.return_value = False

        # Patch at vllm.platforms since the function does a local import
        with patch("vllm.platforms.current_platform", mock_platform):
            result = get_sequence_parallelism_threshold(
                hidden_size=8192, tp_size=2, element_size=2
            )
        assert result is None

    def test_unsupported_device_capability_returns_none(self):
        """Unsupported device capabilities (e.g., sm80) should return None."""
        mock_platform = MagicMock()
        mock_platform.is_cuda.return_value = True
        mock_platform.get_device_capability.return_value = DeviceCapability(8, 0)

        with patch("vllm.platforms.current_platform", mock_platform):
            result = get_sequence_parallelism_threshold(
                hidden_size=8192, tp_size=2, element_size=2
            )
        assert result is None

    def test_small_hidden_size_returns_none(self):
        """H100 with hidden_size below threshold should return None."""
        mock_platform = MagicMock()
        mock_platform.is_cuda.return_value = True
        mock_platform.get_device_capability.return_value = DeviceCapability(9, 0)

        with patch("vllm.platforms.current_platform", mock_platform):
            result = get_sequence_parallelism_threshold(
                hidden_size=4096,
                tp_size=2,
                element_size=2,  # 4096 < 8192
            )
        assert result is None

    def test_h100_large_model_returns_threshold(self):
        """H100 with large enough hidden_size should return calculated threshold."""
        mock_platform = MagicMock()
        mock_platform.is_cuda.return_value = True
        mock_platform.get_device_capability.return_value = DeviceCapability(9, 0)

        with patch("vllm.platforms.current_platform", mock_platform):
            hidden_size = 8192
            tp_size = 2
            element_size = 2  # float16/bfloat16

            result = get_sequence_parallelism_threshold(
                hidden_size=hidden_size,
                tp_size=tp_size,
                element_size=element_size,
            )

            # Verify calculation: (8 * 2 * 1024 * 1024) // (8192 * 2) = 1024
            MiB = 1024 * 1024
            expected = int(
                (SP_MIN_PER_GPU_SIZE_MB[90] * tp_size * MiB)
                // (hidden_size * element_size)
            )
            assert result == expected
            assert result == 1024

    @pytest.mark.parametrize(
        "hidden_size,tp_size,element_size,expected",
        [
            # Boundary: exactly at min hidden size threshold, tp_size=1
            # (8 * 1 * 1024 * 1024) // (8192 * 2) = 512
            (8192, 1, 2, 512),
            # Larger hidden size reduces token threshold
            # (8 * 1 * 1024 * 1024) // (16384 * 2) = 256
            (16384, 1, 2, 256),
            # Larger tp_size increases token threshold
            # (8 * 4 * 1024 * 1024) // (8192 * 2) = 2048
            (8192, 4, 2, 2048),
            # Larger element_size (fp32) reduces token threshold
            # (8 * 2 * 1024 * 1024) // (8192 * 4) = 512
            (8192, 2, 4, 512),
        ],
    )
    def test_threshold_calculation_variations(
        self, hidden_size, tp_size, element_size, expected
    ):
        """Test threshold calculation with various parameter combinations."""
        mock_platform = MagicMock()
        mock_platform.is_cuda.return_value = True
        mock_platform.get_device_capability.return_value = DeviceCapability(9, 0)

        with patch("vllm.platforms.current_platform", mock_platform):
            result = get_sequence_parallelism_threshold(
                hidden_size=hidden_size,
                tp_size=tp_size,
                element_size=element_size,
            )
            assert result == expected

    def test_hidden_size_boundary(self):
        """Test behavior at the exact hidden_size boundary."""
        mock_platform = MagicMock()
        mock_platform.is_cuda.return_value = True
        mock_platform.get_device_capability.return_value = DeviceCapability(9, 0)

        with patch("vllm.platforms.current_platform", mock_platform):
            # Just below threshold
            result = get_sequence_parallelism_threshold(
                hidden_size=SP_MIN_HIDDEN_SIZE[90] - 1,
                tp_size=2,
                element_size=2,
            )
            assert result is None

            # Exactly at threshold
            result = get_sequence_parallelism_threshold(
                hidden_size=SP_MIN_HIDDEN_SIZE[90],
                tp_size=2,
                element_size=2,
            )
            assert result is not None
