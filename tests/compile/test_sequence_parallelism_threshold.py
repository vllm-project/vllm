# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.compilation.passes.fusion.sequence_parallelism import (
    SP_MIN_HIDDEN_SIZE,
    SP_MIN_PER_GPU_SIZE_MB,
    get_sequence_parallelism_threshold,
)


class TestGetSequenceParallelismThreshold:
    """Tests for get_sequence_parallelism_threshold function."""

    def test_non_cuda_alike_returns_none(self, mock_cuda_platform):
        """Non-CUDA-alike platforms should return None."""
        with mock_cuda_platform(is_cuda=False, is_cuda_alike=False):
            result = get_sequence_parallelism_threshold(
                hidden_size=8192, tp_size=2, element_size=2
            )
        assert result is None

    def test_unsupported_device_capability_returns_none(self, mock_cuda_platform):
        """Unsupported device capabilities (e.g., sm80) should return None."""
        with mock_cuda_platform(capability=(8, 0)):
            result = get_sequence_parallelism_threshold(
                hidden_size=8192, tp_size=2, element_size=2
            )
        assert result is None

    def test_small_hidden_size_returns_none(self, mock_cuda_platform):
        """H100 with hidden_size below threshold should return None."""
        with mock_cuda_platform(capability=(9, 0)):
            result = get_sequence_parallelism_threshold(
                hidden_size=4096,
                tp_size=2,
                element_size=2,  # 4096 < 8192
            )
        assert result is None

    def test_h100_large_model_returns_threshold(self, mock_cuda_platform):
        """H100 with large enough hidden_size should return calculated threshold."""
        with mock_cuda_platform(capability=(9, 0)):
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
        self, mock_cuda_platform, hidden_size, tp_size, element_size, expected
    ):
        """Test threshold calculation with various parameter combinations."""
        with mock_cuda_platform(capability=(9, 0)):
            result = get_sequence_parallelism_threshold(
                hidden_size=hidden_size,
                tp_size=tp_size,
                element_size=element_size,
            )
            assert result == expected

    def test_hidden_size_boundary(self, mock_cuda_platform):
        """Test behavior at the exact hidden_size boundary."""
        with mock_cuda_platform(capability=(9, 0)):
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


class TestGetSequenceParallelismThresholdROCm:
    """Tests for get_sequence_parallelism_threshold on ROCm platforms."""

    def test_mi300x_gfx942_returns_threshold(self, mock_cuda_platform):
        """MI300X (gfx942, capability 9.4) should return a calculated threshold."""
        with mock_cuda_platform(is_cuda=False, is_cuda_alike=True, capability=(9, 4)):
            hidden_size = 7168
            tp_size = 8
            element_size = 2

            result = get_sequence_parallelism_threshold(
                hidden_size=hidden_size,
                tp_size=tp_size,
                element_size=element_size,
            )

            MiB = 1024 * 1024
            expected = int(
                (SP_MIN_PER_GPU_SIZE_MB[94] * tp_size * MiB)
                // (hidden_size * element_size)
            )
            assert result == expected

    def test_mi355x_gfx950_returns_threshold(self, mock_cuda_platform):
        """MI355X (gfx950, capability 9.5) should return a calculated threshold."""
        with mock_cuda_platform(is_cuda=False, is_cuda_alike=True, capability=(9, 5)):
            hidden_size = 7168
            tp_size = 4
            element_size = 2

            result = get_sequence_parallelism_threshold(
                hidden_size=hidden_size,
                tp_size=tp_size,
                element_size=element_size,
            )

            MiB = 1024 * 1024
            expected = int(
                (SP_MIN_PER_GPU_SIZE_MB[95] * tp_size * MiB)
                // (hidden_size * element_size)
            )
            assert result == expected

    def test_mi355x_small_hidden_size_returns_none(self, mock_cuda_platform):
        """MI355X with hidden_size below threshold should return None."""
        with mock_cuda_platform(is_cuda=False, is_cuda_alike=True, capability=(9, 5)):
            result = get_sequence_parallelism_threshold(
                hidden_size=4096,
                tp_size=4,
                element_size=2,
            )
            assert result is None

    def test_mi355x_hidden_size_boundary(self, mock_cuda_platform):
        """Test boundary at the exact hidden_size threshold for MI355X."""
        with mock_cuda_platform(is_cuda=False, is_cuda_alike=True, capability=(9, 5)):
            # Just below threshold
            result = get_sequence_parallelism_threshold(
                hidden_size=SP_MIN_HIDDEN_SIZE[95] - 1,
                tp_size=4,
                element_size=2,
            )
            assert result is None

            # Exactly at threshold
            result = get_sequence_parallelism_threshold(
                hidden_size=SP_MIN_HIDDEN_SIZE[95],
                tp_size=4,
                element_size=2,
            )
            assert result is not None

    @pytest.mark.parametrize(
        "capability,cap_int",
        [
            ((9, 4), 94),
            ((9, 5), 95),
        ],
    )
    def test_rocm_tp_scaling(self, mock_cuda_platform, capability, cap_int):
        """Verify threshold scales with TP size on ROCm."""
        with mock_cuda_platform(
            is_cuda=False, is_cuda_alike=True, capability=capability
        ):
            hidden_size = SP_MIN_HIDDEN_SIZE[cap_int]
            element_size = 2

            result_tp2 = get_sequence_parallelism_threshold(
                hidden_size=hidden_size, tp_size=2, element_size=element_size
            )
            result_tp8 = get_sequence_parallelism_threshold(
                hidden_size=hidden_size, tp_size=8, element_size=element_size
            )
            assert result_tp2 is not None
            assert result_tp8 is not None
            assert result_tp8 == 4 * result_tp2
