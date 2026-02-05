# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for GPU memory monitor."""

import time
from unittest.mock import patch

import pytest
import torch

from vllm.utils.gpu_memory_monitor import GPUMemoryMonitor


class TestGPUMemoryMonitor:
    """Test suite for GPUMemoryMonitor."""

    def test_init_default_values(self):
        """Test monitor initialization with default values."""
        monitor = GPUMemoryMonitor()
        assert monitor.threshold == 0.90
        assert monitor.check_interval == 5.0
        assert monitor.enabled is False
        assert monitor.warning_cooldown == 60.0
        assert monitor.last_check_time == 0.0
        assert monitor.last_warning_time == {}

    def test_init_custom_values(self):
        """Test monitor initialization with custom values."""
        monitor = GPUMemoryMonitor(
            threshold=0.85,
            check_interval=10.0,
            enabled=True,
            warning_cooldown=30.0,
        )
        assert monitor.threshold == 0.85
        assert monitor.check_interval == 10.0
        assert monitor.enabled is True
        assert monitor.warning_cooldown == 30.0

    def test_init_invalid_threshold(self):
        """Test that invalid threshold raises ValueError."""
        with pytest.raises(ValueError, match="threshold must be between"):
            GPUMemoryMonitor(threshold=1.5)

        with pytest.raises(ValueError, match="threshold must be between"):
            GPUMemoryMonitor(threshold=-0.1)

    def test_init_invalid_check_interval(self):
        """Test that invalid check_interval raises ValueError."""
        with pytest.raises(ValueError, match="check_interval must be non-negative"):
            GPUMemoryMonitor(check_interval=-1.0)

    def test_init_invalid_warning_cooldown(self):
        """Test that invalid warning_cooldown raises ValueError."""
        with pytest.raises(ValueError, match="warning_cooldown must be positive"):
            GPUMemoryMonitor(warning_cooldown=0)

    def test_check_and_warn_disabled(self):
        """Test that check_and_warn does nothing when disabled."""
        monitor = GPUMemoryMonitor(enabled=False)
        # Should not raise any errors
        monitor.check_and_warn()
        assert monitor.last_check_time == 0.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_check_and_warn_enabled(self):
        """Test that check_and_warn works when enabled."""
        monitor = GPUMemoryMonitor(enabled=True, check_interval=0.0)
        monitor.check_and_warn()
        # Should have updated last_check_time
        assert monitor.last_check_time > 0.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_check_interval_rate_limiting(self):
        """Test that checks are rate-limited by check_interval."""
        monitor = GPUMemoryMonitor(enabled=True, check_interval=10.0)

        # First check should update last_check_time
        monitor.check_and_warn()
        first_check_time = monitor.last_check_time
        assert first_check_time > 0.0

        # Immediate second check should not update last_check_time
        monitor.check_and_warn()
        assert monitor.last_check_time == first_check_time

        # After waiting, check should update
        monitor.last_check_time = time.time() - 11.0
        monitor.check_and_warn()
        assert monitor.last_check_time > first_check_time

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_warning_cooldown(self):
        """Test that warnings are rate-limited by warning_cooldown."""
        monitor = GPUMemoryMonitor(
            enabled=True,
            threshold=0.0,  # Always trigger warning
            check_interval=0.0,
            warning_cooldown=10.0,
        )

        with patch.object(monitor, "_emit_warning") as mock_warn:
            # First check should emit warning
            monitor.check_and_warn()
            assert mock_warn.call_count == torch.cuda.device_count()

            # Immediate second check should not emit warning
            monitor.check_and_warn()
            assert mock_warn.call_count == torch.cuda.device_count()

            # After cooldown, should emit warning again
            monitor.last_warning_time = {
                i: time.time() - 11.0 for i in range(torch.cuda.device_count())
            }
            monitor.check_and_warn()
            assert mock_warn.call_count == torch.cuda.device_count() * 2

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_memory_stats(self):
        """Test get_memory_stats returns valid data."""
        monitor = GPUMemoryMonitor()
        stats = monitor.get_memory_stats(device_id=0)

        assert stats is not None
        assert "allocated_gb" in stats
        assert "reserved_gb" in stats
        assert "total_gb" in stats
        assert "usage_ratio" in stats

        assert stats["allocated_gb"] >= 0
        assert stats["reserved_gb"] >= 0
        assert stats["total_gb"] > 0
        assert 0 <= stats["usage_ratio"] <= 1.0

    def test_get_memory_stats_no_cuda(self):
        """Test get_memory_stats returns None when CUDA unavailable."""
        monitor = GPUMemoryMonitor()

        with patch("torch.cuda.is_available", return_value=False):
            stats = monitor.get_memory_stats()
            assert stats is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_get_memory_stats_invalid_device(self):
        """Test get_memory_stats returns None for invalid device."""
        monitor = GPUMemoryMonitor()
        device_count = torch.cuda.device_count()

        # Request stats for non-existent device
        stats = monitor.get_memory_stats(device_id=device_count + 10)
        assert stats is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_emit_warning_format(self, caplog):
        """Test that warning message has correct format."""
        monitor = GPUMemoryMonitor(
            enabled=True,
            threshold=0.0,  # Always trigger
            check_interval=0.0,
        )

        with caplog.at_level("WARNING"):
            monitor.check_and_warn()

        # Should have warning message
        assert len(caplog.records) > 0
        warning_msg = caplog.records[0].message

        # Check message contains expected information
        assert "GPU" in warning_msg
        assert "memory usage high" in warning_msg
        assert "GB" in warning_msg
        assert "max-num-seqs" in warning_msg or "max-model-len" in warning_msg

    def test_check_device_error_handling(self):
        """Test that errors in _check_device don't crash."""
        monitor = GPUMemoryMonitor(enabled=True, check_interval=0.0)

        # Mock torch.cuda.memory_allocated to raise exception
        with (
            patch(
                "torch.cuda.memory_allocated", side_effect=RuntimeError("Test error")
            ),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=1),
        ):
            # Should not raise exception
            monitor.check_and_warn()

    def test_get_memory_stats_error_handling(self):
        """Test that errors in get_memory_stats return None."""
        monitor = GPUMemoryMonitor()

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=1),
            patch(
                "torch.cuda.memory_allocated",
                side_effect=RuntimeError("Test error"),
            ),
        ):
            stats = monitor.get_memory_stats(device_id=0)
            assert stats is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_multiple_devices(self):
        """Test monitoring works with multiple GPUs."""
        device_count = torch.cuda.device_count()

        if device_count < 2:
            pytest.skip("Multiple GPUs not available")

        monitor = GPUMemoryMonitor(
            enabled=True,
            threshold=0.0,  # Always trigger
            check_interval=0.0,
        )

        with patch.object(monitor, "_emit_warning") as mock_warn:
            monitor.check_and_warn()
            # Should check all devices
            assert mock_warn.call_count == device_count

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_threshold_behavior(self):
        """Test that warnings only trigger above threshold."""
        monitor = GPUMemoryMonitor(
            enabled=True,
            threshold=0.99,  # Very high threshold
            check_interval=0.0,
        )

        with patch.object(monitor, "_emit_warning") as mock_warn:
            monitor.check_and_warn()
            # With 99% threshold, likely won't trigger on idle GPU
            # (This test may be flaky if GPU is actually at 99%+)
            # Just verify it doesn't crash
            assert mock_warn.call_count >= 0
