# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU memory monitoring and warning system."""

import logging
import time

import torch

logger = logging.getLogger(__name__)


class GPUMemoryMonitor:
    """Monitors GPU memory usage and emits warnings when threshold exceeded.

    This is an opt-in monitoring system that helps prevent OOM crashes by
    warning users when GPU memory usage is high. It has zero overhead when
    disabled (default).

    Args:
        threshold: Memory usage ratio (0.0-1.0) that triggers warnings.
            Default is 0.90 (90%).
        check_interval: Minimum seconds between memory checks.
            Default is 5.0 seconds.
        enabled: Whether monitoring is enabled. Default is False.
        warning_cooldown: Minimum seconds between warning messages to avoid
            log spam. Default is 60.0 seconds.

    Example:
        >>> monitor = GPUMemoryMonitor(enabled=True, threshold=0.85)
        >>> monitor.check_and_warn()  # Call periodically
    """

    def __init__(
        self,
        threshold: float = 0.90,
        check_interval: float = 5.0,
        enabled: bool = False,
        warning_cooldown: float = 60.0,
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be between 0.0 and 1.0, got {threshold}")
        if check_interval < 0:
            raise ValueError(
                f"check_interval must be non-negative, got {check_interval}"
            )
        if warning_cooldown <= 0:
            raise ValueError(
                f"warning_cooldown must be positive, got {warning_cooldown}"
            )

        self.threshold = threshold
        self.check_interval = check_interval
        self.enabled = enabled
        self.warning_cooldown = warning_cooldown

        self.last_check_time = 0.0
        self.last_warning_time: dict[int, float] = {}

    def check_and_warn(self) -> None:
        """Check GPU memory usage and emit warning if threshold exceeded.

        This method is designed to be called frequently (e.g., after each
        model execution step) but will only perform actual checks based on
        check_interval to minimize overhead.

        Warnings are rate-limited by warning_cooldown to avoid log spam.
        """
        if not self.enabled:
            return

        current_time = time.time()

        # Rate limit checks
        if current_time - self.last_check_time < self.check_interval:
            return

        self.last_check_time = current_time

        # Only check if CUDA is available
        if not torch.cuda.is_available():
            return

        # Check each GPU device
        for device_id in range(torch.cuda.device_count()):
            self._check_device(device_id, current_time)

    def _check_device(self, device_id: int, current_time: float) -> None:
        """Check memory usage for a specific GPU device."""
        try:
            memory_allocated = torch.cuda.memory_allocated(device_id)
            memory_reserved = torch.cuda.memory_reserved(device_id)
            memory_total = torch.cuda.get_device_properties(device_id).total_memory

            # Use reserved memory for threshold check as it's more accurate
            # for predicting OOM
            usage_ratio = memory_reserved / memory_total

            last_warning_time_device = self.last_warning_time.get(device_id, 0.0)
            if (
                usage_ratio >= self.threshold
                and current_time - last_warning_time_device > self.warning_cooldown
            ):
                self._emit_warning(
                    device_id,
                    usage_ratio,
                    memory_allocated,
                    memory_reserved,
                    memory_total,
                )
                self.last_warning_time[device_id] = current_time

        except Exception as e:
            # Don't let monitoring errors crash the system
            logger.debug(
                "Error checking GPU %d memory: %s",
                device_id,
                e,
                exc_info=True,
            )

    def _emit_warning(
        self,
        device_id: int,
        usage_ratio: float,
        memory_allocated: int,
        memory_reserved: int,
        memory_total: int,
    ) -> None:
        """Emit structured warning about high GPU memory usage."""
        allocated_gb = memory_allocated / 1e9
        reserved_gb = memory_reserved / 1e9
        total_gb = memory_total / 1e9

        logger.warning(
            "GPU %d memory usage high: %.1f%% "
            "(reserved: %.2fGB / %.2fGB, "
            "allocated: %.2fGB). "
            "Consider reducing --max-num-seqs, --max-model-len, "
            "or using a smaller model to avoid OOM.",
            device_id,
            usage_ratio * 100,
            reserved_gb,
            total_gb,
            allocated_gb,
        )

    def get_memory_stats(self, device_id: int = 0) -> dict | None:
        """Get current memory statistics for a GPU device.

        Args:
            device_id: GPU device ID to query. Default is 0.

        Returns:
            Dictionary with memory statistics, or None if CUDA unavailable.
            Keys: allocated_gb, reserved_gb, total_gb, usage_ratio
        """
        if not torch.cuda.is_available():
            return None

        if device_id >= torch.cuda.device_count():
            return None

        try:
            memory_allocated = torch.cuda.memory_allocated(device_id)
            memory_reserved = torch.cuda.memory_reserved(device_id)
            memory_total = torch.cuda.get_device_properties(device_id).total_memory

            return {
                "allocated_gb": memory_allocated / 1e9,
                "reserved_gb": memory_reserved / 1e9,
                "total_gb": memory_total / 1e9,
                "usage_ratio": memory_reserved / memory_total,
            }
        except Exception as e:
            logger.debug(
                "Error getting GPU %d memory stats: %s",
                device_id,
                e,
                exc_info=True,
            )
            return None
