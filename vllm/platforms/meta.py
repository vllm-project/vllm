# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-less platform capabilities for meta-mode execution."""


class MetaPlatformMixin:
    """Overrides mixed over the detected native platform in meta mode."""

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        return False

    @classmethod
    def get_current_memory_usage(cls, device=None) -> float:
        return 0.0

    @classmethod
    def num_compute_units(cls, device_id: int = 0) -> int:
        return 1

    @classmethod
    def device_count(cls) -> int:
        return 8


def wrap_meta_platform(platform_cls: type) -> type:
    """Mix meta-device capabilities over a detected platform class."""
    return type(
        f"Meta{platform_cls.__name__}",
        (MetaPlatformMixin, platform_cls),
        {},
    )
