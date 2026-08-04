# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for distributed tests."""

# First Party
from lmcache.v1.platform import current_device_spec


def should_use_lazy_alloc() -> bool:
    """Return whether the current platform supports lazy L1 allocation."""
    return current_device_spec.is_pin_supported
