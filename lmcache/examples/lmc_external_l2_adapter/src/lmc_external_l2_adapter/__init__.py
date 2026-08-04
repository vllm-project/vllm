# SPDX-License-Identifier: Apache-2.0
"""
lmc_external_l2_adapter - Example external L2 adapter
plugin for LMCache.

This package provides a simple in-memory L2 adapter that
demonstrates how to build an external plugin loaded via
the ``PluginL2AdapterConfig`` mechanism.
"""

# Third Party
from lmc_external_l2_adapter.adapter import (
    InMemoryL2Adapter,
    InMemoryL2AdapterConfig,
)

__all__ = ["InMemoryL2Adapter", "InMemoryL2AdapterConfig"]
