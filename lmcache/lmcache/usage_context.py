# SPDX-License-Identifier: Apache-2.0
"""Backward-compatibility shim; the implementation lives in
:mod:`lmcache.usage_telemetry`."""

# First Party
from lmcache.usage_telemetry import (
    ContinuousContextMessage,
    ContinuousUsageContext,
    EngineMessage,
    EnvMessage,
    InitializeUsageContext,
    MetadataMessage,
    UsageContext,
)

__all__ = [
    "ContinuousContextMessage",
    "ContinuousUsageContext",
    "EngineMessage",
    "EnvMessage",
    "InitializeUsageContext",
    "MetadataMessage",
    "UsageContext",
]
