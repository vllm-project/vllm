# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum


class CacheHitSource(str, Enum):
    """Bounded origins for cached prompt tokens."""

    DEVICE = "device"
    HOST = "host"
    DISK = "disk"
    P2P = "p2p"
    # Fallback: uninstrumented connector, remote store, or a range mixing tiers.
    EXTERNAL_UNSPECIFIED = "external_unspecified"
