# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum


class CacheHitSource(str, Enum):
    """Where cached prompt tokens' KV came from.

    Members are listed in the order blocks are offloaded: accelerator, then
    host memory, then secondary tiers. Per-source token counts are plain
    ``dict[CacheHitSource, int]`` mappings holding only non-zero entries.
    """

    DEVICE = "device"
    HOST = "host"
    P2P = "p2p"
    DISK = "disk"
    # Fallback: uninstrumented connector or remote store. Last, so mixing
    # with a known tier never reports the known one.
    EXTERNAL_UNSPECIFIED = "external_unspecified"

    @classmethod
    def outermost(cls, sources: Iterable[CacheHitSource]) -> CacheHitSource:
        """The tier in ``sources`` farthest from the accelerator."""
        return max(sources, key=_SOURCE_ORDER.__getitem__)


_SOURCE_ORDER = {source: rank for rank, source in enumerate(CacheHitSource)}
