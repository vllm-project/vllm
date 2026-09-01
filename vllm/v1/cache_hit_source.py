# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum


class CacheHitSource(str, Enum):
    """Bounded origins for cached prompt tokens."""

    DEVICE = "device"
    CPU = "cpu"
    DISK = "disk"
    P2P = "p2p"
    EXTERNAL = "external"


_SOURCE_ALIASES = {
    "device": CacheHitSource.DEVICE,
    "gpu": CacheHitSource.DEVICE,
    "cpu": CacheHitSource.CPU,
    "dram": CacheHitSource.CPU,
    "host": CacheHitSource.CPU,
    "disk": CacheHitSource.DISK,
    "file": CacheHitSource.DISK,
    "fs": CacheHitSource.DISK,
    "nvme": CacheHitSource.DISK,
    "p2p": CacheHitSource.P2P,
}


def normalize_cache_hit_source(source: str | CacheHitSource) -> CacheHitSource:
    """Map connector-specific tier names into the public source contract."""
    if isinstance(source, CacheHitSource):
        return source
    return _SOURCE_ALIASES.get(source.lower(), CacheHitSource.EXTERNAL)
