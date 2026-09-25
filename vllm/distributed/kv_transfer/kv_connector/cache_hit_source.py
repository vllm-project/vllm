# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class CacheHitSource(str, Enum):
    """Bounded origins for cached prompt tokens, declared fastest to slowest."""

    DEVICE = "device"
    HOST = "host"
    P2P = "p2p"
    DISK = "disk"
    # Fallback: uninstrumented connector or remote store. Ranks slowest so a
    # token partly served from an unknown tier is never labeled a known one.
    EXTERNAL_UNSPECIFIED = "external_unspecified"

    @classmethod
    def slowest(cls, sources: Iterable[CacheHitSource]) -> CacheHitSource:
        """The slowest of ``sources``; a token needs KV from all of them."""
        order = list(cls)
        return max(sources, key=order.index)


@dataclass
class CachedTokensBySource:
    """Token counts per ``CacheHitSource``.

    Fixed int fields, not a dict: an unknown source fails type-checking
    instead of creating a new metric series.
    """

    device: int = 0
    host: int = 0
    p2p: int = 0
    disk: int = 0
    external_unspecified: int = 0

    def add(self, source: CacheHitSource, num_tokens: int) -> None:
        name = CacheHitSource(source).value
        setattr(self, name, getattr(self, name) + num_tokens)

    def merge(self, other: CachedTokensBySource) -> None:
        for source in CacheHitSource:
            self.add(source, getattr(other, source.value))

    def items(self) -> list[tuple[str, int]]:
        """Non-zero ``(label, count)`` pairs in ``CacheHitSource`` order."""
        return [
            (source.value, num_tokens)
            for source in CacheHitSource
            if (num_tokens := getattr(self, source.value))
        ]

    @property
    def total(self) -> int:
        return sum(getattr(self, source.value) for source in CacheHitSource)
