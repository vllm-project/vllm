# SPDX-License-Identifier: Apache-2.0
"""
LRU cache implementations for the cache simulator.

Two variants are provided:

* ``LRUCacheFast`` — O(1) OrderedDict-backed.  Supports only hit/miss queries.
  Use this for capacity sweeps where per-chunk statistics are not needed.

* ``LRUCache`` — O(log n) dict + SortedList.  Adds ``position(key)`` for
  computing cache-position statistics (0 = MRU, capacity-1 = LRU).
  Use this for the detailed single-run report.
"""

# Standard
from collections import OrderedDict
from typing import Any

# Third Party
from sortedcontainers import SortedList


class LRUCacheFast:
    """
    Lightweight LRU cache using a single :class:`OrderedDict`.

    All operations are O(1).  No per-key position tracking.

    Parameters
    ----------
    capacity:
        Maximum number of entries before LRU eviction kicks in.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"LRUCacheFast capacity must be >= 1, got {capacity}")
        self.capacity = capacity
        self._cache: OrderedDict[Any, None] = OrderedDict()
        self.eviction_count: int = 0

    def contains(self, key: Any) -> bool:
        return key in self._cache

    def access(self, key: Any) -> None:
        """Mark an existing entry as most-recently used. O(1)."""
        self._cache.move_to_end(key)

    def insert(self, key: Any) -> None:
        """Insert a new entry, or refresh if already present. O(1)."""
        if key in self._cache:
            self._cache.move_to_end(key)
            return
        if len(self._cache) >= self.capacity:
            self._cache.popitem(last=False)
            self.eviction_count += 1
        self._cache[key] = None

    def __len__(self) -> int:
        return len(self._cache)


class LRUCache:
    """
    LRU cache backed by a dict (O(1) lookup) and a
    :class:`~sortedcontainers.SortedList` (O(log n) rank queries).

    Each entry carries a strictly-increasing clock value so that the SortedList
    order is unambiguous.

    * LRU end = smallest clock value = ``SortedList[0]``
    * MRU end = largest clock value  = ``SortedList[-1]``

    Parameters
    ----------
    capacity:
        Maximum number of entries before LRU eviction kicks in.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"LRUCache capacity must be >= 1, got {capacity}")
        self.capacity = capacity
        self._clock: int = 0
        self._map: dict[Any, int] = {}  # key -> clock value at last access
        self._sl: SortedList = SortedList(key=lambda x: x[0])
        self.eviction_count: int = 0

    def contains(self, key: Any) -> bool:
        return key in self._map

    def position(self, key: Any) -> int:
        """
        LRU rank of *key*: 0 = most-recently used, ``len-1`` = least-recently used.
        O(log n).
        """
        clock = self._map[key]
        idx = self._sl.index((clock, key))
        return len(self._sl) - 1 - idx

    def access(self, key: Any) -> None:
        """Mark an existing entry as most-recently used. O(log n)."""
        old_clock = self._map[key]
        self._sl.remove((old_clock, key))
        self._clock += 1
        self._map[key] = self._clock
        self._sl.add((self._clock, key))

    def insert(self, key: Any) -> None:
        """Insert a new entry, or refresh if already present. O(log n)."""
        if key in self._map:
            self.access(key)
            return
        if len(self._map) >= self.capacity:
            lru_clock, lru_key = self._sl[0]
            self._sl.remove((lru_clock, lru_key))
            del self._map[lru_key]
            self.eviction_count += 1
        self._clock += 1
        self._map[key] = self._clock
        self._sl.add((self._clock, key))

    def __len__(self) -> int:
        return len(self._map)
