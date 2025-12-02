# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import UserDict
from collections.abc import Callable, Hashable, Iterator, KeysView, Mapping
from types import MappingProxyType
from typing import NamedTuple, TypeVar, cast, overload

import cachetools

_K = TypeVar("_K", bound=Hashable)
_V = TypeVar("_V")
_T = TypeVar("_T")


class _Sentinel: ...


ALL_PINNED_SENTINEL = _Sentinel()


class _MappingOrderCacheView(UserDict[_K, _V]):
    def __init__(self, data: Mapping[_K, _V], ordered_keys: Mapping[_K, None]):
        super().__init__(data)
        self.ordered_keys = ordered_keys

    def __iter__(self) -> Iterator[_K]:
        return iter(self.ordered_keys)

    def keys(self) -> KeysView[_K]:
        return KeysView(self.ordered_keys)


class CacheInfo(NamedTuple):
    hits: int
    total: int

    @property
    def hit_ratio(self) -> float:
        if self.total == 0:
            return 0

        return self.hits / self.total

    def __sub__(self, other: "CacheInfo"):
        return CacheInfo(
            hits=self.hits - other.hits,
            total=self.total - other.total,
        )


class LRUCache(cachetools.LRUCache[_K, _V]):
    def __init__(self, capacity: float, getsizeof: Callable[[_V], float] | None = None):
        super().__init__(capacity, getsizeof)

        self.pinned_items = set[_K]()

        self._hits = 0
        self._total = 0
        self._last_info = CacheInfo(hits=0, total=0)

    def __getitem__(self, key: _K, *, update_info: bool = True) -> _V:
        value = super().__getitem__(key)

        if update_info:
            self._hits += 1
            self._total += 1

        return value

    def __delitem__(self, key: _K) -> None:
        run_on_remove = key in self
        value = self.__getitem__(key, update_info=False)  # type: ignore[call-arg]
        super().__delitem__(key)
        if key in self.pinned_items:
            # Todo: add warning to inform that del pinned item
            self._unpin(key)
        if run_on_remove:
            self._on_remove(key, value)

    @property
    def cache(self) -> Mapping[_K, _V]:
        """Return the internal cache dictionary in order (read-only)."""
        return _MappingOrderCacheView(
            self._Cache__data,  # type: ignore
            self.order,
        )

    @property
    def order(self) -> Mapping[_K, None]:
        """Return the internal order dictionary (read-only)."""
        return MappingProxyType(self._LRUCache__order)  # type: ignore

    @property
    def capacity(self) -> float:
        return self.maxsize

    @property
    def usage(self) -> float:
        if self.maxsize == 0:
            return 0

        return self.currsize / self.maxsize

    def stat(self, *, delta: bool = False) -> CacheInfo:
        """
        Gets the cumulative number of hits and queries against this cache.

        If `delta=True`, instead gets these statistics
        since the last call that also passed `delta=True`.
        """
        info = CacheInfo(hits=self._hits, total=self._total)

        if delta:
            info_delta = info - self._last_info
            self._last_info = info
            info = info_delta

        return info

    def touch(self, key: _K) -> None:
        try:
            self._LRUCache__order.move_to_end(key)  # type: ignore
        except KeyError:
            self._LRUCache__order[key] = None  # type: ignore

    @overload
    def get(self, key: _K, /) -> _V | None: ...

    @overload
    def get(self, key: _K, /, default: _V | _T) -> _V | _T: ...

    def get(self, key: _K, /, default: _V | _T | None = None) -> _V | _T | None:
        value: _V | _T | None
        if key in self:
            value = self.__getitem__(key, update_info=False)  # type: ignore[call-arg]

            self._hits += 1
        else:
            value = default

        self._total += 1
        return value

    @overload
    def pop(self, key: _K) -> _V: ...

    @overload
    def pop(self, key: _K, default: _V | _T) -> _V | _T: ...

    def pop(self, key: _K, default: _V | _T | None = None) -> _V | _T | None:
        value: _V | _T | None
        if key not in self:
            return default

        value = self.__getitem__(key, update_info=False)  # type: ignore[call-arg]
        self.__delitem__(key)
        return value

    def put(self, key: _K, value: _V) -> None:
        self.__setitem__(key, value)

    def pin(self, key: _K) -> None:
        """
        Pins a key in the cache preventing it from being
        evicted in the LRU order.
        """
        if key not in self:
            raise ValueError(f"Cannot pin key: {key} not in cache.")
        self.pinned_items.add(key)

    def _unpin(self, key: _K) -> None:
        """
        Unpins a key in the cache allowing it to be
        evicted in the LRU order.
        """
        self.pinned_items.remove(key)

    def _on_remove(self, key: _K, value: _V | None) -> None:
        pass

    def remove_oldest(self, *, remove_pinned: bool = False) -> None:
        if len(self) == 0:
            return

        self.popitem(remove_pinned=remove_pinned)

    def _remove_old_if_needed(self) -> None:
        while self.currsize > self.capacity:
            self.remove_oldest()

    def popitem(self, remove_pinned: bool = False):
        """Remove and return the `(key, value)` pair least recently used."""
        if not remove_pinned:
            # pop the oldest item in the cache that is not pinned
            lru_key = next(
                (key for key in self.order if key not in self.pinned_items),
                ALL_PINNED_SENTINEL,
            )
            if lru_key is ALL_PINNED_SENTINEL:
                raise RuntimeError(
                    "All items are pinned, cannot remove oldest from the cache."
                )
        else:
            lru_key = next(iter(self.order))
        value = self.pop(cast(_K, lru_key))
        return (lru_key, value)

    def clear(self) -> None:
        while len(self) > 0:
            self.remove_oldest(remove_pinned=True)

        self._hits = 0
        self._total = 0
        self._last_info = CacheInfo(hits=0, total=0)
