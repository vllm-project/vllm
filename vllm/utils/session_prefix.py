# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded process-local candidate snapshots, including in-flight readers."""

import struct
import sys
import threading
from collections import OrderedDict
from collections.abc import Hashable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generic, TypeVar

_T = TypeVar("_T")


def lcp(a: str, b: str) -> int:
    """Exact common prefix length; slice comparisons run in CPython's C code."""
    n = min(len(a), len(b))
    if a[:n] == b[:n]:
        return n
    lo, hi = 0, n  # a[:lo] == b[:lo] and a[:hi] != b[:hi]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if a[lo:mid] == b[lo:mid]:
            lo = mid
        else:
            hi = mid
    return lo


def token_bytes(ids: list[int]) -> int:
    if (
        sys.implementation.name == "cpython"
        and type(ids) is list
        and not (set(map(type, ids)) - {int})
    ):
        maximum = max(ids, default=0)
        if min(ids, default=0) >= 0 and maximum <= 0xFFFFFFFF:
            # Count every occurrence, including shared integer objects.
            return sys.getsizeof([]) + len(ids) * (
                struct.calcsize("P") + sys.getsizeof(maximum)
            )
    return (
        sys.getsizeof([])
        + len(ids) * struct.calcsize("P")
        + sum(map(sys.getsizeof, ids))
    )


@dataclass
class _Entry(Generic[_T]):
    value: _T
    size: int
    readers: int = 0
    resident: bool = True


class Reservation(Generic[_T]):
    def __init__(self, store: "SessionPrefixStore[_T]", key: Hashable, size: int):
        self.store = store
        self.key = key
        self.size = size
        self.epoch = store.epoch
        self.active = True

    def commit(self, value: _T, size: int) -> bool:
        if not 0 <= size <= self.size:
            raise ValueError("snapshot exceeds its reservation")
        previous = None
        with self.store._lock:
            if not self.active:
                return False
            self.active = False
            self.store.bytes_used -= self.size
            if self.store.closed or self.epoch != self.store.epoch:
                return False
            previous = self.store._entries.pop(self.key, None)
            if previous is not None:
                self.store._retire(previous)
            self.store._entries[self.key] = _Entry(value, size)
            self.store.bytes_used += size
        return True

    def cancel(self) -> None:
        with self.store._lock:
            if self.active:
                self.active = False
                self.store.bytes_used -= self.size


class SessionPrefixStore(Generic[_T]):
    """One candidate per key; capacity includes reservations and retired readers.

    Values are owned exclusively by this store and must not be mutated after
    publication. Completion order determines which candidate is retained.
    """

    def __init__(self, capacity: int):
        if capacity < 0:
            raise ValueError("session history capacity must be nonnegative")
        self.capacity = capacity
        self.bytes_used = 0
        self.epoch = 0
        self.closed = False
        self._lock = threading.Lock()
        self._entries: OrderedDict[Hashable, _Entry[_T]] = OrderedDict()

    def _retire(self, entry: _Entry[_T]) -> None:
        entry.resident = False
        if not entry.readers:
            self.bytes_used -= entry.size

    @contextmanager
    def borrow(self, key: Hashable) -> Iterator[_T | None]:
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                entry.readers += 1
                self._entries.move_to_end(key)
        try:
            yield entry.value if entry is not None else None
        finally:
            if entry is not None:
                with self._lock:
                    entry.readers -= 1
                    if not entry.resident and not entry.readers:
                        self.bytes_used -= entry.size

    def reserve(self, key: Hashable, size: int) -> Reservation[_T] | None:
        retired = []
        with self._lock:
            if self.closed or size <= 0 or size > self.capacity:
                return None
            while self.bytes_used + size > self.capacity and self._entries:
                _, entry = self._entries.popitem(last=False)
                self._retire(entry)
                retired.append(entry)
            if self.bytes_used + size > self.capacity:
                return None
            self.bytes_used += size
            result = Reservation(self, key, size)
        return result

    def clear(self, *, close: bool = False) -> None:
        with self._lock:
            self.epoch += 1
            self.closed |= close
            entries, self._entries = self._entries, OrderedDict()
            for entry in entries.values():
                self._retire(entry)


@dataclass
class PrefixContext:
    """Request-local publication, never serialized or shared between requests."""

    key: tuple[str | None, str]
    epoch: int
    pending: tuple[Reservation, object, int] | None = None
    active: bool = True
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def stage(self, reservation: Reservation, value: object, size: int) -> None:
        with self._lock:
            if self.active and self.pending is None and reservation.epoch == self.epoch:
                self.pending = reservation, value, size
                return
        reservation.cancel()

    def finish(self, *, publish: bool) -> None:
        with self._lock:
            self.active = False
            pending, self.pending = self.pending, None
        if pending is not None:
            reservation, value, size = pending
            if publish:
                reservation.commit(value, size)
            else:
                reservation.cancel()
