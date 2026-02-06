# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Contains helpers that are applied to collections.

This is similar in concept to the `collections` module.
"""

import threading
from collections import defaultdict
from collections.abc import Callable, Generator, Hashable, Iterable, Mapping, Sequence
from typing import Generic, Literal, TypeVar

from typing_extensions import TypeIs, assert_never, overload

T = TypeVar("T")

_K = TypeVar("_K", bound=Hashable)
_V = TypeVar("_V")


class LazyDict(Mapping[str, _V], Generic[_V]):
    """
    Evaluates dictionary items only when they are accessed.

    Adapted from: https://stackoverflow.com/a/47212782/5082708
    """

    def __init__(self, factory: dict[str, Callable[[], _V]]):
        self._factory = factory
        self._dict: dict[str, _V] = {}

    def __getitem__(self, key: str) -> _V:
        if key not in self._dict:
            if key not in self._factory:
                raise KeyError(key)
            self._dict[key] = self._factory[key]()
        return self._dict[key]

    def __setitem__(self, key: str, value: Callable[[], _V]):
        self._factory[key] = value

    def __iter__(self):
        return iter(self._factory)

    def __len__(self):
        return len(self._factory)


def as_list(maybe_list: Iterable[T]) -> list[T]:
    """Convert iterable to list, unless it's already a list."""
    return maybe_list if isinstance(maybe_list, list) else list(maybe_list)


def as_iter(obj: T | Iterable[T]) -> Iterable[T]:
    if isinstance(obj, str) or not isinstance(obj, Iterable):
        return [obj]  # type: ignore[list-item]
    return obj


def is_list_of(
    value: object,
    typ: type[T] | tuple[type[T], ...],
    *,
    check: Literal["first", "all"] = "first",
) -> TypeIs[list[T]]:
    if not isinstance(value, list):
        return False

    if check == "first":
        return len(value) == 0 or isinstance(value[0], typ)
    elif check == "all":
        return all(isinstance(v, typ) for v in value)

    assert_never(check)


@overload
def common_prefix(items: Sequence[str]) -> str: ...


@overload
def common_prefix(items: Sequence[Sequence[T]]) -> Sequence[T]: ...


def common_prefix(items: Sequence[Sequence[T] | str]) -> Sequence[T] | str:
    """Find the longest prefix common to all items."""
    if len(items) == 0:
        return []
    if len(items) == 1:
        return items[0]

    shortest = min(items, key=len)
    if not shortest:
        return shortest[:0]

    for match_len in range(1, len(shortest) + 1):
        match = shortest[:match_len]
        for item in items:
            if item[:match_len] != match:
                return shortest[: match_len - 1]

    return shortest


def chunk_list(lst: list[T], chunk_size: int) -> Generator[list[T]]:
    """Yield successive chunk_size chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def flatten_2d_lists(lists: Iterable[Iterable[T]]) -> list[T]:
    """Flatten a list of lists to a single list."""
    return [item for sublist in lists for item in sublist]


def full_groupby(values: Iterable[_V], *, key: Callable[[_V], _K]):
    """
    Unlike [`itertools.groupby`][], groups are not broken by
    non-contiguous data.
    """
    groups = defaultdict[_K, list[_V]](list)

    for value in values:
        groups[key(value)].append(value)

    return groups.items()


def swap_dict_values(obj: dict[_K, _V], key1: _K, key2: _K) -> None:
    """Swap values between two keys."""
    v1 = obj.get(key1)
    v2 = obj.get(key2)
    if v1 is not None:
        obj[key2] = v1
    else:
        obj.pop(key2, None)
    if v2 is not None:
        obj[key1] = v2
    else:
        obj.pop(key1, None)


# Define type variables for generic key and value types
KT = TypeVar("KT")  # Key type variable
VT = TypeVar("VT")  # Value type variable


class ThreadSafeDict(Generic[KT, VT]):
    """
    A thread-safe generic dictionary implementation.
    Supports all basic dictionary operations with proper synchronization
    using a reentrant lock, and maintains type safety through generics.
    """

    def __init__(self) -> None:
        """Initialize an empty thread-safe dictionary with an internal lock."""
        self._storage: dict[KT, VT] = {}  # Underlying storage structure
        self._lock = threading.RLock()  # Reentrant lock for synchronization

    def __setitem__(self, key: KT, value: VT) -> None:
        """
        Thread-safe implementation of dictionary item assignment.
        Equivalent to dict[key] = value.

        Args:
            key: The key to associate with the value
            value: The value to store
        """
        with self._lock:
            self._storage[key] = value

    def __getitem__(self, key: KT) -> VT:
        """
        Thread-safe implementation of dictionary item retrieval.
        Equivalent to dict[key].

        Args:
            key: The key to look up

        Returns:
            The value associated with the key

        Raises:
            KeyError: If the key is not found
        """
        with self._lock:
            return self._storage[key]

    def __delitem__(self, key: KT) -> None:
        """
        Thread-safe implementation of dictionary item deletion.
        Equivalent to del dict[key].

        Args:
            key: The key to remove

        Raises:
            KeyError: If the key is not found
        """
        with self._lock:
            del self._storage[key]

    def get(self, key: KT, default: VT | None = None) -> VT | None:
        """
        Thread-safe implementation of dict.get().

        Args:
            key: The key to look up
            default: Value to return if key is not found (default: None)

        Returns:
            The value associated with the key, or default if not found
        """
        with self._lock:
            return self._storage.get(key, default)

    def setdefault(self, key: KT, default: VT) -> VT:
        """
        Thread-safe implementation of dict.setdefault().
        Inserts key with default value if key is not present.

        Args:
            key: The key to check/insert
            default: Value to insert if key is not found

        Returns:
            The existing value or the inserted default value
        """
        with self._lock:
            return self._storage.setdefault(key, default)

    def update(self, items: Iterable[tuple[KT, VT]]) -> None:
        """
        Thread-safe implementation of dict.update().
        Updates dictionary with multiple key-value pairs.

        Args:
            items: Iterable of (key, value) tuples to add/update
        """
        with self._lock:
            self._storage.update(items)

    def pop(self, key: KT, default: VT | None = None) -> VT | None:
        """
        Thread-safe implementation of dict.pop().
        Removes and returns value associated with key.

        Args:
            key: The key to remove
            default: Value to return if key is not found (optional)

        Returns:
            The removed value or default if key not found

        Raises:
            KeyError: If key not found and no default provided
        """
        with self._lock:
            return self._storage.pop(key, default)

    def __contains__(self, key: KT) -> bool:
        """
        Thread-safe implementation of 'key in dict' check.

        Args:
            key: The key to check for existence

        Returns:
            True if key exists, False otherwise
        """
        with self._lock:
            return key in self._storage

    def __len__(self) -> int:
        """
        Thread-safe implementation of len(dict).

        Returns:
            Number of key-value pairs in the dictionary
        """
        with self._lock:
            return len(self._storage)

    def clear(self) -> None:
        """Thread-safe implementation of dict.clear(). Removes all items."""
        with self._lock:
            self._storage.clear()

    def keys(self) -> list[KT]:
        """
        Thread-safe implementation of dict.keys().
        Returns a copy of all keys to prevent concurrent modification issues.

        Returns:
            List of all keys in the dictionary
        """
        with self._lock:
            return list(self._storage.keys())

    def values(self) -> list[VT]:
        """
        Thread-safe implementation of dict.values().
        Returns a copy of all values to prevent concurrent modification issues.

        Returns:
            List of all values in the dictionary
        """
        with self._lock:
            return list(self._storage.values())

    def items(self) -> list[tuple[KT, VT]]:
        """
        Thread-safe implementation of dict.items().
        Returns a copy of all key-value pairs to prevent concurrent modification issues.

        Returns:
            List of (key, value) tuples
        """
        with self._lock:
            return list(self._storage.items())

    def __str__(self) -> str:
        """
        Thread-safe string representation.

        Returns:
            String representation of the dictionary
        """
        with self._lock:
            return str(self._storage)

    def __repr__(self) -> str:
        """
        Thread-safe representation for debugging.

        Returns:
            Debug-friendly string representation
        """
        with self._lock:
            return f"ThreadSafeDict({self._storage!r})"

    # ------------------------------
    # Critical: JSON serialization support
    # ------------------------------
    def to_dict(self) -> dict[KT, VT]:
        """Convert ThreadSafeDict to a standard Python dict (thread-safe)."""
        with self._lock:
            return self._storage.copy()  # Return a copy of internal data
