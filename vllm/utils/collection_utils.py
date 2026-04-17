# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Contains helpers that are applied to collections.

This is similar in concept to the `collections` module.
"""

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
