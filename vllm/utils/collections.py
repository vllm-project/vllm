# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Contains helpers that are applied to collections.

This is similar in concept to the `collections` module.
"""

from collections import UserDict, defaultdict
from collections.abc import Callable, Generator, Hashable, Iterable, Mapping
from typing import Generic, Literal, TypeVar

from typing_extensions import TypeIs, assert_never

T = TypeVar("T")
U = TypeVar("U")

_K = TypeVar("_K", bound=Hashable)
_V = TypeVar("_V")


class ClassRegistry(UserDict[type[T], _V]):
    """
    A registry that acts like a dictionary but searches for other classes
    in the MRO if the original class is not found.
    """

    def __getitem__(self, key: type[T]) -> _V:
        for cls in key.mro():
            if cls in self.data:
                return self.data[cls]

        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        return self.contains(key)

    def contains(self, key: object, *, strict: bool = False) -> bool:
        if not isinstance(key, type):
            return False

        if strict:
            return key in self.data

        return any(cls in self.data for cls in key.mro())


class LazyDict(Mapping[str, T], Generic[T]):
    """
    Evaluates dictionary items only when they are accessed.

    Adapted from: https://stackoverflow.com/a/47212782/5082708
    """

    def __init__(self, factory: dict[str, Callable[[], T]]):
        self._factory = factory
        self._dict: dict[str, T] = {}

    def __getitem__(self, key: str) -> T:
        if key not in self._dict:
            if key not in self._factory:
                raise KeyError(key)
            self._dict[key] = self._factory[key]()
        return self._dict[key]

    def __setitem__(self, key: str, value: Callable[[], T]):
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
