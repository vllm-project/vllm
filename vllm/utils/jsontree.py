# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helper functions to work with nested JSON structures."""
from collections.abc import Iterable
from functools import reduce
from typing import Callable, TypeVar, Union, overload

_T = TypeVar("_T")
_U = TypeVar("_U")

JSONTree = Union[dict[str, "JSONTree[_T]"], list["JSONTree[_T]"],
                 tuple["JSONTree[_T]", ...], _T]
"""A nested JSON structure where the leaves need not be JSON-serializable."""


def json_iter_leaves(value: JSONTree[_T]) -> Iterable[_T]:
    """Iterate through each leaf in a nested JSON structure."""
    if isinstance(value, dict):
        for v in value.values():
            yield from json_iter_leaves(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from json_iter_leaves(v)
    else:
        yield value


def json_map_leaves(
    func: Callable[[_T], _U],
    value: JSONTree[_T],
) -> JSONTree[_U]:
    """Apply a function to each leaf in a nested JSON structure."""
    if isinstance(value, dict):
        return {k: json_map_leaves(func, v) for k, v in value.items()}
    elif isinstance(value, list):
        return [json_map_leaves(func, v) for v in value]
    elif isinstance(value, tuple):
        return tuple(json_map_leaves(func, v) for v in value)
    else:
        return func(value)


@overload
def json_reduce_leaves(
    func: Callable[[_T, _T], _T],
    value: JSONTree[_T],
    /,
) -> _T:
    ...


@overload
def json_reduce_leaves(
    func: Callable[[_U, _T], _U],
    value: JSONTree[_T],
    initial: _U,
    /,
) -> _U:
    ...


def json_reduce_leaves(
    func: Callable[..., Union[_T, _U]],
    value: JSONTree[_T],
    initial: _U = ...,  # type: ignore[assignment]
    /,
) -> Union[_T, _U]:
    """
    Apply a function of two arguments cumulatively to each leaf in a
    nested JSON structure, from left to right, so as to reduce the
    sequence to a single value.
    """
    if initial is ...:
        return reduce(func, json_iter_leaves(value))  # type: ignore[arg-type]

    return reduce(
        func,  # type: ignore[arg-type]
        json_iter_leaves(value),
        initial,
    )
