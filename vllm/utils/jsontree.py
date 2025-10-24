# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helper functions to work with nested JSON structures."""

from collections.abc import Callable, Iterable
from functools import reduce
from typing import TYPE_CHECKING, TypeAlias, TypeVar, cast, overload

if TYPE_CHECKING:
    import torch

    from vllm.multimodal.inputs import BatchedTensorInputs

_T = TypeVar("_T")
_U = TypeVar("_U")

JSONTree: TypeAlias = (
    dict[str, "JSONTree[_T]"] | list["JSONTree[_T]"] | tuple["JSONTree[_T]", ...] | _T
)
"""A nested JSON structure where the leaves need not be JSON-serializable."""

_JSONTree: TypeAlias = (
    dict[str, "JSONTree[_T]"]
    | list["JSONTree[_T]"]
    | tuple["JSONTree[_T]", ...]
    | dict[str, _T]
    | list[_T]
    | tuple[_T, ...]
    | _T
)
"""
Same as `JSONTree` but with additional `Union` members to satisfy overloads.
"""


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


@overload
def json_map_leaves(
    func: Callable[["torch.Tensor"], "torch.Tensor"],
    value: "BatchedTensorInputs",
) -> "BatchedTensorInputs": ...


@overload
def json_map_leaves(
    func: Callable[[_T], _U],
    value: _T | dict[str, _T],
) -> _U | dict[str, _U]: ...


@overload
def json_map_leaves(
    func: Callable[[_T], _U],
    value: _T | list[_T],
) -> _U | list[_U]: ...


@overload
def json_map_leaves(
    func: Callable[[_T], _U],
    value: _T | tuple[_T, ...],
) -> _U | tuple[_U, ...]: ...


@overload
def json_map_leaves(
    func: Callable[[_T], _U],
    value: JSONTree[_T],
) -> JSONTree[_U]: ...


def json_map_leaves(
    func: Callable[[_T], _U],
    value: "BatchedTensorInputs" | _JSONTree[_T],
) -> "BatchedTensorInputs" | _JSONTree[_U]:
    """Apply a function to each leaf in a nested JSON structure."""
    if isinstance(value, dict):
        return {
            k: json_map_leaves(func, v)  # type: ignore[arg-type]
            for k, v in value.items()
        }
    elif isinstance(value, list):
        return [json_map_leaves(func, v) for v in value]
    elif isinstance(value, tuple):
        return tuple(json_map_leaves(func, v) for v in value)
    else:
        return func(value)


@overload
def json_reduce_leaves(
    func: Callable[[_T, _T], _T],
    value: _T | dict[str, _T],
    /,
) -> _T: ...


@overload
def json_reduce_leaves(
    func: Callable[[_T, _T], _T],
    value: _T | list[_T],
    /,
) -> _T: ...


@overload
def json_reduce_leaves(
    func: Callable[[_T, _T], _T],
    value: _T | tuple[_T, ...],
    /,
) -> _T: ...


@overload
def json_reduce_leaves(
    func: Callable[[_T, _T], _T],
    value: JSONTree[_T],
    /,
) -> _T: ...


@overload
def json_reduce_leaves(
    func: Callable[[_U, _T], _U],
    value: JSONTree[_T],
    initial: _U,
    /,
) -> _U: ...


def json_reduce_leaves(
    func: Callable[..., _T | _U],
    value: _JSONTree[_T],
    initial: _U = cast(_U, ...),  # noqa: B008
    /,
) -> _T | _U:
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


def json_count_leaves(value: JSONTree[_T]) -> int:
    """Count the number of leaves in a nested JSON structure."""
    return sum(1 for _ in json_iter_leaves(value))
