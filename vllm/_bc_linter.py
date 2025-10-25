# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# vllm/_bc_linter.py
from collections.abc import Callable
from typing import Any, TypeVar, overload

T = TypeVar("T")


@overload
def bc_linter_skip(obj: T) -> T: ...


@overload
def bc_linter_skip(*, reason: str | None = ...) -> Callable[[T], T]: ...


def bc_linter_skip(obj: Any = None, *, reason: str | None = None):
    """
    No-op decorator to mark symbols/files for BC-linter suppression.

    Usage:
        @bc_linter_skip
        def legacy_api(...): ...
    """

    def _wrap(x: T) -> T:
        return x

    return _wrap if obj is None else obj


@overload
def bc_linter_include(obj: T) -> T: ...


@overload
def bc_linter_include(*, reason: str | None = ...) -> Callable[[T], T]: ...


def bc_linter_include(obj: Any = None, *, reason: str | None = None):
    """
    Usage:
        @bc_linter_include
        def public_api(...): ...
    """

    def _wrap(x: T) -> T:
        return x

    return _wrap if obj is None else obj


__all__ = ["bc_linter_skip", "bc_linter_include"]
