# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import Any


class lazy:
    """Wrap a zero-argument callable evaluated only during log formatting."""

    __slots__ = ("_factory",)

    def __init__(self, factory: Callable[[], Any]) -> None:
        self._factory = factory

    def __str__(self) -> str:
        return str(self._factory())

    def __repr__(self) -> str:
        return str(self)
