# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Structured result of one logical weight-loader invocation."""

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import TypeAlias

FragmentScalar: TypeAlias = str | int | float | bool
FragmentValue: TypeAlias = FragmentScalar | tuple[FragmentScalar, ...]


class LoadCollisionPolicy(str, Enum):
    """Whether more than one source may claim the same target fragment."""

    UNIQUE = "unique"
    OVERWRITE = "overwrite"


@dataclass(frozen=True)
class LoadFragment:
    """Stable logical identity of a target slice.

    Items are stored in caller-provided schema order. Tensor values and other
    process-local objects are intentionally unsupported.
    """

    items: tuple[tuple[str, FragmentValue], ...] = ()

    @classmethod
    def from_fields(cls, **fields: FragmentValue | None) -> "LoadFragment":
        return cls(
            tuple(
                (name, value)
                for name, value in fields.items()
                if value is not None
            )
        )

    def format(self) -> str:
        return ",".join(f"{name}={value!r}" for name, value in self.items)


@dataclass(frozen=True)
class LoadReceipt:
    """Whether a loader consumed a source tensor and which fragment it wrote."""

    consumed: bool = True
    fragment: LoadFragment = LoadFragment()
    collision_policy: LoadCollisionPolicy = LoadCollisionPolicy.UNIQUE

    def __bool__(self) -> bool:
        """Preserve compatibility with conditional loaders returning bool."""
        return self.consumed

    @classmethod
    def accepted(
        cls,
        *,
        collision_policy: LoadCollisionPolicy = LoadCollisionPolicy.UNIQUE,
        **fragment: FragmentValue | None,
    ) -> "LoadReceipt":
        return cls(
            consumed=True,
            fragment=LoadFragment.from_fields(**fragment),
            collision_policy=collision_policy,
        )

    @classmethod
    def skipped(
        cls,
        *,
        collision_policy: LoadCollisionPolicy = LoadCollisionPolicy.UNIQUE,
        **fragment: FragmentValue | None,
    ) -> "LoadReceipt":
        return cls(
            consumed=False,
            fragment=LoadFragment.from_fields(**fragment),
            collision_policy=collision_policy,
        )


def returns_load_receipt(*fragment_arguments: str):
    """Declare a loader's fragment schema and return a structured receipt.

    This adapter lets an existing None/bool loader migrate without duplicating
    receipt construction at every return branch. Conditional loaders retain
    their historical truthiness through ``LoadReceipt.__bool__``.
    """

    def decorate(loader: Callable) -> Callable:
        signature = inspect.signature(loader)

        @wraps(loader)
        def wrapped(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            result = loader(*args, **kwargs)
            if isinstance(result, LoadReceipt):
                return result
            consumed = (
                result is True
                if bound.arguments.get("return_success") is True
                else True
            )
            fragment = {
                name: bound.arguments.get(name)
                for name in fragment_arguments
                if bound.arguments.get(name) is not None
            }
            return LoadReceipt(
                consumed=consumed,
                fragment=LoadFragment.from_fields(**fragment),
            )

        wrapped.load_receipt_fragment_arguments = fragment_arguments
        return wrapped

    return decorate


__all__ = [
    "FragmentValue",
    "LoadCollisionPolicy",
    "LoadFragment",
    "LoadReceipt",
    "returns_load_receipt",
]
