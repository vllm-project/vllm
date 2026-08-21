# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Structured result of one logical weight-loader invocation."""

from dataclasses import dataclass
from typing import TypeAlias

FragmentScalar: TypeAlias = str | int | float | bool
FragmentValue: TypeAlias = FragmentScalar | tuple[FragmentScalar, ...]


@dataclass(frozen=True, order=True)
class LoadFragment:
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
    consumed: bool = True
    fragment: LoadFragment = LoadFragment()

    def __bool__(self) -> bool:
        return self.consumed
