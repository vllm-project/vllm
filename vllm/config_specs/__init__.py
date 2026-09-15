# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config field declarations without runtime initialization."""

from dataclasses import MISSING, Field, fields
from types import UnionType
from typing import Any, Literal, NamedTuple, Union, get_args, get_origin


class CLI(NamedTuple):
    """CLI exposure of a field; unspecified defaults use its declared default."""

    order: int
    dest: str | None = None
    default: Any = MISSING
    human_readable: bool = False
    flags: tuple[str, ...] = ()
    python_after: str | None = None


def _validate_cli_annotation(cls: type, field: Field) -> None:
    annotation = field.type
    origin = get_origin(annotation)
    supported_scalars = (bool, int, float, str)
    supported = any(annotation is scalar for scalar in supported_scalars)

    if origin is Literal:
        values = get_args(annotation)
        supported = bool(values) and type(values[0]) in (int, str)
        supported = supported and all(
            type(value) is type(values[0]) for value in values
        )
    elif origin in (Union, UnionType):
        members = get_args(annotation)
        non_null = tuple(member for member in members if member is not type(None))
        nullable = len(non_null) != len(members)
        supported = (
            nullable
            and len(non_null) == 1
            and any(non_null[0] is scalar for scalar in supported_scalars)
        ) or (
            len(non_null) == 2
            and any(member is str for member in non_null)
            and any(member == type[object] for member in non_null)
        )

    if not supported:
        raise ValueError(
            f"Unsupported CLI annotation for {cls.__name__}.{field.name}: "
            f"{annotation!r}"
        )


def cli_fields(cls: type) -> list[tuple[Field, CLI]]:
    exposed = sorted(
        [
            (field, field.metadata["cli"])
            for field in fields(cls)
            if "cli" in field.metadata
        ],
        key=lambda pair: pair[1].order,
    )
    for field, _ in exposed:
        _validate_cli_annotation(cls, field)
    return exposed


def cli_default(cls: type, name: str) -> Any:
    field = {field.name: field for field in fields(cls)}[name]
    declaration = field.metadata["cli"]
    if declaration.default is not MISSING:
        return declaration.default
    if field.default is MISSING:
        raise ValueError(f"{cls.__name__}.{name} needs an explicit CLI default")
    return field.default


def runtime_values(cls: type, values: Any) -> dict[str, Any]:
    """Map resolved CLI/EngineArgs values to their runtime field names."""
    return {
        field.name: getattr(values, declaration.dest or field.name)
        for field, declaration in cli_fields(cls)
    }
