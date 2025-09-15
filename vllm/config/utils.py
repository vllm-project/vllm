# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import MISSING, Field, field, fields, is_dataclass
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

    ConfigType = type[DataclassInstance]
else:
    ConfigType = type

ConfigT = TypeVar("ConfigT", bound=ConfigType)


def config(cls: ConfigT) -> ConfigT:
    """
    A decorator that ensures all fields in a dataclass have default values
    and that each field has a docstring.

    If a `ConfigT` is used as a CLI argument itself, the `type` keyword argument
    provided by `get_kwargs` will be
    `pydantic.TypeAdapter(ConfigT).validate_json(cli_arg)` which treats the
    `cli_arg` as a JSON string which gets validated by `pydantic`.

    Config validation is performed by the tools/validate_config.py
    script, which is invoked during the pre-commit checks.
    """
    return cls


def get_field(cls: ConfigType, name: str) -> Field:
    """Get the default factory field of a dataclass by name. Used for getting
    default factory fields in `EngineArgs`."""
    if not is_dataclass(cls):
        raise TypeError("The given class is not a dataclass.")
    cls_fields = {f.name: f for f in fields(cls)}
    if name not in cls_fields:
        raise ValueError(f"Field '{name}' not found in {cls.__name__}.")
    named_field: Field = cls_fields[name]
    if (default_factory := named_field.default_factory) is not MISSING:
        return field(default_factory=default_factory)
    if (default := named_field.default) is not MISSING:
        return field(default=default)
    raise ValueError(
        f"{cls.__name__}.{name} must have a default value or default factory.")
