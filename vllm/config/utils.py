# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
