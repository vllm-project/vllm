# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import hashlib
import json
import pathlib
from collections.abc import Mapping, Sequence, Set
from contextlib import suppress
from dataclasses import fields
from typing import TYPE_CHECKING, TypeVar

from vllm.logger import init_logger

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

    ConfigType = type[DataclassInstance]
else:
    ConfigType = type

ConfigT = TypeVar("ConfigT", bound=ConfigType)

logger = init_logger(__name__)


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


def normalize_value(x):
    """Return a stable, JSON-serializable canonical form for hashing.

    Order: primitives, special types (Enum, callable, torch.dtype, Path), then
    generic containers (Mapping/Set/Sequence) with recursion.
    """
    # Fast path
    if x is None or isinstance(x, (bool, int, float, str)):
        return x

    # Enums by value
    if isinstance(x, enum.Enum):
        return x.value

    # Callable by qualified name
    try:
        if callable(x):
            module = getattr(x, "__module__", "")
            qual = getattr(x, "__qualname__", repr(x))
            return f"{module}.{qual}" if module else qual
    except Exception:
        pass

    # Torch dtype without hard dependency
    try:
        import torch
        if isinstance(x, torch.dtype):
            return str(x)
    except Exception:
        pass

    # Bytes
    if isinstance(x, (bytes, bytearray)):
        return x.hex()

    # Paths (canonicalize)
    if isinstance(x, pathlib.Path):
        try:
            return str(x.expanduser().resolve())
        except Exception:
            return str(x)

    # Containers (generic)
    if isinstance(x, Mapping):
        return tuple(sorted(
            (str(k), normalize_value(v)) for k, v in x.items()))
    if isinstance(x, Set):
        return tuple(sorted(repr(normalize_value(v)) for v in x))
    if isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray)):
        return tuple(normalize_value(v) for v in x)

    # Nested configs which provide a uuid() method
    if hasattr(x, "uuid") and callable(x.uuid):
        return x.uuid()

    # PretrainedConfig
    if hasattr(x, "to_json_string") and callable(x.to_json_string):
        return x.to_json_string()

    # Unsupported type
    with suppress(Exception):
        logger.debug("normalize_value: unsupported type '%s'",
                     type(x).__name__)
    raise TypeError


def get_hash_factors(config: ConfigT,
                     ignored_factors: set[str]) -> dict[str, object]:
    """Gets the factors used for hashing a config class.

    - Includes all dataclass fields not in `ignored_factors`.
    - Skips non-normalizable values.
    """
    factors: dict[str, object] = {}
    for field in fields(config):
        factor = field.name
        if factor in ignored_factors:
            continue
        value = getattr(config, factor, None)
        try:
            factors[factor] = normalize_value(value)
        except TypeError:
            # Warn once per key to surface potential under-hashing. If this is
            # expected, add the key to `ignored_factors` explicitly.
            with suppress(Exception):
                logger.warning(
                    "Hash skip: unsupported type for key '%s' — add to "
                    "ignored_factors to silence",
                    factor,
                )
            continue
    return factors


def hash_factors(items: dict[str, object]) -> str:
    """Return a SHA-256 hex digest of the canonical items structure."""
    return hashlib.sha256(json.dumps(items,
                                     sort_keys=True).encode()).hexdigest()
