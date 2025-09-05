# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import hashlib
import json
import pathlib
from collections.abc import Mapping, Sequence, Set
from dataclasses import fields
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

    If a `ConfigT` is used as a CLI argument itself, the `type`
    keyword argument provided by `get_kwargs` will be
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

    # Enums by value (normalize underlying value recursively)
    if isinstance(x, enum.Enum):
        return normalize_value(x.value)

    # Callables: classes, functions (incl. lambdas/nested), callable instances
    if callable(x):
        fn = getattr(x, "__func__", x)  # bound methods -> function
        code = getattr(fn, "__code__", None)
        module = getattr(fn, "__module__", "")
        qual = getattr(fn, "__qualname__", getattr(fn, "__name__", ""))
        base = ".".join([p for p in (module, qual) if p]) or repr(fn)
        if code is not None:
            fname = pathlib.Path(getattr(code, "co_filename", "")).name
            lineno = getattr(code, "co_firstlineno", None)
            if fname and isinstance(lineno, int):
                return f"{base}@{fname}:{lineno}"
        # classes or callable instances (no __code__)
        t = x if isinstance(x, type) else type(x)
        t_module = getattr(t, "__module__", "")
        t_qual = getattr(t, "__qualname__", getattr(t, "__name__", ""))
        ident = ".".join([p for p in (t_module, t_qual) if p])
        return ident or repr(t)

    # Torch dtype without import (identify by type module/name)
    t = type(x)
    if getattr(t, "__module__", "") == "torch" and getattr(t, "__name__",
                                                           "") == "dtype":
        return str(x)

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

    # Unsupported type: e.g., modules, generators, open files, or objects
    # without a stable JSON/UUID representation. Hard-error to avoid
    # under-hashing.
    raise TypeError(f"normalize_value: unsupported type '{type(x).__name__}'")


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
        except TypeError as e:
            raise TypeError(
                f"get_hash_factors: unsupported type for key '{factor}' "
                f"({type(value).__name__})") from e
    return factors


def hash_factors(items: dict[str, object]) -> str:
    """Return a SHA-256 hex digest of the canonical items structure."""
    return hashlib.sha256(json.dumps(items,
                                     sort_keys=True).encode()).hexdigest()
