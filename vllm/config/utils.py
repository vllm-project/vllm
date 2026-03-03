# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility functions for vLLM config dataclasses."""

import ast
import enum
import hashlib
import inspect
import json
import pathlib
import textwrap
from collections.abc import Callable, Mapping, Sequence, Set
from dataclasses import MISSING, Field, dataclass, field, fields, is_dataclass, replace
from itertools import pairwise
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

import regex as re
import torch
from pydantic.fields import FieldInfo
from typing_extensions import runtime_checkable

from vllm.logger import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    from _typeshed import DataclassInstance
else:
    DataclassInstance = Any

ConfigType = type[DataclassInstance]
ConfigT = TypeVar("ConfigT", bound=ConfigType)


def config(cls: ConfigT) -> ConfigT:
    """
    A decorator that ensures all fields in a dataclass have default values
    and that each field has a docstring.

    If a `ConfigT` is used as a CLI argument itself, the `type` keyword argument
    provided by `get_kwargs` will be
    `pydantic.TypeAdapter(ConfigT).validate_json(cli_arg)` which treats the
    `cli_arg` as a JSON string which gets validated by `pydantic`.

    Config validation is performed by the tools/pre_commit/validate_config.py
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
        if isinstance(default, FieldInfo):
            # Handle pydantic.Field defaults
            if default.default_factory is not None:
                return field(default_factory=default.default_factory)
            else:
                default = default.default
        return field(default=default)

    raise ValueError(
        f"{cls.__name__}.{name} must have a default value or default factory."
    )


def getattr_iter(
    object: object,
    names: Sequence[str],
    default: Any | None = None,
    default_factory: Callable[[], Any] | None = None,
    warn: bool = False,
) -> Any:
    """
    A helper function that retrieves an attribute from an object which may
    have multiple possible names. This is useful when fetching attributes from
    arbitrary `transformers.PretrainedConfig` instances.

    In the case where the first name in `names` is the preferred name, and
    any other names are deprecated aliases, setting `warn=True` will log a
    warning when a deprecated name is used.
    """
    for i, name in enumerate(names):
        if hasattr(object, name):
            if warn and i > 0:
                logger.warning_once(
                    "%s contains a deprecated attribute name '%s'. "
                    "Please use the preferred attribute name '%s' instead.",
                    type(object).__name__,
                    name,
                    names[0],
                )
            return getattr(object, name)
    return default_factory() if default_factory is not None else default


def contains_object_print(text: str) -> bool:
    """
    Check if the text looks like a printed Python object, e.g.
    contains any substring matching the pattern: "at 0xFFFFFFF>"
    We match against 0x followed by 2-16 hex chars (there's
    a max of 16 on a 64-bit system).

    Args:
        text (str): The text to check

    Returns:
        result (bool): `True` if a match is found, `False` otherwise.
    """
    pattern = r"at 0x[a-fA-F0-9]{2,16}>"
    match = re.search(pattern, text)
    return match is not None


def assert_hashable(text: str) -> bool:
    if not contains_object_print(text):
        return True
    raise AssertionError(
        f"vLLM tried to hash some configs that may have Python objects ids "
        f"in them. This is a bug, please file an issue. "
        f"Text being hashed: {text}"
    )


def get_attr_docs(cls: type[Any]) -> dict[str, str]:
    """
    Get any docstrings placed after attribute assignments in a class body.

    https://davidism.com/mit-license/
    """

    cls_node = ast.parse(textwrap.dedent(inspect.getsource(cls))).body[0]

    if not isinstance(cls_node, ast.ClassDef):
        raise TypeError("Given object was not a class.")

    out = {}

    # Consider each pair of nodes.
    for a, b in pairwise(cls_node.body):
        # Must be an assignment then a constant string.
        if (
            not isinstance(a, (ast.Assign, ast.AnnAssign))
            or not isinstance(b, ast.Expr)
            or not isinstance(b.value, ast.Constant)
            or not isinstance(b.value.value, str)
        ):
            continue

        doc = inspect.cleandoc(b.value.value)

        # An assignment can have multiple targets (a = b = v), but an
        # annotated assignment only has one target.
        targets = a.targets if isinstance(a, ast.Assign) else [a.target]

        for target in targets:
            # Must be assigning to a plain name.
            if not isinstance(target, ast.Name):
                continue

            out[target.id] = doc

    return out


def is_init_field(cls: ConfigType, name: str) -> bool:
    return next(f for f in fields(cls) if f.name == name).init


@runtime_checkable
class SupportsHash(Protocol):
    def compute_hash(self) -> str: ...


class SupportsMetricsInfo(Protocol):
    def metrics_info(self) -> dict[str, str]: ...


def update_config(config: ConfigT, overrides: dict[str, Any]) -> ConfigT:
    processed_overrides = {}
    for field_name, value in overrides.items():
        assert hasattr(config, field_name), (
            f"{type(config)} has no field `{field_name}`"
        )
        current_value = getattr(config, field_name)
        if is_dataclass(current_value) and not is_dataclass(value):
            assert isinstance(value, dict), (
                f"Overrides to {type(config)}.{field_name} must be a dict"
                f"  or {type(current_value)}, but got {type(value)}"
            )
            value = update_config(
                current_value,  # type: ignore[type-var]
                value,
            )
        processed_overrides[field_name] = value
    return replace(config, **processed_overrides)


def normalize_value(x):
    """Return a stable, JSON-serializable canonical form for hashing.
    Order: primitives, special types (Enum, callable, torch.dtype, Path), then
    generic containers (Mapping/Set/Sequence) with recursion.
    """
    # Fast path
    if x is None or isinstance(x, (bool, int, float, str)):
        return x

    # Enums: tag with FQN to avoid primitive collisions.
    # Ex: Enum(1) vs int(1) -> ("module.QualName", value).
    if isinstance(x, enum.Enum):
        enum_type = f"{x.__class__.__module__}.{x.__class__.__qualname__}"
        return (enum_type, normalize_value(x.value))

    # Classes (types) are accepted and canonicalized by their fully-qualified
    # name (module.qualname) for a stable identifier.
    # Instances are only accepted if they expose uuid(); otherwise they are
    # rejected to avoid under-hashing object state.

    # Callables: accept classes only; reject funcs/lambdas/methods.
    # Used by LogitsProcessor types and ModelConfig.hf_overrides.
    if isinstance(x, type):
        module = getattr(x, "__module__", "")
        qual = getattr(x, "__qualname__", getattr(x, "__name__", ""))
        return ".".join([p for p in (module, qual) if p]) or repr(x)

    # Prefer stable uuid identifiers for objects that provide them, even if
    # they are callable instances (e.g., InductorPass wrappers).
    if hasattr(x, "uuid") and callable(getattr(x, "uuid", None)):
        return x.uuid()

    if callable(x):
        raise TypeError("normalize_value: function or callable instance unsupported")

    # Torch dtype: stringify (torch.float64 -> "torch.float64").
    # We rely on the string form here; dtype-bearing fields that need additional
    # disambiguation should encode that at the config layer.
    if isinstance(x, torch.dtype):
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

    # Dataclasses: represent as (FQN, sorted(field,value) tuple) for stability.
    if is_dataclass(x):
        type_fqn = f"{x.__class__.__module__}.{x.__class__.__qualname__}"
        items = tuple(
            (f.name, normalize_value(getattr(x, f.name)))
            for f in sorted(fields(x), key=lambda f: f.name)
        )
        return (type_fqn, items)

    # Containers (generic)
    if isinstance(x, Mapping):
        return tuple(sorted((str(k), normalize_value(v)) for k, v in x.items()))
    if isinstance(x, Set):
        return tuple(sorted(repr(normalize_value(v)) for v in x))
    if isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray)):
        return tuple(normalize_value(v) for v in x)

    # PretrainedConfig
    if hasattr(x, "to_json_string") and callable(x.to_json_string):
        return x.to_json_string()

    # Unsupported type: e.g., modules, generators, open files, or objects
    # without a stable JSON/UUID representation. Hard-error to avoid
    # under-hashing.
    # If you hit this, either reshape your config to use supported primitives
    # and containers, or extend normalize_value to provide a stable encoding
    # (e.g., via uuid() or to_json_string()) for this type.
    raise TypeError(
        f"normalize_value: unsupported type '{type(x).__name__}'. "
        "Ensure config values use supported primitives/containers or add a "
        "stable representation for this type."
    )


def get_hash_factors(config: ConfigT, ignored_factors: set[str]) -> dict[str, object]:
    """Gets the factors used for hashing a config class.
    - Includes all dataclass fields not in `ignored_factors`.
    - Errors on non-normalizable values.
    """
    factors: dict[str, object] = {}
    for dc_field in fields(config):
        factor = dc_field.name
        if factor in ignored_factors:
            continue
        value = getattr(config, factor, None)
        try:
            factors[factor] = normalize_value(value)
        except TypeError as e:
            raise TypeError(
                f"get_hash_factors: unsupported type for key '{factor}' "
                f"({type(value).__name__})"
            ) from e
    return factors


def hash_factors(items: dict[str, object]) -> str:
    """Return a SHA-256 hex digest of the canonical items structure."""
    return hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()


def handle_deprecated(
    config: ConfigT,
    old_name: str,
    new_name_or_names: str | list[str],
    removal_version: str,
) -> None:
    old_val = getattr(config, old_name)
    if old_val is None:
        return

    if isinstance(new_name_or_names, str):
        new_names = [new_name_or_names]
    else:
        new_names = new_name_or_names

    msg = (
        f"{old_name} is deprecated and will be removed in {removal_version}. "
        f"Use {', '.join(new_names)} instead."
    )
    logger.warning(msg)

    for new_name in new_names:
        setattr(config, new_name, old_val)


@dataclass
class Range:
    """
    A range of numbers.
    Inclusive of start, inclusive of end.
    """

    start: int
    end: int

    def is_single_size(self) -> bool:
        return self.start == self.end

    def __contains__(self, size: int) -> bool:
        # Inclusive of start, inclusive of end
        return self.start <= size <= self.end

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Range):
            return False
        return self.start == other.start and self.end == other.end

    def __hash__(self) -> int:
        return hash((self.start, self.end))

    def __str__(self) -> str:
        return f"({self.start}, {self.end})"

    def __repr__(self) -> str:
        return self.__str__()
