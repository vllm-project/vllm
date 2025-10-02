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
from collections.abc import Iterable, Mapping, Sequence, Set
from dataclasses import MISSING, Field, field, fields, is_dataclass, replace
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

import regex as re
from typing_extensions import runtime_checkable

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
    if callable(x):
        raise TypeError(
            "normalize_value: function or callable instance unsupported")

    # Torch dtype: stringify (torch.float64 -> "torch.float64").
    # Collisions with same literal ok; tag as ("torch.dtype", str(x)).
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
                f"({type(value).__name__})") from e
    return factors


def hash_factors(items: dict[str, object]) -> str:
    """Return a SHA-256 hex digest of the canonical items structure."""
    return hashlib.sha256(json.dumps(items,
                                     sort_keys=True).encode()).hexdigest()


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


def getattr_iter(object: object, names: Iterable[str], default: Any) -> Any:
    """
    A helper function that retrieves an attribute from an object which may
    have multiple possible names. This is useful when fetching attributes from
    arbitrary `transformers.PretrainedConfig` instances.
    """
    for name in names:
        if hasattr(object, name):
            return getattr(object, name)
    return default


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
    pattern = r'at 0x[a-fA-F0-9]{2,16}>'
    match = re.search(pattern, text)
    return match is not None


def assert_hashable(text: str) -> bool:
    if not contains_object_print(text):
        return True
    raise AssertionError(
        f"vLLM tried to hash some configs that may have Python objects ids "
        f"in them. This is a bug, please file an issue. "
        f"Text being hashed: {text}")


def get_attr_docs(cls: type[Any]) -> dict[str, str]:
    """
    Get any docstrings placed after attribute assignments in a class body.

    https://davidism.com/mit-license/
    """

    def pairwise(iterable):
        """
        Manually implement itertools.pairwise.

        See: https://docs.python.org/3/library/itertools.html
        #itertools.pairwise

        Can be removed when Python 3.9 support is dropped.
        """
        iterator = iter(iterable)
        a = next(iterator, None)

        for b in iterator:
            yield a, b
            a = b

    try:
        cls_node = ast.parse(textwrap.dedent(inspect.getsource(cls))).body[0]
    except (OSError, KeyError, TypeError):
        # HACK: Python 3.13+ workaround - set missing __firstlineno__
        # Workaround can be removed after we upgrade to pydantic==2.12.0
        with open(inspect.getfile(cls)) as f:
            for i, line in enumerate(f):
                if f"class {cls.__name__}" in line and ":" in line:
                    cls.__firstlineno__ = i + 1
                    break
        cls_node = ast.parse(textwrap.dedent(inspect.getsource(cls))).body[0]

    if not isinstance(cls_node, ast.ClassDef):
        raise TypeError("Given object was not a class.")

    out = {}

    # Consider each pair of nodes.
    for a, b in pairwise(cls_node.body):
        # Must be an assignment then a constant string.
        if (not isinstance(a, (ast.Assign, ast.AnnAssign))
                or not isinstance(b, ast.Expr)
                or not isinstance(b.value, ast.Constant)
                or not isinstance(b.value.value, str)):
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

    def compute_hash(self) -> str:
        ...


class SupportsMetricsInfo(Protocol):

    def metrics_info(self) -> dict[str, str]:
        ...


def update_config(config: ConfigT, overrides: dict[str, Any]) -> ConfigT:
    processed_overrides = {}
    for field_name, value in overrides.items():
        assert hasattr(
            config, field_name), f"{type(config)} has no field `{field_name}`"
        current_value = getattr(config, field_name)
        if is_dataclass(current_value) and not is_dataclass(value):
            assert isinstance(value, dict), (
                f"Overrides to {type(config)}.{field_name} must be a dict"
                f"  or {type(current_value)}, but got {type(value)}")
            value = update_config(
                current_value,  # type: ignore[type-var]
                value)
        processed_overrides[field_name] = value
    return replace(config, **processed_overrides)
