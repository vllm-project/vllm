# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility functions for vLLM config dataclasses."""

import ast
import enum
import hashlib
import inspect
import json
import os
import pathlib
import textwrap
from collections.abc import Callable, Mapping, Sequence, Set
from dataclasses import MISSING, field, fields, is_dataclass
from itertools import pairwise
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
    TypeVar,
    cast,
    overload,
)

import torch
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from pydantic.fields import Field as PydanticField
from pydantic.fields import FieldInfo as PydanticFieldInfo
from typing_extensions import dataclass_transform, runtime_checkable

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    from _typeshed import DataclassInstance
else:
    DataclassInstance = Any

ConfigType = type[DataclassInstance]
ConfigT = TypeVar("ConfigT", bound=DataclassInstance)


@overload
@dataclass_transform(field_specifiers=(PydanticField,))
def config(cls: type[ConfigT]) -> type[ConfigT]: ...


@overload
@dataclass_transform(field_specifiers=(PydanticField,))
def config(
    *, config: ConfigDict | None = None, **kwargs: Any
) -> Callable[[type[ConfigT]], type[ConfigT]]: ...


@dataclass_transform(field_specifiers=(PydanticField,))
def config(
    cls: type[ConfigT] | None = None,
    *,
    config: ConfigDict | None = None,
    **kwargs: Any,
) -> type[ConfigT] | Callable[[type[ConfigT]], type[ConfigT]]:
    """Decorator to create a pydantic dataclass with default config. The default config
    for the dataclass forbids extra fields.

    All config classes in vLLM should use this decorator.

    Args:
        cls: The class to decorate
        config: The pydantic ConfigDict to use. If provided, it will be merged with
            the default config.
        **kwargs: Additional arguments to pass to pydantic.dataclass."""
    # Extra fields are forbidden by default
    merged_config = ConfigDict(extra="forbid")
    if config is not None:
        merged_config.update(config)

    def decorator(cls: type[ConfigT]) -> type[ConfigT]:
        return dataclass(cls, config=merged_config, **kwargs)  # type: ignore[return-value]

    # Called with arguments: @config(config=...)
    if cls is None:
        return decorator
    # Called without arguments: @config
    return decorator(cls)


def get_field(cls: ConfigType, name: str) -> Any:
    """Get the default factory field of a dataclass by name. Used for getting
    default factory fields in `EngineArgs`."""
    if not is_dataclass(cls):
        raise TypeError("The given class is not a dataclass.")
    try:
        named_field = next(f for f in fields(cls) if f.name == name)
    except StopIteration as e:
        raise ValueError(f"Field '{name}' not found in {cls.__name__}.") from e

    # The arguments to copy to the new field
    default = named_field.default
    default_factory = named_field.default_factory
    init = named_field.init

    # Handle pydantic.Field
    if isinstance(default, PydanticFieldInfo):
        if default.init is not None:
            init = default.init
        if default.default_factory is not None:
            default_factory = cast(Callable[[], Any], default.default_factory)
            default = MISSING
        else:
            default = default.default

    if default is MISSING and default_factory is MISSING:
        logger.warning_once(
            "%s.%s has no default or default factory.", cls.__name__, name
        )
    return field(default=default, default_factory=default_factory, init=init)


def is_init_field(cls: ConfigType, name: str) -> bool:
    return get_field(cls, name).init


def replace(dataclass_instance: ConfigT, /, **kwargs) -> ConfigT:
    """Like [`dataclasses.replace`](https://docs.python.org/3/library/dataclasses.html#dataclasses.replace),
    but compatible with Pydantic dataclasses which use `pydantic.fields.Field` instead
    of `dataclasses.field`"""
    cls = type(dataclass_instance)
    dataclass_dict = dataclass_instance.__dict__
    dataclass_dict = {k: v for k, v in dataclass_dict.items() if is_init_field(cls, k)}
    dataclass_dict.update(kwargs)
    return cls(**dataclass_dict)


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

    # Deferred sentinel: a field was never initialized before compute_hash()
    # was called.  This is always a bug — surface it clearly.
    if isinstance(x, _DeferredValue):
        raise TypeError(
            f"normalize_value: encountered uninitialized deferred field "
            f"({x!r}).  Call initialize_deferred_fields() on the containing "
            "config before computing hashes."
        )

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
        try:
            return x.to_json_string()
        except (TypeError, ValueError):
            # to_json_string() may fail for trust-remote-code configs
            # with non-JSON-serializable nested objects. Fall back to
            # normalizing the dict representation recursively.
            if hasattr(x, "to_dict") and callable(x.to_dict):
                return normalize_value(x.to_dict())
            raise

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


def get_from_deprecated_env_if_set(
    env_name: str,
    removal_version: str,
    field_name: str | None = None,
) -> str | None:
    """
    Get value from deprecated environment variable with warning.

    Args:
        env_name: Name of the deprecated environment variable
        removal_version: Version when it will be removed
        field_name: Name of the field to suggest as alternative

    Returns:
        The environment variable value if set, None otherwise
    """
    if envs.is_set(env_name):
        value = os.environ.get(env_name)
        alt_msg = f" Please use {field_name} instead." if field_name else ""
        logger.warning_once(
            "Using %s environment variable is deprecated and will be removed in %s.%s",
            env_name,
            removal_version,
            alt_msg,
        )
        return value
    return None


def set_from_deprecated_env_if_set(
    config: ConfigT,
    env_name: str,
    removal_version: str,
    field_name: str,
    to_bool: bool = False,
    to_int: bool = False,
) -> None:
    """
    Set object field from deprecated environment variable with warning.

    Args:
        config: Config object to set the field on
        env_name: Name of the deprecated environment variable
        removal_version: Version when the env var will be removed
        field_name: Name of the field to set
        to_bool: Whether to convert the environment variable value to boolean
        to_int: Whether to convert the environment variable value to integer
    Returns:
        None
    """
    if to_bool and to_int:
        raise ValueError("Cannot convert to both boolean and integer.")

    env_value = get_from_deprecated_env_if_set(env_name, removal_version, field_name)
    if env_value is not None:
        field_value: str | bool | int = env_value
        if to_bool:
            field_value = env_value.lower() in ("1", "true")
        elif to_int:
            field_value = int(env_value)
        setattr(config, field_name, field_value)


# ---------------------------------------------------------------------------
# Deferred field initialization
#
# Fields whose values cannot be determined at dataclass construction time
# (they depend on being resolved at runtime in VllmConfig.__post_init__)
# are declared as:
#
#     field_name: bool = Deferred(False)
#     field_name: bool = Deferred(lambda cfg: cfg.parallel_config.tp > 1)
#
# Deferred(factory) sets the field's pydantic default to a _DeferredValue
# instance that carries the factory.  The sentinel is falsy so that
# __post_init__ guards like ``if self.fuse_norm_quant:`` do not fire before
# initialization.  VllmConfig calls initialize_deferred_fields() to replace
# every _DeferredValue with its resolved value, and
# validate_deferred_fields_initialized() (auto-discovered at the end of
# VllmConfig.__post_init__) asserts that no sentinels remain.
# ---------------------------------------------------------------------------


class _DeferredValue:
    """Per-field sentinel that doubles as an optional factory carrier.

    Each ``Deferred(...)`` call produces one ``_DeferredValue`` instance
    that becomes the pydantic default for that field.  Embedding the factory
    in the default value means:

    * Detection uses only ``isinstance(value, _DeferredValue)`` on the live
      field value — no dependency on ``FieldInfo.metadata`` or pydantic
      internals.
    * ``__bool__`` returns ``False`` so guards in ``__post_init__`` that read
      deferred fields before initialization treat them as falsy, matching the
      behaviour of the old ``None`` sentinel.
    * When no factory is provided (``MISSING``), the field has no fallback and
      must be set by the user or the optimization-level table; validation will
      fail if it remains unset.
    """

    def __init__(self, factory: "Callable[[Any], Any] | Any" = MISSING) -> None:
        self.factory = factory

    def __bool__(self) -> bool:
        return False

    def __copy__(self) -> "_DeferredValue":
        return self

    def __deepcopy__(self, memo: dict) -> "_DeferredValue":
        # _DeferredValue is a sentinel whose identity matters.  Pydantic
        # deepcopies mutable defaults when creating each model instance;
        # returning self preserves factory identity so that ``is MISSING``
        # checks and __repr__ work correctly on every instance.
        return self

    def __repr__(self) -> str:
        if self.factory is MISSING:
            return "<Deferred()>"
        factory_name = getattr(self.factory, "__name__", repr(self.factory))
        return f"<Deferred({factory_name})>"


def Deferred(factory: "Callable[[Any], Any] | Any" = MISSING) -> Any:
    """Declare a config field whose value is resolved in VllmConfig.__post_init__.

    Returns a pydantic Field whose default is a ``_DeferredValue`` instance.
    The declared annotation is the true runtime type — no ``| None`` needed.
    ``validate_default=False`` tells pydantic to skip coercion of the sentinel
    through the field's type validator.

    When called with no argument, the field has no fallback: it must be set by
    the user or the optimization-level table, or validation will raise.  Pass a
    factory only when a last-resort fallback is genuinely needed (e.g. for
    fields absent from the optimization-level tables).

    Args:
        factory: Optional static fallback value (e.g. ``False``) or a callable
                 that accepts a VllmConfig instance and returns the value.
                 Omit to require the field to be set externally.

    Example::

        @config
        class MyConfig(DeferredFieldsMixin):
            required_flag: bool = Deferred()  # must be set by table/user
            fallback_flag: bool = Deferred(False)  # falls back to False
            tp_flag: bool = Deferred(lambda cfg: cfg.parallel_config.tp > 1)
    """
    return PydanticField(  # type: ignore[call-overload]
        default=_DeferredValue(factory),
        validate_default=False,
    )


class DeferredFieldsMixin:
    """Mixin for config classes that contain Deferred fields.

    Provides initialize_deferred_fields() and validate_deferred_fields_initialized().
    VllmConfig.__post_init__ is responsible for calling both at the appropriate
    points (see vllm/config/vllm.py).

    Usage::

        @config
        class PassConfig(DeferredFieldsMixin):
            fuse_norm_quant: bool = Deferred(False)
    """

    def initialize_deferred_fields(self, vllm_config: Any) -> None:
        """Replace every field still holding a _DeferredValue with its value.

        Fields explicitly set by the caller (e.g. the optimization-level table)
        are left unchanged; only sentinel-valued fields are touched.

        Uses ``dataclasses.fields()`` for discovery so that the mechanism works
        regardless of which pydantic internals (``__pydantic_fields__``,
        ``FieldInfo.metadata``) are available.

        Args:
            vllm_config: The VllmConfig instance passed to factory callables.
        """
        for dc_field in fields(self):  # type: ignore[arg-type]
            current = getattr(self, dc_field.name)
            if isinstance(current, _DeferredValue) and current.factory is not MISSING:
                factory = current.factory
                value = factory(vllm_config) if callable(factory) else factory
                setattr(self, dc_field.name, value)

    def validate_deferred_fields_initialized(self) -> None:
        """Assert that no field still holds a _DeferredValue.

        Called automatically by VllmConfig.__post_init__ via auto-discovery.
        Raises ValueError listing every offending field name.
        """
        uninitialized = [
            dc_field.name
            for dc_field in fields(self)  # type: ignore[arg-type]
            if isinstance(getattr(self, dc_field.name), _DeferredValue)
        ]
        if uninitialized:
            raise ValueError(
                f"{type(self).__name__}: deferred fields were never initialized: "
                f"{uninitialized}. This is a bug in VllmConfig initialization."
            )
