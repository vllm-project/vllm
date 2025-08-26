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

# Canonicalize common value types used in config hashing.
# Keeps imports light at module load; heavier bits are imported inside.
def canon_value(x):
    """Return a stable, JSON-serializable canonical form for hashing.

    Order: primitives, special types (Enum, callable, torch.dtype, Path), then
    generic containers (Mapping/Set/Sequence) with recursion.
    """
    import enum
    import pathlib
    from collections.abc import Mapping, Sequence, Set

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

    # Paths
    if isinstance(x, pathlib.Path):
        return str(x)

    # Containers (generic)
    if isinstance(x, Mapping):
        return tuple(sorted((str(k), canon_value(v)) for k, v in x.items()))
    if isinstance(x, Set):
        return tuple(sorted(repr(canon_value(v)) for v in x))
    if isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray)):
        return tuple(canon_value(v) for v in x)

    # Unsupported type
    try:
        from vllm.logger import init_logger
        init_logger(__name__).debug(
            "canon_value: unsupported type '%s'", type(x).__name__)
    except Exception:
        try:
            import logging
            logging.getLogger(__name__).debug(
                "canon_value: unsupported type '%s'", type(x).__name__)
        except Exception:
            pass
    raise TypeError


def get_declared_field_names(cfg) -> list[str]:
    """Declared field names for a config instance.

    Order: dataclass -> pydantic v2 -> __dict__.
    """
    # Dataclass
    if hasattr(cfg, "__dataclass_fields__"):
        # type: ignore[attr-defined]
        return list(cfg.__dataclass_fields__.keys())
    # Pydantic v2
    if hasattr(cfg, "model_fields"):
        try:
            return list(cfg.model_fields.keys())  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            return list(getattr(cfg, "__dict__", {}).keys())
    # Fallback
    return list(getattr(cfg, "__dict__", {}).keys())


def build_opt_out_items(cfg, exclude: set[str]) -> list[tuple[str, object]]:
    """Default-include (opt-out) canonical (key, value) pairs for hashing.

    - Includes declared fields not in `exclude`.
    - Skips non-canonicalizable values.
    """
    import logging
    from contextlib import suppress

    logger = logging.getLogger(__name__)
    items: list[tuple[str, object]] = []
    for key in sorted(get_declared_field_names(cfg)):
        if key in exclude:
            continue
        value = getattr(cfg, key, None)
        try:
            items.append((key, canon_value(value)))
        except TypeError:
            # Log once per key to surface potential under-hashing without
            # spamming logs. The value will be skipped from the hash.
            with suppress(Exception):
                logger.debug("Hash skip: unsupported type for key '%s'", key)
            continue
    return items


def hash_items_sha256(items: list[tuple[str, object]]) -> str:
    """Return a SHA-256 hex digest of the canonical items structure."""
    import hashlib

    return hashlib.sha256(repr(tuple(items)).encode()).hexdigest()


def build_opt_items_override(
    cfg,
    exclude: set[str],
    overrides: dict[str, Callable[[object], object]],
) -> list[tuple[str, object]]:
    """Opt-out items with targeted per-field overrides.

    - Default path uses canon_value.
    - For keys in `overrides`, call the override to produce a stable value.
    - Skips values that cannot be canonicalized.
    """
    import logging
    from contextlib import suppress

    logger = logging.getLogger(__name__)
    items: list[tuple[str, object]] = []
    for key in sorted(get_declared_field_names(cfg)):
        if key in exclude:
            continue
        value = getattr(cfg, key, None)
        if key in overrides:
            try:
                items.append((key, overrides[key](value)))
            except Exception:
                continue
            continue
        try:
            items.append((key, canon_value(value)))
        except TypeError:
            with suppress(Exception):
                logger.debug("Hash skip: unsupported type for key '%s'", key)
            continue
    return item
