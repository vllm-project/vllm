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
    import enum
    import pathlib

    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, enum.Enum):
        return x.value

    # Handle torch dtypes without incurring import cost
    try:
        import torch  # noqa: WPS433 (local import by design)
        if isinstance(x, torch.dtype):
            return str(x)
    except Exception:
        pass

    # Treat callables by qualified name for stability
    try:
        if callable(x):
            module = getattr(x, "__module__", "")
            qual = getattr(x, "__qualname__", str(x))
            return f"{module}.{qual}" if module else qual
    except Exception:
        pass

    if isinstance(x, pathlib.Path):
        return str(x)
    if isinstance(x, dict):
        return tuple(sorted((str(k), canon_value(v)) for k, v in x.items()))
    if isinstance(x, set):
        return tuple(sorted(repr(canon_value(v)) for v in x))
    if isinstance(x, (list, tuple)):
        return tuple(canon_value(v) for v in x)
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
    """Canonical (key, value) items for opt-out hashing.

    Includes declared fields not in `exclude`. Values are canonicalized.
    Unsupported values are skipped.
    """
    items: list[tuple[str, object]] = []
    for key in sorted(get_declared_field_names(cfg)):
        if key in exclude:
            continue
        value = getattr(cfg, key, None)
        try:
            items.append((key, canon_value(value)))
        except TypeError:
            continue
    return items


def hash_items_sha256(items: list[tuple[str, object]]) -> str:
    """Return a SHA-256 hex digest of the canonical items structure."""
    import hashlib
    return hashlib.sha256(repr(tuple(items)).encode()).hexdigest()
