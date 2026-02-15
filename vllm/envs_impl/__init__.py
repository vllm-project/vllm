# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Module-level __getattr__ that reads env vars with automatic type conversion.

Use ``import vllm.envs as envs`` to access environment variables.
"""

import functools
import os
import sys
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from vllm.envs_impl import _variables
from vllm.envs_impl.utils import (
    EnvFactory,
    is_type_with_args,
    parse_list,
    parse_path,
    unwrap_optional,
)

if TYPE_CHECKING:
    from vllm.envs_impl._variables import *  # noqa: F403  # type: ignore[assignment]

_type_hints = get_type_hints(_variables)
_env_defaults = _variables._defaults


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true")


def _is_bool_like(value: str) -> bool:
    """Check if value is clearly boolean, not a number like '5'."""
    return value.strip().lower() in ("1", "0", "true", "false")


# Ordered by specificity: bool before int (bool is subclass of int).
_type_parsers: dict[type, Any] = {
    bool: _parse_bool,
    int: int,
    float: float,
    Path: parse_path,
    str: lambda v: v,
}


def __getattr__(name: str) -> Any:
    """Read an env var, apply type conversion or custom parsing, else default."""
    if name not in _env_defaults:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    default = _env_defaults[name]

    if isinstance(default, EnvFactory):
        env_value = os.getenv(name)
        if env_value is None:
            default_val = default.default_value
            return default_val() if callable(default_val) else default_val
        return default.parse(env_value)

    env_value = os.getenv(name)
    if env_value is None:
        return default

    # Automatic type conversion based on type hint
    var_type = _type_hints.get(name)
    if var_type is None:
        return env_value

    var_type = unwrap_optional(var_type)

    parser = _type_parsers.get(var_type)
    if parser is not None:
        return parser(env_value)

    if is_type_with_args(var_type, list, [str]):
        return parse_list(env_value)

    origin = get_origin(var_type)

    if origin is Literal:
        literal_values = get_args(var_type)
        env_lower = env_value.lower()
        for literal_val in literal_values:
            if isinstance(literal_val, str) and literal_val.lower() == env_lower:
                return literal_val
        for literal_val in literal_values:
            if isinstance(literal_val, str):
                continue
            try:
                if type(literal_val)(env_value) == literal_val:
                    return literal_val
            except (ValueError, TypeError):
                continue
        raise ValueError(
            f"Invalid value for {name}: {env_value!r}. "
            f"Must be one of: {', '.join(repr(v) for v in literal_values)}"
        )

    if origin is Union or (
        origin is not None and str(origin) == "<class 'types.UnionType'>"
    ):
        union_args = get_args(var_type)
        non_none_types = {arg for arg in union_args if arg is not type(None)}
        for target_type, parser in _type_parsers.items():
            if target_type not in non_none_types:
                continue
            try:
                result = parser(env_value)
                # Avoid "5" being silently converted to True
                if target_type is bool and not _is_bool_like(env_value):
                    continue
                return result
            except (ValueError, TypeError):
                continue

        return env_value

    import logging

    logger = logging.getLogger(__name__)
    logger.warning(
        "Unsupported type %s for environment variable %s. "
        "Returning raw string value. Consider using "
        "env_factory() for custom parsing.",
        var_type,
        name,
    )
    return env_value


def enable_envs_cache() -> None:
    """Cache env var reads so each variable is parsed only once."""
    global __getattr__
    if not hasattr(__getattr__, "__wrapped__"):
        __getattr__ = functools.cache(__getattr__)


def disable_envs_cache() -> None:
    """Revert enable_envs_cache() so env vars are read fresh."""
    global __getattr__
    if hasattr(__getattr__, "__wrapped__"):
        __getattr__ = __getattr__.__wrapped__


def is_set(env_var_name: str) -> bool:
    """Check if an environment variable is explicitly set (in os.environ)."""
    return env_var_name in os.environ


def maybe_convert_bool(val: str | bool | None) -> bool | None:
    """Convert a string like "1"/"true" to bool. Pass through bool/None."""
    if isinstance(val, bool):
        return val
    if val is None:
        return None
    if isinstance(val, str):
        return val.lower() in ("1", "true", "yes", "on")
    return bool(val)


__all__ = [
    "enable_envs_cache",
    "disable_envs_cache",
    "is_set",
    "maybe_convert_bool",
]


class _EnvsModuleWrapper:
    """Module wrapper to support ``"VAR_NAME" in envs`` syntax."""

    def __init__(self, original_module):
        self._original_module = original_module

    def __getattr__(self, name: str) -> Any:
        if name in self._original_module.__dict__:
            return self._original_module.__dict__[name]
        return self._original_module.__dict__["__getattr__"](name)

    def __contains__(self, env_var_name: str) -> bool:
        return env_var_name in os.environ

    def __dir__(self):
        return list(self._original_module.__dict__.get("_env_defaults", {}).keys())


_this_module = sys.modules[__name__]
sys.modules[__name__] = _EnvsModuleWrapper(_this_module)  # type: ignore[assignment]
