# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Module-level __getattr__ that reads env vars with automatic type conversion.

Uses pydantic's TypeAdapter for coercion and field validators (BeforeValidator)
declared via Annotated types in _variables.py.

Use ``import vllm.envs_impl as envs`` to access environment variables.
"""

import functools
import os
import sys
from typing import TYPE_CHECKING, Any, get_type_hints

from pydantic import TypeAdapter
from pydantic_core import PydanticUndefined

from vllm.envs_impl import _variables

if TYPE_CHECKING:
    from vllm.envs_impl._variables import *  # noqa: F403  # type: ignore[assignment]

# get_type_hints with include_extras=True preserves Annotated[T, BeforeValidator(...)]
# so TypeAdapter can discover and run the validators.
_type_hints: dict[str, Any] = get_type_hints(_variables, include_extras=True)

# Field info (default / default_factory) extracted exactly like get_kwargs() does it.
_env_fields = _variables._fields

# Pre-build one TypeAdapter per env var so validate_python() has no construction cost.
_adapters: dict[str, TypeAdapter] = {
    name: TypeAdapter(hint) for name, hint in _type_hints.items() if name in _env_fields
}


def __getattr__(name: str) -> Any:
    """Read an env var, run its BeforeValidator (if any), else return the default."""
    if name not in _env_fields:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    env_value = os.getenv(name)

    if env_value is None:
        field = _env_fields[name]
        if field.default is not PydanticUndefined:
            return field.default
        if field.default_factory is not None:
            return field.default_factory()
        raise ValueError(f"Env var {name!r} has no default value")  # shouldn't happen

    return _adapters[name].validate_python(env_value)


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


def _is_envs_cache_enabled() -> bool:
    """Return True if __getattr__ is currently wrapped with functools.cache."""
    return hasattr(__getattr__, "__wrapped__")


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
    "_is_envs_cache_enabled",
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
        return list(self._original_module.__dict__.get("_env_fields", {}).keys())


_this_module = sys.modules[__name__]
sys.modules[__name__] = _EnvsModuleWrapper(_this_module)  # type: ignore[assignment]
