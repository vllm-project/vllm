# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""vLLM environment variables module.

This module provides access to all vLLM environment variables with:
- Type-safe declarations (visible to IDEs and type checkers)
- Automatic type conversion based on type hints
- Lazy evaluation (variables are read from environment only when accessed)
- Caching support for performance
- Custom parsing logic for complex types

Usage:
    import vllm.envs as envs

    # Access environment variables
    log_level = envs.VLLM_LOGGING_LEVEL  # Returns str, auto-parsed
    timeout = envs.VLLM_ENGINE_ITERATION_TIMEOUT_S  # Returns int, auto-parsed
    use_sampler = envs.VLLM_USE_FLASHINFER_SAMPLER  # Returns bool | None
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

# Import the variables module for accessing defaults and type hints
from vllm.envs import _variables
from vllm.envs.utils import (
    EnvFactory,
    is_type_with_args,
    parse_list,
    parse_path,
    unwrap_optional,
)

if TYPE_CHECKING:
    # For type checkers and IDEs, import all variable declarations directly
    # This provides full type information and autocomplete
    from vllm.envs._variables import *  # noqa: F403

# Get type hints from _variables module for runtime type conversion
_type_hints = get_type_hints(_variables)

# Access the defaults dictionary from _variables
_env_defaults = _variables._defaults


def __getattr__(name: str) -> Any:
    """Module-level attribute access for environment variables.

    This function is called when accessing any attribute on the envs module.
    It handles:
    1. Reading the environment variable value
    2. Applying custom parsing logic if defined (via EnvFactory)
    3. Automatic type conversion based on type hints for "trivial" types
    4. Returning default values when environment variable is not set

    Args:
        name: Name of the environment variable to access

    Returns:
        The parsed and typed value of the environment variable

    Raises:
        AttributeError: If the variable name is not defined in _variables.py
        ValueError: If automatic type conversion fails or value is invalid
    """
    if name not in _env_defaults:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    default = _env_defaults[name]

    # Handle EnvFactory: custom parser for env var
    if isinstance(default, EnvFactory):
        env_value = os.getenv(name)
        if env_value is None:
            # Use default value from factory
            default_val = default.default_value
            return default_val() if callable(default_val) else default_val
        else:
            # Parse the env value using factory's parser
            return default.parse(env_value)

    # Handle plain callables (env_default_factory, env_with_choices, etc.)
    # These handle env var reading internally, so always call them
    if callable(default):
        return default()

    # Handle non-callable defaults with automatic type conversion
    env_value = os.getenv(name)

    if env_value is None:
        # Not found in environment, return default value
        return default

    # Automatic parsing of "trivial" data types based on type hint
    var_type = _type_hints.get(name)
    if var_type is None:
        # No type hint, return string value
        return env_value

    # Unwrap Optional types (e.g., Optional[int] -> int)
    var_type = unwrap_optional(var_type)

    # Handle basic types
    if var_type is str:
        return env_value

    if var_type is int:
        return int(env_value)

    if var_type is float:
        return float(env_value)

    if var_type is bool:
        # Parse boolean: "1", "true", "True", "TRUE" -> True
        # Everything else including "0", "false", "False", "" -> False
        return env_value.strip().lower() in ("1", "true")

    if var_type is Path or var_type is type(Path):
        return parse_path(env_value)

    # Handle list[str]
    if is_type_with_args(var_type, list, [str]):
        return parse_list(env_value)

    # Handle Literal types with validation
    origin = get_origin(var_type)
    if origin is Literal:
        literal_values = get_args(var_type)
        # All string literals - do case-insensitive matching
        if all(isinstance(v, str) for v in literal_values):
            env_lower = env_value.lower()
            for literal_val in literal_values:
                if literal_val.lower() == env_lower:
                    # Return the canonical value from the Literal
                    return literal_val
            # Value not in Literal choices
            raise ValueError(
                f"Invalid value for {name}: {env_value!r}. "
                f"Must be one of (case-insensitive): "
                f"{', '.join(repr(v) for v in literal_values)}"
            )
        # Mixed types or non-string literals - exact match only
        if env_value in literal_values:
            return env_value
        # Try type conversion for non-string literals
        for literal_val in literal_values:
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
        # Handle Union types by trying each type in order of specificity
        # More specific types first (bool, int, float) before generic (str)
        union_args = get_args(var_type)
        # Filter out NoneType - if value exists in env, it's not None
        non_none_types = [arg for arg in union_args if arg is not type(None)]

        # Define conversion priority: more specific types first
        type_priority = [
            (bool, lambda v: v.strip().lower() in ("1", "true")),
            (int, lambda v: int(v)),
            (float, lambda v: float(v)),
            (Path, lambda v: parse_path(v)),
            (type(Path), lambda v: parse_path(v)),
            (str, lambda v: v),  # str always succeeds, so try it last
        ]

        # Try conversions in priority order, but only for types in the Union
        for target_type, converter in type_priority:
            if target_type in non_none_types or (
                target_type is type(Path) and Path in non_none_types
            ):
                try:
                    result = converter(env_value)
                    # For bool, ensure the conversion is meaningful
                    # If the value is a number like "5", don't convert to bool
                    if target_type is bool and env_value.strip().lower() not in (
                        "1",
                        "0",
                        "true",
                        "false",
                    ):
                        # Only accept bool if value is clearly boolean
                        continue
                    return result
                except (ValueError, TypeError):
                    continue

        # If no conversion succeeded, return as string
        return env_value

    # Fallback: unsupported type, return as string with a warning
    # In production, you might want to raise an error instead
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
    """Enable caching of environment variable access.

    After calling this function, each environment variable will be read and
    parsed only once, with subsequent accesses returning the cached value.

    This improves performance but means changes to environment variables
    after the first access will not be reflected.

    Note: This modifies the module's __getattr__ function to wrap it with
    functools.cache. Tests should call disable_envs_cache() to reset.
    """
    global __getattr__
    # Only wrap if not already wrapped
    if not hasattr(__getattr__, "__wrapped__"):
        __getattr__ = functools.cache(__getattr__)


def disable_envs_cache() -> None:
    """Disable caching of environment variable access.

    Reverts the effect of enable_envs_cache(), making subsequent accesses
    read from the environment again.

    This is primarily useful for testing.
    """
    global __getattr__
    # Unwrap if wrapped
    if hasattr(__getattr__, "__wrapped__"):
        __getattr__ = __getattr__.__wrapped__


def is_set(env_var_name: str) -> bool:
    """Check if an environment variable is explicitly set in the environment.

    This checks os.environ, not whether the variable is defined in vLLM.
    Use this to detect if a variable was explicitly set by the user.

    Args:
        env_var_name: Name of the environment variable to check

    Returns:
        True if the environment variable is set, False otherwise

    Example:
        >>> # Assuming VLLM_CUSTOM_VAR is not set
        >>> is_set("VLLM_CUSTOM_VAR")
        False
        >>> os.environ["VLLM_CUSTOM_VAR"] = "some_value"
        >>> is_set("VLLM_CUSTOM_VAR")
        True
    """
    return env_var_name in os.environ


def maybe_convert_bool(val: str | bool | None) -> bool | None:
    """Convert a value to boolean if it's a string representation.

    Args:
        val: Value to convert (string, bool, or None)

    Returns:
        Boolean value, or None if input is None

    Example:
        >>> maybe_convert_bool("1")
        True
        >>> maybe_convert_bool("false")
        False
        >>> maybe_convert_bool(None)
        None
        >>> maybe_convert_bool(True)
        True
    """
    if isinstance(val, bool):
        return val
    if val is None:
        return None
    if isinstance(val, str):
        return val.lower() in ("1", "true", "yes", "on")
    return bool(val)


# Re-export key utility functions for convenience
__all__ = [
    "enable_envs_cache",
    "disable_envs_cache",
    "is_set",
    "maybe_convert_bool",
]


# ============================================================================
# Module wrapper to support 'in' operator for checking if env var is set
# ============================================================================


class _EnvsModuleWrapper:
    """Wrapper that allows using 'in' operator to check if env var is set.

    This enables Pythonic syntax:
        >>> import vllm.envs as envs
        >>> "VLLM_LOGGING_LEVEL" in envs  # Returns True if env var is set
    """

    def __init__(self, original_module):
        self._original_module = original_module

    def __getattr__(self, name: str) -> Any:
        # First, try to get from the original module's dict (functions, classes, etc.)
        if name in self._original_module.__dict__:
            return self._original_module.__dict__[name]
        # Then delegate to the original __getattr__ for env vars
        return self._original_module.__dict__["__getattr__"](name)

    def __contains__(self, env_var_name: str) -> bool:
        """Check if an environment variable is explicitly set in the environment.

        This checks os.environ, not whether the variable is defined in vLLM.
        Use this to detect if a variable was explicitly set by the user.

        Args:
            env_var_name: Name of the environment variable to check

        Returns:
            True if the environment variable is set, False otherwise

        Example:
            >>> import vllm.envs as envs
            >>> # Assuming VLLM_CUSTOM_VAR is not set
            >>> "VLLM_CUSTOM_VAR" in envs
            False
            >>> import os
            >>> os.environ["VLLM_CUSTOM_VAR"] = "some_value"
            >>> "VLLM_CUSTOM_VAR" in envs
            True
        """
        return env_var_name in os.environ

    def __dir__(self):
        # Return the list of available env vars from _env_defaults
        return list(self._original_module.__dict__.get("_env_defaults", {}).keys())


# Replace this module with the wrapper to enable 'in' operator support
_this_module = sys.modules[__name__]
sys.modules[__name__] = _EnvsModuleWrapper(_this_module)
