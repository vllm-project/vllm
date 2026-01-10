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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union, get_args, get_origin, get_type_hints

# Import the variables module for accessing defaults and type hints
from vllm.envs import _variables
from vllm.envs.utils import EnvFactory, is_type_with_args, parse_list, parse_path, unwrap_optional

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

    # Check if environment variable is set
    env_value = os.getenv(name)

    if env_value is None:
        # Not found in environment, return default value
        if isinstance(default, EnvFactory):
            default = default.default_value

        # Call if lazy-initialized (callable default)
        return default() if callable(default) else default

    # Environment variable is set, parse the string value
    if isinstance(default, EnvFactory):
        # If factory provided, use it to parse the string
        return default.parse(env_value)

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

    # Handle Literal types (just validate and return the string)
    origin = get_origin(var_type)
    if origin is type(Union):
        # This might be a Literal or other Union, handle as string for now
        # More sophisticated handling could validate against Literal values
        return env_value

    # Fallback: unsupported type, return as string with a warning
    # In production, you might want to raise an error instead
    import logging

    logger = logging.getLogger(__name__)
    logger.warning(
        f"Unsupported type {var_type} for environment variable {name}. "
        f"Returning raw string value. Consider using env_factory() for custom parsing."
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
    """Check if an environment variable is explicitly set.

    Args:
        env_var_name: Name of the environment variable

    Returns:
        True if the variable is set in the environment, False otherwise

    Example:
        >>> is_set("VLLM_LOGGING_LEVEL")
        False
        >>> os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
        >>> is_set("VLLM_LOGGING_LEVEL")
        True
    """
    return env_var_name in os.environ


# Re-export key utility functions for convenience
__all__ = [
    "enable_envs_cache",
    "disable_envs_cache",
    "is_set",
]
