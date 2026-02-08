# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Utility functions and classes for environment variable handling."""

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Generic, TypeVar, Union, get_args, get_origin

T = TypeVar("T")


class EnvFactory(Generic[T]):
    """Factory for environment variable parsing with custom logic.

    This allows specifying both a default value and a custom parser function
    for environment variables that need special handling beyond simple type conversion.

    Args:
        default_value: The default value to use if the env var is not set.
                      Can be a callable for lazy initialization.
        parse_fn: Function to parse the string value from the environment.
                 Takes the raw string and returns the parsed value.

    Example:
        VLLM_LOGGING_LEVEL: str = EnvFactory("INFO", lambda x: x.upper())
        VLLM_PORT: Optional[int] = EnvFactory(None, get_vllm_port)
    """

    def __init__(self, default_value: T, parse_fn: Callable[[str], T]):
        self.default_value = default_value
        self.parse_fn = parse_fn

    def parse(self, value: str) -> T:
        """Parse the environment variable string value."""
        return self.parse_fn(value)


def env_factory(default_value: T, parse_fn: Callable[[str], T]) -> T:
    """Convenience function to create an EnvFactory.

    This is a more readable alias for creating EnvFactory instances.

    Note: At runtime, this returns an EnvFactory object, but the return type
    is T for type checking purposes so that variable declarations type-check
    correctly (e.g., VLLM_LOGGING_LEVEL: str = env_factory(...)).

    Example:
        VLLM_LOGGING_LEVEL: str = env_factory("INFO", lambda x: x.upper())
    """
    return EnvFactory(default_value, parse_fn)  # type: ignore[return-value]


def env_default_factory(factory_fn: Callable[[], T]) -> T:
    """Create a lazy-initialized default value.

    This is used when the default value needs to be computed at runtime
    rather than at module import time.

    Note: At runtime, this returns the callable itself, but the return type
    is T for type checking purposes so that variable declarations type-check
    correctly (e.g., VLLM_CACHE_ROOT: str = env_default_factory(...)).

    Example:
        VLLM_CACHE_ROOT: str = env_default_factory(
            lambda: parse_path(os.getenv("XDG_CACHE_HOME", "~/.cache")) / "vllm"
        )
    """
    return factory_fn  # type: ignore[return-value]


def parse_path(value: str) -> Path:
    """Parse a path string, expanding user home directory and environment variables.

    Args:
        value: Path string, may contain ~ or environment variables

    Returns:
        Resolved Path object
    """
    return Path(os.path.expanduser(os.path.expandvars(value)))


def parse_list(value: str, separator: str = ",") -> list[str]:
    """Parse a comma-separated (or custom separator) string into a list.

    Args:
        value: String to parse (e.g., "val1,val2,val3")
        separator: Separator character (default: ",")

    Returns:
        List of stripped string values
    """
    if not value.strip():
        return []
    return [item.strip() for item in value.split(separator) if item.strip()]


def unwrap_optional(type_hint: Any) -> Any:
    """Unwrap Optional[T] or T | None to get the underlying type T.

    Args:
        type_hint: A type hint that may be Optional

    Returns:
        The unwrapped type, or the original type if not Optional
    """
    origin = get_origin(type_hint)

    # Handle Union types (including Optional which is Union[T, None])
    # Also handle Python 3.10+ union syntax (int | None creates types.UnionType)
    if origin is Union or (
        origin is not None and str(origin) == "<class 'types.UnionType'>"
    ):
        args = get_args(type_hint)
        # Filter out NoneType
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            return non_none_args[0]
        elif len(non_none_args) > 1:
            # Multiple non-None types in Union, return as-is
            return type_hint

    return type_hint


def is_type_with_args(
    type_hint: Any, base_type: type, expected_args: list[type]
) -> bool:
    """Check if a type hint matches a generic type with specific arguments.

    For example, check if type_hint is list[str].

    Args:
        type_hint: The type hint to check
        base_type: The expected base type (e.g., list)
        expected_args: List of expected type arguments (e.g., [str])

    Returns:
        True if the type matches, False otherwise
    """
    origin = get_origin(type_hint)
    if origin is not base_type:
        return False

    args = get_args(type_hint)
    return list(args) == expected_args


def env_with_choices(
    env_name: str,
    default: str | None,
    choices: list[str] | Callable[[], list[str]],
    case_sensitive: bool = True,
) -> Callable[[], str | None]:
    """
    Creates a lambda that validates environment variable against allowed choices.

    Args:
        env_name: Name of the environment variable
        default: Default value if env var not set (can be None)
        choices: List of allowed values or callable that returns the list
        case_sensitive: Whether to enforce case-sensitive matching

    Returns:
        A callable that returns the validated environment variable value (or None)

    Raises:
        ValueError: If the environment variable value is not in allowed choices
    """

    def _get_validated_env() -> str | None:
        value = os.environ.get(env_name, default)

        # Return None if value is None (env var not set and default is None)
        if value is None:
            return None

        allowed_choices = choices() if callable(choices) else choices

        if case_sensitive:
            if value not in allowed_choices:
                raise ValueError(
                    f"Invalid value for {env_name}: {value}. "
                    f"Must be one of: {', '.join(allowed_choices)}"
                )
        else:
            value_lower = value.lower()
            choices_lower = [c.lower() for c in allowed_choices]
            if value_lower not in choices_lower:
                raise ValueError(
                    f"Invalid value for {env_name}: {value}. "
                    f"Must be one of (case-insensitive): {', '.join(allowed_choices)}"
                )
            # Return the canonical choice (preserve original case from choices)
            idx = choices_lower.index(value_lower)
            value = allowed_choices[idx]

        return value

    return _get_validated_env


def env_list_with_choices(
    env_name: str,
    default: list[str],
    choices: list[str] | Callable[[], list[str]],
    case_sensitive: bool = True,
) -> Callable[[], list[str]]:
    """
    Creates a lambda that validates environment variable containing
    comma-separated values against allowed choices.

    Args:
        env_name: Name of the environment variable
        default: Default list of values if env var not set
        choices: List of allowed values or callable that returns the list
        case_sensitive: Whether to enforce case-sensitive matching

    Returns:
        A callable that returns the validated list of values

    Raises:
        ValueError: If any value in the environment variable is not in allowed choices
    """

    def _get_validated_env_list() -> list[str]:
        value = os.environ.get(env_name, None)

        if value is None:
            return default

        values = parse_list(value)
        allowed_choices = choices() if callable(choices) else choices

        if case_sensitive:
            for val in values:
                if val not in allowed_choices:
                    raise ValueError(
                        f"Invalid value '{val}' in {env_name}. "
                        f"Must be one of: {', '.join(allowed_choices)}"
                    )
        else:
            validated_values = []
            choices_lower = [c.lower() for c in allowed_choices]
            for val in values:
                val_lower = val.lower()
                if val_lower not in choices_lower:
                    raise ValueError(
                        f"Invalid value '{val}' in {env_name}. "
                        f"Must be one of (case-insensitive): "
                        f"{', '.join(allowed_choices)}"
                    )
                # Use canonical choice
                idx = choices_lower.index(val_lower)
                validated_values.append(allowed_choices[idx])
            values = validated_values

        return values

    return _get_validated_env_list


def env_set_with_choices(
    env_name: str,
    default: list[str],
    choices: list[str] | Callable[[], list[str]],
    case_sensitive: bool = True,
) -> Callable[[], set[str]]:
    """
    Creates a lambda that validates environment variable containing
    comma-separated values against allowed choices and returns as a set.

    Args:
        env_name: Name of the environment variable
        default: Default list of values if env var not set
        choices: List of allowed values or callable that returns the list
        case_sensitive: Whether to enforce case-sensitive matching

    Returns:
        A callable that returns the validated set of values
    """

    def _get_validated_env_set() -> set[str]:
        return set(env_list_with_choices(env_name, default, choices, case_sensitive)())

    return _get_validated_env_set
