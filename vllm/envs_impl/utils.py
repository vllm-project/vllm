# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Utility functions and classes for environment variable handling."""

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Generic, TypeVar, Union, get_args, get_origin

T = TypeVar("T")


class EnvFactory(Generic[T]):
    """Pairs a default value with a custom parse function for an env var."""

    def __init__(self, default_value: T, parse_fn: Callable[[str], T]):
        self.default_value = default_value
        self.parse_fn = parse_fn

    def parse(self, value: str) -> T:
        return self.parse_fn(value)


def env_factory(default_value: T, parse_fn: Callable[[str], T]) -> T:
    """Create an EnvFactory (typed as T for type-checker compatibility)."""
    return EnvFactory(default_value, parse_fn)  # type: ignore[return-value]


def parse_path(value: str) -> Path:
    """Expand ~ and env vars, return a Path."""
    return Path(os.path.expanduser(os.path.expandvars(value)))


def parse_list(value: str, separator: str = ",") -> list[str]:
    """Split a separated string into a list of stripped values."""
    if not value.strip():
        return []
    return [item.strip() for item in value.split(separator) if item.strip()]


def unwrap_optional(type_hint: Any) -> Any:
    """Unwrap Optional[T] or T | None to get T."""
    origin = get_origin(type_hint)

    if origin is Union or (
        origin is not None and str(origin) == "<class 'types.UnionType'>"
    ):
        args = get_args(type_hint)
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            return non_none_args[0]
        elif len(non_none_args) > 1:
            return type_hint

    return type_hint


def is_type_with_args(
    type_hint: Any, base_type: type, expected_args: list[type]
) -> bool:
    """Check if type_hint is e.g. list[str]."""
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
    """Return a callable that reads and validates an env var against choices."""

    def _get_validated_env() -> str | None:
        value = os.environ.get(env_name, default)
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
    """Like env_with_choices but for comma-separated list values."""

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
    """Like env_list_with_choices but returns a set."""

    def _get_validated_env_set() -> set[str]:
        return set(env_list_with_choices(env_name, default, choices, case_sensitive)())

    return _get_validated_env_set


def validate_choice(
    value: str,
    env_name: str,
    choices: list[str],
    case_sensitive: bool = True,
) -> str:
    """Validate a value against allowed choices (pure parse, no env reading)."""
    if case_sensitive:
        if value not in choices:
            raise ValueError(
                f"Invalid value for {env_name}: {value}. "
                f"Must be one of: {', '.join(choices)}"
            )
        return value
    else:
        value_lower = value.lower()
        choices_lower = [c.lower() for c in choices]
        if value_lower not in choices_lower:
            raise ValueError(
                f"Invalid value for {env_name}: {value}. "
                f"Must be one of (case-insensitive): {', '.join(choices)}"
            )
        idx = choices_lower.index(value_lower)
        return choices[idx]


def validate_list_choices(
    value: str,
    env_name: str,
    choices: list[str],
    case_sensitive: bool = True,
) -> list[str]:
    """Parse comma-separated value and validate each item against choices."""
    values = parse_list(value)
    if case_sensitive:
        for val in values:
            if val not in choices:
                raise ValueError(
                    f"Invalid value '{val}' in {env_name}. "
                    f"Must be one of: {', '.join(choices)}"
                )
        return values
    else:
        validated = []
        choices_lower = [c.lower() for c in choices]
        for val in values:
            val_lower = val.lower()
            if val_lower not in choices_lower:
                raise ValueError(
                    f"Invalid value '{val}' in {env_name}. "
                    f"Must be one of (case-insensitive): {', '.join(choices)}"
                )
            idx = choices_lower.index(val_lower)
            validated.append(choices[idx])
        return validated


def validate_set_choices(
    value: str,
    env_name: str,
    choices: list[str],
    case_sensitive: bool = True,
) -> set[str]:
    """Parse comma-separated value and validate each item, returning a set."""
    return set(validate_list_choices(value, env_name, choices, case_sensitive))
