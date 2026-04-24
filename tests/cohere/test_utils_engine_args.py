#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Shared utilities for parsing engine args from hardware profiles in tests.
"""

import dataclasses
import os
import shlex
from typing import Any

from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser


def parse_engine_args_to_dict(args_string: str) -> dict[str, Any]:
    """
    Parse CLI-style engine args into kwargs using vLLM's EngineArgs.

    This leverages vLLM's built-in FlexibleArgumentParser which automatically
    handles kebab-case to snake_case conversion and all engine argument types.

    Args:
        args_string: Space-separated CLI args like
            "--max-model-len 32768 --enable-chunked-prefill"

    Returns:
        Dictionary of kwargs suitable for LLM constructor
    """
    if not args_string or not args_string.strip():
        return {}

    # Create a parser with all engine args
    parser = FlexibleArgumentParser()
    parser = EngineArgs.add_cli_args(parser)

    # Parse the string args
    args_list = shlex.split(args_string)
    try:
        namespace = parser.parse_args(args_list)
    except SystemExit as err:
        # argparse calls sys.exit on error, catch and raise our own
        raise ValueError(f"Failed to parse engine args: {args_string}") from err

    # Convert to EngineArgs and extract as dict
    engine_args = EngineArgs.from_cli_args(namespace)

    # Convert EngineArgs dataclass to dict, excluding None values
    kwargs = {
        field.name: getattr(engine_args, field.name)
        for field in dataclasses.fields(engine_args)
        if getattr(engine_args, field.name) is not None
    }

    return kwargs


def get_engine_kwargs_with_overrides(
    test_kwargs: dict[str, Any],
    engine_args_override: str | None = None,
) -> dict[str, Any]:
    """
    Get effective engine kwargs by merging profile args with test-specific overrides.

    Args:
        test_kwargs: Test-specific required settings that always override
            profile defaults
        engine_args_override: CLI-style engine args (if None, uses
            VLLM_HARDWARE_PROFILE_ARGS env var)

    Returns:
        Merged kwargs dictionary with test_kwargs overriding profile_kwargs
    """
    # Capture source before fallback reassignment for accurate diagnostics.
    source_is_cli_override = engine_args_override is not None

    # Get engine args from CLI override or environment variable
    if engine_args_override is None:
        engine_args_override = os.environ.get("VLLM_HARDWARE_PROFILE_ARGS", "")

    source_label = (
        "CLI override" if source_is_cli_override else "VLLM_HARDWARE_PROFILE_ARGS env"
    )
    print(f"Engine args source: {source_label}")
    if engine_args_override:
        print(f"Raw engine args: {engine_args_override}")

    # Parse profile-derived engine args using vLLM's EngineArgs
    profile_kwargs = parse_engine_args_to_dict(engine_args_override)

    # Merge: profile args provide defaults, test args override any overlaps
    # Dictionary unpacking order matters: {**a, **b}
    # means b overwrites a for overlapping keys.
    effective_kwargs = {**profile_kwargs, **test_kwargs}

    # Log if any profile args were overridden by test args
    overridden_keys = set(profile_kwargs.keys()) & set(test_kwargs.keys())
    if overridden_keys:
        print(f"Test-specific args overriding profile args: {sorted(overridden_keys)}")

    print("Effective engine kwargs:")
    for key, value in sorted(effective_kwargs.items()):
        print(f"  {key}: {value}")

    return effective_kwargs


def get_async_engine_args_with_overrides(
    test_kwargs: dict[str, Any],
    engine_args_override: str | None = None,
) -> AsyncEngineArgs:
    """
    Get AsyncEngineArgs instance by merging profile args with test-specific overrides.

    Args:
        test_kwargs: Test-specific required settings that always override
            profile defaults
        engine_args_override: CLI-style engine args (if None, uses
            VLLM_HARDWARE_PROFILE_ARGS env var)

    Returns:
        AsyncEngineArgs instance with merged configuration
    """
    # Get merged kwargs
    effective_kwargs = get_engine_kwargs_with_overrides(
        test_kwargs, engine_args_override
    )

    # Convert to AsyncEngineArgs instance
    return AsyncEngineArgs(**effective_kwargs)
