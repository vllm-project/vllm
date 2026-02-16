# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests to ensure entrypoints/utils.py doesn't trigger early
platform detection.
"""

import contextlib


def test_utils_import_no_platform_detection():
    """Test that importing utils doesn't trigger platform detection."""
    import vllm.platforms

    # Reset platform detection state
    vllm.platforms._current_platform = None

    # Import utils module
    import vllm.entrypoints.utils  # noqa: F401

    # Verify that importing utils didn't trigger platform detection
    assert vllm.platforms._current_platform is None, (
        "Importing vllm.entrypoints.utils shouldn't trigger detection"
    )


def test_get_max_tokens_triggers_platform_detection():
    """Test that get_max_tokens triggers platform detection via lazy."""
    import vllm.platforms
    from vllm.entrypoints.utils import get_max_tokens

    # Reset platform detection state
    vllm.platforms._current_platform = None

    # Call get_max_tokens with correct signature:
    # get_max_tokens(max_model_len, max_tokens, input_length,
    #                default_sampling_params)
    # This triggers platform detection via lazy import inside the function
    with contextlib.suppress(Exception):
        get_max_tokens(
            max_model_len=1000,
            max_tokens=100,
            input_length=10,
            default_sampling_params={},
        )

    # Verify that platform detection was triggered
    assert vllm.platforms._current_platform is not None, (
        "get_max_tokens should trigger platform detection via lazy import"
    )
