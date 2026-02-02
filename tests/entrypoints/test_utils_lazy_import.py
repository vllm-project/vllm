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
    from vllm.inputs import TokensPrompt

    # Reset platform detection state
    vllm.platforms._current_platform = None

    # Create a minimal request and prompt for testing
    class FakeRequest:
        max_tokens = 100

    prompt = TokensPrompt(prompt_token_ids=[1, 2, 3])

    # Call get_max_tokens - this triggers platform detection via lazy
    with contextlib.suppress(Exception):
        get_max_tokens(
            max_model_len=1000,
            request=FakeRequest(),  # type: ignore
            prompt=prompt,
            default_sampling_params={},
        )

    # Verify that platform detection was triggered
    assert vllm.platforms._current_platform is not None, (
        "get_max_tokens should trigger platform detection via lazy import"
    )
