# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests to ensure entrypoints/utils.py doesn't trigger early
platform detection.
"""


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
