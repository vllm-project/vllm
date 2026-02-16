# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the CLI main entrypoint to ensure help doesn't trigger
platform detection.
"""

import contextlib
import sys
from unittest.mock import patch

import pytest


@pytest.mark.parametrize(
    "argv",
    [
        ["vllm", "--help"],
        ["vllm", "serve", "--help"],
        ["vllm", "-h"],
        ["vllm", "bench", "--help"],
    ],
)
def test_help_flag_skips_platform_detection(argv):
    """Test that help flags don't trigger platform detection."""
    import vllm.platforms

    vllm.platforms._current_platform = None

    with patch.object(sys, "argv", argv), patch.object(sys, "exit"):
        from vllm.entrypoints.cli.main import main

        with contextlib.suppress(SystemExit):
            main()

    assert vllm.platforms._current_platform is None, (
        f"Platform should not be detected when showing help with {argv}"
    )


def test_no_help_flag_allows_platform_detection():
    """Test that the runtime help check correctly detects no help."""
    from vllm.engine.arg_utils import needs_help

    # Verify that needs_help() correctly detects absence of help flags
    with patch.object(sys, "argv", ["vllm", "serve", "--model", "test"]):
        assert not needs_help(), "Should not detect help when no help flag is present"

    # Verify that needs_help() correctly detects help flags
    with patch.object(sys, "argv", ["vllm", "serve", "--help"]):
        assert needs_help(), "Should detect help when --help flag is present"
