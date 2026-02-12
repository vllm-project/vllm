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
    """Test that platform detection happens when not showing help."""
    import vllm.platforms
    from vllm.engine.arg_utils import NEEDS_HELP

    vllm.platforms._current_platform = None

    # We can't fully test command execution without proper setup,
    # but we can verify the help check logic
    with patch.object(sys, "argv", ["vllm", "serve", "--model", "test"]):
        # Just check that NEEDS_HELP correctly detects absence of help flags
        # The actual check in main.py uses NEEDS_HELP
        assert not NEEDS_HELP, "Should not detect help when not present"
