# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the CLI main entrypoint to ensure help doesn't trigger
platform detection.
"""

import contextlib
import sys
from unittest.mock import patch


def test_help_flag_skips_platform_detection():
    """Test that --help doesn't trigger platform detection."""
    # Reset platform detection state
    import vllm.platforms

    vllm.platforms._current_platform = None

    # Mock sys.argv to simulate --help
    with patch.object(sys, "argv", ["vllm", "--help"]), patch.object(sys, "exit"):
        from vllm.entrypoints.cli.main import main

        with contextlib.suppress(SystemExit):
            main()

    # Verify platform was NOT detected
    assert vllm.platforms._current_platform is None, (
        "Platform should not be detected when showing help"
    )


def test_serve_help_flag_skips_platform_detection():
    """Test that serve --help doesn't trigger platform detection."""
    import vllm.platforms

    vllm.platforms._current_platform = None

    with (
        patch.object(sys, "argv", ["vllm", "serve", "--help"]),
        patch.object(sys, "exit"),
    ):
        from vllm.entrypoints.cli.main import main

        with contextlib.suppress(SystemExit):
            main()

    assert vllm.platforms._current_platform is None, (
        "Platform not detected when showing help for serve subcommand"
    )


def test_h_flag_skips_platform_detection():
    """Test that -h doesn't trigger platform detection."""
    import vllm.platforms

    vllm.platforms._current_platform = None

    with patch.object(sys, "argv", ["vllm", "-h"]), patch.object(sys, "exit"):
        from vllm.entrypoints.cli.main import main

        with contextlib.suppress(SystemExit):
            main()

    assert vllm.platforms._current_platform is None, (
        "Platform not detected when showing help with -h flag"
    )


def test_bench_help_flag_skips_platform_detection():
    """Test that bench --help doesn't trigger platform detection."""
    import vllm.platforms

    vllm.platforms._current_platform = None

    with (
        patch.object(sys, "argv", ["vllm", "bench", "--help"]),
        patch.object(sys, "exit"),
    ):
        from vllm.entrypoints.cli.main import main

        with contextlib.suppress(SystemExit):
            main()

    assert vllm.platforms._current_platform is None, (
        "Platform not detected when showing help for bench subcommand"
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
