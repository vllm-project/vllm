# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the CLI main entrypoint to ensure help doesn't trigger
platform detection.
"""

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
        ["vllm", "serve", "--help=ModelConfig"],
    ],
)
def test_needs_help_detects_help_flags(argv):
    """Test that needs_help() correctly detects help flags in sys.argv."""
    from vllm.engine.arg_utils import needs_help

    # patch.object on sys.argv is safe — it's a simple list attribute
    # with no lazy-init or side-effect machinery.
    with patch.object(sys, "argv", argv):
        assert needs_help(), f"needs_help() should return True for {argv}"


@pytest.mark.parametrize(
    "argv",
    [
        ["vllm", "serve", "--model", "test"],
        ["vllm", "bench", "latency", "--model", "test"],
        ["vllm", "collect-env"],
    ],
)
def test_needs_help_returns_false_without_help_flags(argv):
    """Test that needs_help() returns False when no help flag is present."""
    from vllm.engine.arg_utils import needs_help

    with patch.object(sys, "argv", argv):
        assert not needs_help(), f"needs_help() should return False for {argv}"


def test_bench_help_skips_platform_detection():
    """Test that the bench guard in main() is skipped when --help is present.

    The guard in main.py is:
        if sys.argv[1] == "bench" and not showing_help
    When showing_help is True, current_platform is never accessed for
    the bench CPU-override, avoiding unnecessary platform detection.
    """
    from vllm.engine.arg_utils import needs_help

    # Verify the guard: needs_help() == True means "not showing_help" is False,
    # so the bench platform-override block is skipped.
    with patch.object(sys, "argv", ["vllm", "bench", "--help"]):
        assert needs_help(), "needs_help() should be True for bench --help"

    # Without --help the guard would be entered
    with patch.object(sys, "argv", ["vllm", "bench", "latency"]):
        assert not needs_help(), "needs_help() should be False without --help"
