# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for regex compilation timeout guard.

Verifies that adversarial regex patterns that would cause exponential
DFA state-space explosion are rejected with a timeout rather than
hanging indefinitely.

Addresses advisory GHSA-rwxx-mrjm-wc2m.
"""

import time
from unittest.mock import patch

import pytest

from vllm.v1.structured_output.utils import compile_regex_with_timeout


class TestCompileRegexWithTimeout:
    """Unit tests for the compile_regex_with_timeout utility."""

    def test_normal_regex_compiles_successfully(self):
        result = compile_regex_with_timeout(lambda pat: "compiled", r"[a-z]+")
        assert result == "compiled"

    def test_timeout_raises_value_error(self):
        def slow_compile(pattern: str):
            time.sleep(10)
            return "never"

        with (
            patch("vllm.envs.VLLM_REGEX_COMPILATION_TIMEOUT_S", 1),
            pytest.raises(ValueError, match="timed out"),
        ):
            compile_regex_with_timeout(slow_compile, r"(a+)+b")

    def test_timeout_disabled_when_zero(self):
        result = None
        with patch("vllm.envs.VLLM_REGEX_COMPILATION_TIMEOUT_S", 0):
            result = compile_regex_with_timeout(lambda pat: "no_timeout", r"(a+)+b")
        assert result == "no_timeout"

    def test_compilation_error_propagates(self):
        def failing_compile(pattern: str):
            raise RuntimeError("compilation failed")

        with pytest.raises(RuntimeError, match="compilation failed"):
            compile_regex_with_timeout(failing_compile, r"bad")

    def test_pattern_included_in_error_message(self):
        def slow_compile(pattern: str):
            time.sleep(10)
            return "never"

        pattern = r"(a+)+b"
        with (
            patch("vllm.envs.VLLM_REGEX_COMPILATION_TIMEOUT_S", 1),
            pytest.raises(ValueError, match=r"\(a\+\)\+b"),
        ):
            compile_regex_with_timeout(slow_compile, pattern)
