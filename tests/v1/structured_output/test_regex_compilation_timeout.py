# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for regex compilation timeout guard.

Verifies that adversarial regex patterns that would cause exponential
DFA state-space explosion are rejected with a timeout rather than
hanging indefinitely, and that timed-out compilation work is killed
(no lingering processes).

Addresses advisories GHSA-rwxx-mrjm-wc2m and GHSA-g773-5cq5-5j53.
"""

import contextlib
import os
import subprocess
import time
from unittest.mock import patch

import pytest

from vllm.v1.structured_output.utils import compile_regex_with_timeout


def _slow_compile(pattern: str) -> str:
    """Top-level picklable function that blocks indefinitely."""
    time.sleep(120)
    return "never"


def _fast_compile(pattern: str) -> str:
    """Top-level picklable function that returns immediately."""
    return f"compiled:{pattern}"


def _failing_compile(pattern: str) -> str:
    """Top-level picklable function that raises."""
    raise RuntimeError("compilation failed")


class TestCompileRegexWithTimeout:
    """Unit tests for the compile_regex_with_timeout utility."""

    def test_normal_regex_compiles_successfully(self):
        result = compile_regex_with_timeout(_fast_compile, r"[a-z]+")
        assert result == "compiled:[a-z]+"

    def test_timeout_raises_value_error(self):
        with (
            patch("vllm.envs.VLLM_REGEX_COMPILATION_TIMEOUT_S", 0.5),
            pytest.raises(ValueError, match="timed out"),
        ):
            compile_regex_with_timeout(_slow_compile, r"(a+)+b")

    def test_timeout_disabled_when_zero(self):
        with patch("vllm.envs.VLLM_REGEX_COMPILATION_TIMEOUT_S", 0):
            result = compile_regex_with_timeout(_fast_compile, r"(a+)+b")
        assert result == "compiled:(a+)+b"

    def test_compilation_error_propagates(self):
        with pytest.raises(RuntimeError, match="compilation failed"):
            compile_regex_with_timeout(_failing_compile, r"bad")

    def test_pattern_included_in_error_message(self):
        pattern = r"(a+)+b"
        with (
            patch("vllm.envs.VLLM_REGEX_COMPILATION_TIMEOUT_S", 0.5),
            pytest.raises(ValueError, match=r"\(a\+\)\+b"),
        ):
            compile_regex_with_timeout(_slow_compile, pattern)


class TestNoLingeringProcesses:
    """Regression tests for GHSA-g773-5cq5-5j53.

    Verifies that timed-out compilation subprocesses are killed before
    the timeout error is returned, and that repeated timeouts do not
    accumulate lingering workers.
    """

    def test_no_lingering_after_timeout(self):
        """Child process must be dead when ValueError is raised."""
        with (
            patch("vllm.envs.VLLM_REGEX_COMPILATION_TIMEOUT_S", 0.5),
            contextlib.suppress(ValueError),
        ):
            compile_regex_with_timeout(_slow_compile, "linger_test")

        time.sleep(0.1)
        result = subprocess.run(
            ["pgrep", "-f", "_slow_compile"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0, (
            f"Lingering process found after timeout: {result.stdout}"
        )

    def test_no_accumulation_after_sequential_timeouts(self):
        """N sequential timeouts must leave 0 lingering processes."""
        n = 4
        with patch("vllm.envs.VLLM_REGEX_COMPILATION_TIMEOUT_S", 0.3):
            for i in range(n):
                with contextlib.suppress(ValueError):
                    compile_regex_with_timeout(_slow_compile, f"accum_pattern_{i}")

        time.sleep(0.1)
        result = subprocess.run(
            ["pgrep", "-f", "_slow_compile"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0, (
            f"Lingering processes after {n} timeouts: {result.stdout}"
        )

    def test_fast_compilation_returns_normally(self):
        """Fast compiles must still return correctly."""
        with patch("vllm.envs.VLLM_REGEX_COMPILATION_TIMEOUT_S", 5):
            results = []
            for i in range(4):
                results.append(compile_regex_with_timeout(_fast_compile, f"fast_{i}"))
        assert results == [f"compiled:fast_{i}" for i in range(4)]

    def test_semaphore_limits_concurrent_compiles(self):
        """With MAX_CONCURRENT=1, only one compile can run at a time."""
        with (
            patch("vllm.envs.VLLM_REGEX_COMPILATION_TIMEOUT_S", 0.3),
            patch("vllm.envs.VLLM_REGEX_COMPILATION_MAX_CONCURRENT", 1),
        ):
            import vllm.v1.structured_output.utils as utils_mod

            old_sema = utils_mod._compile_semaphore
            utils_mod._compile_semaphore = None
            try:
                t0 = time.perf_counter()
                timeouts = 0
                for i in range(3):
                    with contextlib.suppress(ValueError):
                        compile_regex_with_timeout(_slow_compile, f"sema_{i}")
                    timeouts += 1
                elapsed = time.perf_counter() - t0
                assert timeouts == 3
                assert elapsed >= 0.9, (
                    f"Expected sequential execution (~0.9s), got {elapsed:.2f}s"
                )
            finally:
                utils_mod._compile_semaphore = old_sema

    def test_process_killed_on_sigkill(self):
        """Verify the child PID no longer exists after timeout."""
        child_pids = []
        original_start = None

        import multiprocessing.process as mp_proc

        original_start = mp_proc.BaseProcess.start

        def tracking_start(self):
            original_start(self)
            if self.pid:
                child_pids.append(self.pid)

        with (
            patch("vllm.envs.VLLM_REGEX_COMPILATION_TIMEOUT_S", 0.5),
            patch.object(mp_proc.BaseProcess, "start", tracking_start),
            contextlib.suppress(ValueError),
        ):
            compile_regex_with_timeout(_slow_compile, "kill_test")

        time.sleep(0.1)
        for pid in child_pids:
            try:
                os.kill(pid, 0)
                pytest.fail(f"Child process {pid} still alive after timeout")
            except ProcessLookupError:
                pass
