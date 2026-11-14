# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for regex compilation timeout guard.

Verifies that adversarial regex patterns that would cause exponential
DFA state-space explosion are rejected with a timeout rather than
hanging indefinitely, and that timed-out compilation work is killed
(no lingering processes).
"""

import os
import time

import pytest

import vllm.v1.structured_output.utils as utils_mod
from vllm.v1.structured_output.utils import (
    _outlines_compile_index,
    compile_regex_with_timeout,
    shutdown_regex_compile_pool,
)

pytestmark = pytest.mark.skip_global_cleanup


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


def _pid_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _worker():
    assert utils_mod._compile_pool is not None
    assert len(utils_mod._compile_pool._workers) == 1
    return utils_mod._compile_pool._workers[0]


@pytest.fixture(autouse=True)
def _reset_compile_pool():
    # Compiler workers are started with get_mp_context(). Pin spawn for the
    # suite so a CUDA-free dev machine does not fork a multithreaded pytest
    # process. The CUDA-init test below still enters with fork and checks
    # that the policy overrides it.
    saved = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    shutdown_regex_compile_pool()
    yield
    shutdown_regex_compile_pool()
    if saved is None:
        os.environ.pop("VLLM_WORKER_MULTIPROC_METHOD", None)
    else:
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = saved


class TestCompileRegexWithTimeout:
    """Unit tests for the compile_regex_with_timeout utility."""

    def test_normal_regex_compiles_successfully(self):
        result = compile_regex_with_timeout(_fast_compile, r"[a-z]+", pattern=r"[a-z]+")
        assert result == "compiled:[a-z]+"

    def test_timeout_raises_value_error(self):
        with (
            pytest.MonkeyPatch.context() as monkeypatch,
            pytest.raises(ValueError, match="timed out"),
        ):
            monkeypatch.setattr("vllm.envs.VLLM_REGEX_COMPILATION_TIMEOUT_S", 0.5)
            compile_regex_with_timeout(_slow_compile, r"(a+)+b", pattern=r"(a+)+b")

    def test_timeout_disabled_when_zero(self):
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr("vllm.envs.VLLM_REGEX_COMPILATION_TIMEOUT_S", 0)
            result = compile_regex_with_timeout(
                _fast_compile, r"(a+)+b", pattern=r"(a+)+b"
            )
        assert result == "compiled:(a+)+b"

    def test_compilation_error_propagates(self):
        with pytest.raises(RuntimeError, match="compilation failed"):
            compile_regex_with_timeout(_failing_compile, r"bad", pattern="bad")

    def test_pattern_included_in_error_message(self):
        pattern = r"(a+)+b"
        with (
            pytest.MonkeyPatch.context() as monkeypatch,
            pytest.raises(ValueError, match=r"\(a\+\)\+b"),
        ):
            monkeypatch.setattr("vllm.envs.VLLM_REGEX_COMPILATION_TIMEOUT_S", 0.5)
            compile_regex_with_timeout(_slow_compile, pattern, pattern=pattern)

    def test_compilation_error_reuses_worker(self):
        with pytest.raises(RuntimeError, match="compilation failed"):
            compile_regex_with_timeout(_failing_compile, r"bad", pattern="bad")
        worker = _worker()
        pid = worker.last_pid
        assert _pid_alive(pid)

        result = compile_regex_with_timeout(_fast_compile, "next", pattern="next")
        assert result == "compiled:next"
        assert worker.last_pid == pid
        assert _pid_alive(pid)


class TestNoLingeringProcesses:
    """Timed-out compilation workers are killed and not reused."""

    def test_no_lingering_after_timeout(self):
        with (
            pytest.MonkeyPatch.context() as monkeypatch,
            pytest.raises(ValueError, match="timed out"),
        ):
            monkeypatch.setattr("vllm.envs.VLLM_REGEX_COMPILATION_TIMEOUT_S", 0.5)
            compile_regex_with_timeout(
                _slow_compile, "linger_test", pattern="linger_test"
            )

        pid = _worker().last_pid
        time.sleep(0.1)
        assert not _pid_alive(pid)

    def test_no_accumulation_after_sequential_timeouts(self):
        pids: list[int] = []
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr("vllm.envs.VLLM_REGEX_COMPILATION_TIMEOUT_S", 0.5)
            for i in range(2):
                with pytest.raises(ValueError, match="timed out"):
                    compile_regex_with_timeout(
                        _slow_compile,
                        f"accum_pattern_{i}",
                        pattern=f"accum_pattern_{i}",
                    )
                pid = _worker().last_pid
                assert pid is not None
                pids.append(pid)

        time.sleep(0.1)
        assert pids[0] != pids[1]
        assert all(not _pid_alive(pid) for pid in pids)

    def test_fast_compilation_reuses_worker(self):
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr("vllm.envs.VLLM_REGEX_COMPILATION_TIMEOUT_S", 5)
            first = compile_regex_with_timeout(
                _fast_compile, "fast_0", pattern="fast_0"
            )
            pid = _worker().last_pid
            second = compile_regex_with_timeout(
                _fast_compile, "fast_1", pattern="fast_1"
            )
        assert first == "compiled:fast_0"
        assert second == "compiled:fast_1"
        assert _worker().last_pid == pid
        assert _pid_alive(pid)

    def test_spawn_forced_after_cuda_init_kills_on_timeout(self):
        saved = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"
        try:
            with (
                pytest.MonkeyPatch.context() as monkeypatch,
                pytest.raises(ValueError, match="timed out"),
            ):
                monkeypatch.setattr(
                    "vllm.utils.system_utils.cuda_is_initialized",
                    lambda: True,
                )
                monkeypatch.setattr("vllm.envs.VLLM_REGEX_COMPILATION_TIMEOUT_S", 0.5)
                compile_regex_with_timeout(
                    _slow_compile, "spawn_kill", pattern="spawn_kill"
                )
            worker = _worker()
            assert worker.start_method == "spawn"
            time.sleep(0.1)
            assert not _pid_alive(worker.last_pid)
        finally:
            if saved is None:
                os.environ.pop("VLLM_WORKER_MULTIPROC_METHOD", None)
            else:
                os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = saved


class TestBackendRoundTrip:
    """Backend callables must survive a process boundary."""

    def test_lm_format_enforcer_parser_round_trip(self):
        lmformatenforcer = pytest.importorskip("lmformatenforcer")
        parser = compile_regex_with_timeout(
            lmformatenforcer.RegexParser,
            r"[a-z]+",
            pattern=r"[a-z]+",
        )
        assert isinstance(parser, lmformatenforcer.RegexParser)

    def test_outlines_index_round_trip(self):
        outlines_core = pytest.importorskip("outlines_core")
        vocabulary = outlines_core.Vocabulary(0, {"a": [1]})
        index = compile_regex_with_timeout(
            _outlines_compile_index,
            r"a+",
            vocabulary,
            pattern="a+",
        )
        assert isinstance(index, outlines_core.Index)
