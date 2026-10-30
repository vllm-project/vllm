# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The AMD CI pytest plugin skips native teardown without changing pytest's result."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[2] / ".buildkite/scripts/hardware_ci"

# An object that segfaults when the interpreter tears its module down, after the
# summary, as native library destructors did in vllm/ci #96786.
CRASH_AT_TEARDOWN = """
import faulthandler

class _CrashOnTeardown:
    def __del__(self, _crash=faulthandler._sigsegv):
        _crash()

_keep = _CrashOnTeardown()
"""


def _run(tmp_path, body, *, plugin=True, env=None, args=()):
    (tmp_path / "test_case.py").write_text(CRASH_AT_TEARDOWN + body)
    run_env = {k: v for k, v in os.environ.items() if k != "PYTEST_ADDOPTS"}
    for name in ("ROCP_TOOL_LIBRARIES", "CUDA_INJECTION64_PATH", "VLLM_CI_FAST_EXIT"):
        run_env.pop(name, None)
    run_env["PYTHONPATH"] = str(PLUGIN_DIR)
    run_env.update(env or {})
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *args]
    if plugin:
        cmd += ["-p", "amd_ci_fast_exit"]
    return subprocess.run(
        [*cmd, "test_case.py"],
        cwd=tmp_path,
        env=run_env,
        capture_output=True,
        text=True,
    )


PASSING = "\ndef test_ok():\n    assert True\n"


def test_reproduces_the_teardown_crash_without_the_plugin(tmp_path):
    assert _run(tmp_path, PASSING, plugin=False).returncode == -signal.SIGSEGV


def test_a_passing_run_exits_zero_despite_the_teardown_crash(tmp_path):
    result = _run(tmp_path, PASSING)
    assert result.returncode == 0, result.stderr
    assert "leaving without native teardown" in result.stderr


def test_a_failing_run_keeps_its_exit_status(tmp_path):
    result = _run(tmp_path, "\ndef test_bad():\n    assert False\n")
    assert result.returncode == pytest.ExitCode.TESTS_FAILED


def test_python_exit_handlers_still_run(tmp_path):
    marker = tmp_path / "atexit-ran"
    body = (
        "\nimport atexit, pathlib\n"
        f"atexit.register(pathlib.Path({str(marker)!r}).touch)\n" + PASSING
    )
    assert _run(tmp_path, body).returncode == 0
    assert marker.exists()


@pytest.mark.parametrize(
    "env",
    [
        {"VLLM_CI_FAST_EXIT": "0"},
        {"ROCP_TOOL_LIBRARIES": "/tmp/libkernrec_rocm.so"},
        {"CUDA_INJECTION64_PATH": "/tmp/libkernrec.so"},
    ],
)
def test_off_when_disabled_or_a_kernel_recorder_needs_native_teardown(tmp_path, env):
    assert _run(tmp_path, PASSING, env=env).returncode == -signal.SIGSEGV


def test_a_nested_in_process_session_does_not_end_the_outer_one(tmp_path):
    body = (
        "\npytest_plugins = ['pytester']\n"
        "def test_inner_session(pytester):\n"
        "    pytester.makepyfile('def test_x():\\n    assert True\\n')\n"
        "    assert pytester.runpytest_inprocess('-p', 'amd_ci_fast_exit').ret == 0\n"
        "\ndef test_after_the_inner_session():\n"
        "    assert True\n"
    )
    result = _run(tmp_path, body)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout
