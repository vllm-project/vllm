# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The AMD CI pytest plugin keeps native teardown faults from changing the result."""

import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[2] / ".buildkite/scripts/hardware_ci"
IMAGE_LIB = Path("/usr/local/lib/libamd_ci_fault_trap.so")

# An object that segfaults when the interpreter tears its module down, after the
# summary, as native library destructors did in vllm/ci #96786.
CRASH_AT_TEARDOWN = """
import faulthandler

class _CrashOnTeardown:
    def __del__(self, _crash=faulthandler._sigsegv):
        _crash()

_keep = _CrashOnTeardown()
"""
PASSING = "\ndef test_ok():\n    assert True\n"
FAILING = "\ndef test_bad():\n    assert False\n"
# A real invalid read, as a native library would make.
CRASH_IN_EXIT_HANDLER = (
    "\nimport atexit, ctypes\natexit.register(ctypes.string_at, 0)\n"
)
NO_TRAP = {"VLLM_CI_FAULT_TRAP_LIB": "/nonexistent/libamd_ci_fault_trap.so"}


@pytest.fixture(scope="module")
def trap(tmp_path_factory):
    """The image's fault trap library, else one built from source."""
    if IMAGE_LIB.exists():
        return {"VLLM_CI_FAULT_TRAP_LIB": str(IMAGE_LIB)}
    if not shutil.which("cc"):
        pytest.skip("no fault trap library and no compiler to build one")
    lib = tmp_path_factory.mktemp("trap") / "libamd_ci_fault_trap.so"
    source = PLUGIN_DIR / "amd_ci_fault_trap.c"
    subprocess.run(
        ["cc", "-O2", "-shared", "-fPIC", "-o", str(lib), str(source)], check=True
    )
    return {"VLLM_CI_FAULT_TRAP_LIB": str(lib)}


def _run(tmp_path, body, env, *, plugin=True, teardown_crash=True, script=None):
    (tmp_path / "test_case.py").write_text(
        (CRASH_AT_TEARDOWN if teardown_crash else "") + body
    )
    run_env = {k: v for k, v in os.environ.items() if k != "PYTEST_ADDOPTS"}
    for name in ("ROCP_TOOL_LIBRARIES", "CUDA_INJECTION64_PATH", "VLLM_CI_FAST_EXIT"):
        run_env.pop(name, None)
    # A rocprofiler recorder preloaded into this run would turn fast exit off.
    preload = os.environ.get("LD_PRELOAD", "").split()
    run_env["LD_PRELOAD"] = " ".join(p for p in preload if "rocprofiler" not in p)
    run_env["PYTHONPATH"] = str(PLUGIN_DIR)
    run_env.update(env)
    if script is not None:  # a script that runs pytest in-process
        (tmp_path / "run.py").write_text(script)
        run_env["PYTEST_ADDOPTS"] = "-p amd_ci_teardown -p no:cacheprovider"
        cmd = [sys.executable, "run.py"]
    else:
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
        cmd += ["-p", "amd_ci_teardown"] if plugin else []
        cmd.append("test_case.py")
    return subprocess.run(
        cmd, cwd=tmp_path, env=run_env, capture_output=True, text=True
    )


def test_reproduces_the_teardown_crash_without_the_plugin(tmp_path):
    assert _run(tmp_path, PASSING, {}, plugin=False).returncode == -signal.SIGSEGV


@pytest.mark.parametrize("body, status", [(PASSING, 0), (FAILING, 1)])
def test_fast_exit_keeps_pytests_status(tmp_path, body, status):
    marker = tmp_path / "atexit-ran"
    body += (
        "\nimport atexit, pathlib\n"
        f"atexit.register(pathlib.Path({str(marker)!r}).touch)\n"
    )
    result = _run(tmp_path, body, NO_TRAP)
    assert result.returncode == status, result.stderr
    assert "leaving without native teardown" in result.stderr
    assert "native fault trap unavailable" in result.stderr
    assert marker.exists()  # Python exit handlers still run


@pytest.mark.parametrize("body, status", [(PASSING, 0), (FAILING, 1)])
def test_a_fault_in_an_exit_handler_keeps_pytests_status(tmp_path, trap, body, status):
    result = _run(tmp_path, CRASH_IN_EXIT_HANDLER + body, trap)
    assert result.returncode == status, result.stderr
    assert "native fault: SIGSEGV" in result.stderr
    assert "exiting with pytest's status" in result.stderr


def test_a_fault_in_another_thread_keeps_pytests_status(tmp_path, trap):
    body = (
        "\nimport atexit, ctypes, threading\n"
        "def _fault_in_a_thread():\n"
        "    t = threading.Thread(target=ctypes.string_at, args=(0,))\n"
        "    t.start()\n"
        "    t.join()\n"
        "atexit.register(_fault_in_a_thread)\n" + PASSING
    )
    result = _run(tmp_path, body, trap)
    assert result.returncode == 0, result.stderr
    assert "after pytest finished" in result.stderr


@pytest.mark.parametrize(
    "env", [{"VLLM_CI_FAST_EXIT": "0"}, {"ROCP_TOOL_LIBRARIES": "/tmp/libkernrec.so"}]
)
def test_without_fast_exit_the_trap_covers_interpreter_teardown(tmp_path, trap, env):
    result = _run(tmp_path, PASSING, {**trap, **env})
    assert result.returncode == 0, result.stderr
    assert "leaving without native teardown" not in result.stderr
    assert "after pytest finished" in result.stderr
    # And without the trap, the crash is the job's result as before.
    assert _run(tmp_path, PASSING, {**NO_TRAP, **env}).returncode == -signal.SIGSEGV


def test_a_fault_during_a_test_still_fails_and_is_reported(tmp_path, trap):
    body = "\nimport ctypes\ndef test_crash():\n    ctypes.string_at(0)\n"
    result = _run(tmp_path, body, trap, teardown_crash=False)
    assert result.returncode == -signal.SIGSEGV
    assert "native fault: SIGSEGV" in result.stderr
    assert "Fatal Python error" in result.stderr  # Python's traceback as well
    assert "after pytest finished" not in result.stderr


def test_an_error_in_the_summary_still_fails_the_run(tmp_path, trap):
    (tmp_path / "conftest.py").write_text(
        "def pytest_terminal_summary():\n    raise RuntimeError('summary broke')\n"
    )
    result = _run(tmp_path, PASSING, trap)
    assert result.returncode != 0
    assert "summary broke" in result.stdout + result.stderr


def test_a_later_in_process_run_is_not_covered_by_the_first(tmp_path, trap):
    (tmp_path / "test_crash.py").write_text(
        "import ctypes\ndef test_crash():\n    ctypes.string_at(0)\n"
    )
    script = (
        "import pytest, sys\n"
        "assert pytest.main(['test_case.py']) == 0\n"
        "sys.exit(pytest.main(['test_crash.py']))\n"
    )
    env = {**trap, "VLLM_CI_FAST_EXIT": "0"}
    result = _run(tmp_path, PASSING, env, teardown_crash=False, script=script)
    assert result.returncode == -signal.SIGSEGV, result.stderr
    assert "after pytest finished" not in result.stderr


def test_a_nested_in_process_session_does_not_end_the_outer_one(tmp_path, trap):
    body = (
        "\npytest_plugins = ['pytester']\n"
        "def test_inner_session(pytester):\n"
        "    pytester.makepyfile('def test_x():\\n    assert True\\n')\n"
        "    assert pytester.runpytest_inprocess('-p', 'amd_ci_teardown').ret == 0\n"
        "\ndef test_after_the_inner_session():\n"
        "    assert True\n"
    )
    result = _run(tmp_path, body, trap)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout
