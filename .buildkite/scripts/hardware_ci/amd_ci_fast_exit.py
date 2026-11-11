# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""pytest plugin for AMD CI jobs: leave without native teardown once pytest is done.

A pytest process can crash in native library destructors after its summary:
vllm/ci #96786 printed "236 passed" and then died with SIGSEGV (exit 139) during
interpreter shutdown, past the point where faulthandler reports. Once the outermost
session has finished, this runs the Python exit handlers (engine shutdown, Ray,
multiprocessing cleanup), flushes, and exits with pytest's own status, skipping
module teardown and native destructors. pytest's result is never changed, and a
command chain still stops on a real failure.

Off when a kernel recorder is loaded (it writes its trace during native teardown)
or with VLLM_CI_FAST_EXIT=0.

It also loads a native fault reporter (amd_ci_fault_trap.c, built into the CI image):
a SIGSEGV, SIGBUS or SIGABRT prints the native backtrace, which names the library that
faulted, after Python's traceback. VLLM_CI_FAULT_TRAP_LIB overrides the library's path.
"""

import atexit
import ctypes
import faulthandler
import os
import sys

import pytest

_LIB = os.environ.get(
    "VLLM_CI_FAULT_TRAP_LIB", "/usr/local/lib/libamd_ci_fault_trap.so"
)
# The real stderr, taken before pytest captures fd 2, so a fault report is never
# swallowed by a test's captured output.
_stderr_fd = os.dup(2)
_outer = None
_session = None


def _recorder_loaded() -> bool:
    return bool(
        os.environ.get("ROCP_TOOL_LIBRARIES")
        or os.environ.get("CUDA_INJECTION64_PATH")
        or "rocprofiler" in os.environ.get("LD_PRELOAD", "")
    )


def _install_fault_reporter(lib) -> None:
    # Python's faulthandler restores the handler it replaced when it is disabled, so
    # take its place underneath it: it prints the Python traceback, then hands the
    # fault on.
    python_handler = faulthandler.is_enabled()
    if python_handler:
        faulthandler.disable()
    if lib.amd_ci_trap_install(_stderr_fd):
        os.write(_stderr_fd, b"[amd-ci] could not install the native fault reporter\n")
    if python_handler:
        faulthandler.enable(file=_stderr_fd)


@pytest.hookimpl(trylast=True)  # after pytest's faulthandler plugin has enabled it
def pytest_configure(config):
    global _outer
    if _outer is not None:  # pytester runs nested sessions in-process: act on the outer
        return
    _outer = config
    try:
        lib = ctypes.CDLL(_LIB)
    except OSError as exc:  # an image without it: run the tests without the reporter
        os.write(_stderr_fd, f"[amd-ci] no native fault reporter: {exc}\n".encode())
        return
    lib.amd_ci_trap_install.argtypes = [ctypes.c_int]
    _install_fault_reporter(lib)


def pytest_sessionfinish(session, exitstatus):
    global _session
    if session.config is _outer:
        _session = session


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    if config is not _outer or _session is None:
        return
    if os.environ.get("VLLM_CI_FAST_EXIT") == "0" or _recorder_loaded():
        return
    status = int(_session.exitstatus)
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        atexit._run_exitfuncs()
    except BaseException as exc:  # an exit handler failing must not mask the result
        print(f"[amd-ci] exit handler failed: {exc!r}", file=sys.stderr)
    print(
        f"[amd-ci] pytest finished with exit {status}; leaving without native teardown",
        file=sys.stderr,
        flush=True,
    )
    sys.stdout.flush()
    os._exit(status)
