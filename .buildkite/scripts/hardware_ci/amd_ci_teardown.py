# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""pytest plugin for AMD CI jobs: a native fault after pytest finishes is not a failure.

A pytest process can crash in native library teardown (the ROCm runtime, HIP, Ray)
after its summary: vllm/ci #96786 printed "236 passed" and then died with SIGSEGV
(exit 139), past the point where faulthandler reports. Three layers:

- Fault reporter (amd_ci_fault_trap.c, built into the CI image): a native fault
  prints its native backtrace, naming the library. During the tests the fault then
  proceeds as before, so a crash in a test still fails it.
- Fault trap: once the outermost session has finished, a native fault in any thread
  exits with pytest's own status instead of the signal.
- Fast exit: after the Python exit handlers (engine shutdown, Ray, multiprocessing
  cleanup), leave with pytest's status, skipping module teardown and native
  destructors.

pytest's result is never changed, and a command chain still stops on a real
failure. Fast exit is off when a kernel recorder is loaded (it writes its trace
during native teardown); the trap still covers that teardown. VLLM_CI_FAST_EXIT=0
turns fast exit off. VLLM_CI_FAULT_TRAP_LIB overrides the library's path.
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
_trap = None


def _recorder_loaded() -> bool:
    return bool(
        os.environ.get("ROCP_TOOL_LIBRARIES")
        or os.environ.get("CUDA_INJECTION64_PATH")
        or "rocprofiler" in os.environ.get("LD_PRELOAD", "")
    )


def _warn(message: str) -> None:
    os.write(_stderr_fd, f"[amd-ci] {message}\n".encode())


def _load_trap():
    trap = ctypes.CDLL(_LIB)
    trap.amd_ci_trap_install.argtypes = [ctypes.c_int]
    trap.amd_ci_trap_arm.argtypes = [ctypes.c_int]
    trap.amd_ci_trap_disarm.argtypes = []
    return trap


def _install_trap() -> None:
    # Python's faulthandler restores the handler it replaced when disabled, so take
    # its place underneath it: it reports the Python traceback, then hands the fault on.
    python_handler = faulthandler.is_enabled()
    if python_handler:
        faulthandler.disable()
    if _trap.amd_ci_trap_install(_stderr_fd):
        _warn("could not install the native fault trap")
    if python_handler:
        faulthandler.enable(file=_stderr_fd)


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    global _outer, _trap
    if _trap is not None:  # another run in this process: its faults fail it again
        _trap.amd_ci_trap_disarm()
    # pytester runs nested sessions in-process; act on the outer one.
    if _outer is not None:
        return
    _outer = config
    if _trap is None:
        try:
            _trap = _load_trap()
        except OSError as exc:  # an image without it: run the tests without the trap
            _warn(f"native fault trap unavailable: {exc}")
            return
    _install_trap()


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_sessionfinish(session):
    # Outermost, so this runs once every sessionfinish hook, the summary included, has
    # completed. If one raises, the session never counts as finished and pytest
    # reports the error as usual.
    result = yield
    global _session
    if session.config is _outer:
        _session = session
        if _trap is not None:
            _trap.amd_ci_trap_arm(int(session.exitstatus))
    return result


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    global _outer, _session
    if config is not _outer or _session is None:
        return
    status = int(_session.exitstatus)
    if _trap is not None:
        _install_trap()  # pytest's faulthandler plugin has just reinstalled its own
        _trap.amd_ci_trap_arm(status)
    if os.environ.get("VLLM_CI_FAST_EXIT") == "0" or _recorder_loaded():
        # The trap stays armed through interpreter teardown; a later pytest.main()
        # in this process is a run of its own.
        _outer = _session = None
        return
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
