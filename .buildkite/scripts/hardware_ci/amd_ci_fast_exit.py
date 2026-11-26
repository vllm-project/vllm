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

When a test outlives pytest's `faulthandler_timeout`, faulthandler prints only this
process's Python stacks, while the hang is often in a child (an EngineCore stuck in HIP
init). So the plugin then also prints, for every descendant process, its native and
Python stacks (`py-spy dump --native`, else gdb) and each thread's kernel wait channel.
This is best-effort and bounded: it never changes the result.
"""

import atexit
import ctypes
import faulthandler
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter

import pytest

_LIB = os.environ.get(
    "VLLM_CI_FAULT_TRAP_LIB", "/usr/local/lib/libamd_ci_fault_trap.so"
)
# The real stderr, taken before pytest captures fd 2, so a fault report is never
# swallowed by a test's captured output.
_stderr_fd = os.dup(2)
_outer = None
_session = None

_MAX_PROCESSES = 16
_TOOL_TIMEOUT_S = 60
_DUMP_BUDGET_S = 600
_MAX_BYTES_PER_PROCESS = 64 * 1024


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


def _wait_channels(pid: int) -> str:
    counts: Counter[str] = Counter()
    try:
        for tid in os.listdir(f"/proc/{pid}/task"):
            try:
                with open(f"/proc/{pid}/task/{tid}/wchan") as f:
                    counts[f.read().strip() or "running"] += 1
            except OSError:
                continue
    except OSError:
        return "?"
    return ", ".join(f"{name} x{n}" for name, n in counts.most_common())


def _native_stacks(pid: int, timeout: float) -> str:
    commands = []
    if py_spy := shutil.which("py-spy"):
        commands.append([py_spy, "dump", "--native", "--pid", str(pid)])
    if gdb := shutil.which("gdb"):
        commands.append(
            [gdb, "-p", str(pid), "-batch", "-nx", "-ex", "thread apply all bt"]
        )
    if not commands:
        return "no py-spy or gdb in the image\n"
    errors = []
    for cmd in commands:
        try:
            out = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{os.path.basename(cmd[0])}: {exc}")
            continue
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout
        err = (out.stderr or out.stdout).strip().splitlines()[-1:]
        errors.append(f"{os.path.basename(cmd[0])}: exit {out.returncode} {err}")
    return "".join(f"stack dump failed: {e}\n" for e in errors)


def _dump_descendants(out_fd: int) -> None:
    """Print the stacks of every process below this one, best-effort."""

    def write(text: str) -> None:
        os.write(out_fd, text.encode(errors="replace"))

    try:
        import psutil

        deadline = time.monotonic() + _DUMP_BUDGET_S
        children = psutil.Process().children(recursive=True)
        write(f"[amd-ci] test timed out: stacks of {len(children)} child processes\n")
        for proc in children[:_MAX_PROCESSES]:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                write("[amd-ci] stack dump budget exhausted\n")
                break
            try:
                cmdline = " ".join(proc.cmdline()).replace("\n", " ")[:300]
                status = proc.status()
            except psutil.Error:
                continue
            stacks = _native_stacks(proc.pid, min(_TOOL_TIMEOUT_S, remaining))
            if len(stacks) > _MAX_BYTES_PER_PROCESS:
                stacks = stacks[:_MAX_BYTES_PER_PROCESS] + "\n... (truncated)\n"
            write(
                f"[amd-ci] ---- pid {proc.pid} ppid {proc.ppid()} ({status}): "
                f"{cmdline}\n[amd-ci] kernel wait channels: "
                f"{_wait_channels(proc.pid)}\n{stacks}"
            )
        write("[amd-ci] end of child process stacks\n")
    except Exception as exc:  # diagnostics must never affect the run
        write(f"[amd-ci] child stack dump failed: {exc!r}\n")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    timeout = 0.0
    if item.config is _outer:
        timeout = float(item.config.getini("faulthandler_timeout") or 0)
    if timeout <= 0:
        yield
        return
    # A second after faulthandler's own dump, so the two do not interleave.
    timer = threading.Timer(timeout + 1, _dump_descendants, args=(_stderr_fd,))
    timer.daemon = True
    timer.start()
    try:
        yield
    finally:
        timer.cancel()


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
