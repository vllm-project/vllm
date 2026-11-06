# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for EngineCore subprocess parent-death detection.

These tests demonstrate that the EngineCore child subprocess is reaped
when the parent process dies by any means, including SIGKILL,
uncaught exceptions, and ``os._exit()`` calls that bypass atexit
handlers and weakref finalizers.

Without the parent-death watchdog (``vllm/v1/engine/parent_death.py``),
the SIGKILL and ``os._exit()`` scenarios leak the EngineCore process
indefinitely; it continues to hold GPU memory and blocks subsequent
``LLM()`` instantiations. With the watchdog installed, the child
receives an OS-level signal when the parent's task struct exits and
self-terminates via the existing cooperative SIGTERM path.

Tracking issues: vllm-project/vllm#19849, #17273, #1908.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import psutil
import pytest

# Wait at most this long for the EngineCore subprocess to disappear.
# 10s gives PR_SET_PDEATHSIG (instant), kqueue NOTE_EXIT (~50ms), and
# JobObject (~100ms) plus the cooperative SIGTERM cleanup path
# (LMCache shutdown can take ~6.5s per #31252) ample headroom.
REAP_TIMEOUT_S = 15.0

# Tiny model. Keep test cycle short. Already pinned in other engine tests.
MODEL = "facebook/opt-125m"


def _engine_core_children(parent_pid: int) -> list[psutil.Process]:
    """Return live EngineCore subprocesses descended from ``parent_pid``.

    Identifies children by process name set via ``set_process_title``
    in ``EngineCoreProc.run_engine_core``. Falls back to cmdline scan
    if the process title is not yet set when polled.
    """
    try:
        parent = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return []
    out: list[psutil.Process] = []
    for child in parent.children(recursive=True):
        try:
            name = child.name()
            cmdline = " ".join(child.cmdline())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if name.startswith("EngineCore") or "EngineCore" in cmdline:
            out.append(child)
    return out


def _wait_for_pid_exit(pid: int, timeout: float) -> bool:
    """Poll ``os.kill(pid, 0)`` until the PID is gone or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            # Process exists but is owned by another uid (or has been
            # reaped and the PID was reused). Treat as "still alive"
            # to be conservative.
            pass
        time.sleep(0.05)
    return False


def _wrapper_script(scenario: str, pid_file: str) -> str:
    """Return a Python program that loads an LLM, writes the EngineCore
    PID to ``pid_file``, and then triggers ``scenario``.

    The wrapper is launched in a fresh ``python -c`` subprocess so the
    test driver can SIGKILL or otherwise terminate it without affecting
    the test runner.
    """
    return textwrap.dedent(f"""
        import os, sys, time, psutil
        # Quiet vLLM logs in test output.
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
        from vllm import LLM, SamplingParams

        llm = LLM(
            model={MODEL!r},
            enforce_eager=True,
            gpu_memory_utilization=0.3,
            max_model_len=256,
            disable_log_stats=True,
        )
        # Resolve EngineCore PID via psutil (cross-platform; no
        # dependency on vLLM internals).
        me = psutil.Process(os.getpid())
        engine_pid = None
        for child in me.children(recursive=True):
            try:
                if child.name().startswith("EngineCore") or \\
                        "EngineCore" in " ".join(child.cmdline()):
                    engine_pid = child.pid
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if engine_pid is None:
            sys.stderr.write("FATAL: EngineCore subprocess not found\\n")
            os._exit(2)
        with open({pid_file!r}, "w") as f:
            f.write(str(engine_pid))
        sys.stdout.write("READY\\n"); sys.stdout.flush()

        scenario = {scenario!r}
        if scenario == "sigkill_wait":
            # Wait to be killed externally.
            time.sleep(60)
        elif scenario == "os_exit":
            # Bypass atexit, weakref finalizers, and the cooperative
            # shutdown path entirely.
            os._exit(0)
        elif scenario == "uncaught_exception":
            raise RuntimeError("simulated parent crash")
        elif scenario == "cooperative":
            del llm
            sys.exit(0)
        else:
            sys.stderr.write(f"unknown scenario: {{scenario}}\\n")
            os._exit(3)
    """)


def _spawn_wrapper(scenario: str, tmp_path: Path) -> tuple[subprocess.Popen, int]:
    """Spawn the wrapper subprocess and block until it has written the
    EngineCore PID.

    Returns ``(wrapper_proc, engine_core_pid)``.
    """
    pid_file = tmp_path / "engine_pid.txt"
    proc = subprocess.Popen(
        [sys.executable, "-c", _wrapper_script(scenario, str(pid_file))],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    # Wait for "READY" line (model loaded, EngineCore spawned, PID written).
    deadline = time.monotonic() + 180.0  # opt-125m cold load + cuda init
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            raise RuntimeError(
                f"wrapper exited prematurely with code {proc.returncode}\n"
                f"stdout:\n{stdout}\nstderr:\n{stderr}"
            )
        if pid_file.exists():
            try:
                engine_pid = int(pid_file.read_text().strip())
                return proc, engine_pid
            except ValueError:
                time.sleep(0.1)
                continue
        time.sleep(0.2)
    proc.kill()
    raise RuntimeError("wrapper did not become READY within 180s")


@pytest.fixture
def cuda_required():
    """Skip when no GPU available; vLLM EngineCore needs an accelerator."""
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for EngineCore tests")
    except ImportError:
        pytest.skip("PyTorch not installed")


@pytest.mark.timeout(300)
def test_engine_core_reaped_on_parent_sigkill(cuda_required, tmp_path):
    """Parent SIGKILLed; EngineCore must self-terminate.

    SIGKILL bypasses every userspace cleanup hook (atexit, weakref
    finalize, ``LLM.shutdown``). Without the watchdog this leaks the
    EngineCore subprocess indefinitely. With the watchdog, the kernel
    delivers SIGTERM (Linux PR_SET_PDEATHSIG) or kqueue NOTE_EXIT
    (macOS) or job-object kill (Windows) to the child immediately
    when the parent task struct is reaped.
    """
    wrapper, engine_pid = _spawn_wrapper("sigkill_wait", tmp_path)
    try:
        wrapper.send_signal(signal.SIGKILL)
        wrapper.wait(timeout=10)
        assert _wait_for_pid_exit(engine_pid, REAP_TIMEOUT_S), (
            f"EngineCore (pid={engine_pid}) survived parent SIGKILL "
            f"for {REAP_TIMEOUT_S}s; orphaned subprocess. This is "
            "the bug the parent-death watchdog fixes."
        )
    finally:
        if _wait_for_pid_exit(engine_pid, 0.0) is False:
            # Defensive cleanup so a failing test doesn't leak GPU.
            try:
                os.kill(engine_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.timeout(300)
def test_engine_core_reaped_on_parent_os_exit(cuda_required, tmp_path):
    """Parent calls ``os._exit(0)``; EngineCore must self-terminate.

    ``os._exit`` skips atexit, finalizers, and any registered
    cooperative shutdown path; exactly the failure mode reported in
    #19849 example 2 ("if your main Python process is killed, the
    EngineCore child remains alive"). This scenario also covers
    OOM-kill and signal-7 segfault behavior, which similarly skip
    userspace cleanup.
    """
    wrapper, engine_pid = _spawn_wrapper("os_exit", tmp_path)
    wrapper.wait(timeout=10)
    assert wrapper.returncode == 0
    assert _wait_for_pid_exit(engine_pid, REAP_TIMEOUT_S), (
        f"EngineCore (pid={engine_pid}) survived parent os._exit "
        f"for {REAP_TIMEOUT_S}s; orphaned subprocess."
    )


@pytest.mark.timeout(300)
def test_engine_core_reaped_on_parent_uncaught_exception(cuda_required, tmp_path):
    """Parent dies via uncaught exception; EngineCore must self-terminate.

    Some embedding hosts swallow exceptions before atexit runs, others
    don't. The watchdog must work in both cases. We force a hard exit
    here via Python's default uncaught-exception handler so the test
    is deterministic across hosts.
    """
    wrapper, engine_pid = _spawn_wrapper("uncaught_exception", tmp_path)
    wrapper.wait(timeout=15)
    # Python returns 1 for uncaught exceptions.
    assert wrapper.returncode != 0
    assert _wait_for_pid_exit(engine_pid, REAP_TIMEOUT_S), (
        f"EngineCore (pid={engine_pid}) survived parent crash "
        f"for {REAP_TIMEOUT_S}s; orphaned subprocess."
    )


@pytest.mark.timeout(300)
def test_cooperative_shutdown_unchanged(cuda_required, tmp_path):
    """Existing cooperative shutdown path must continue to work.

    Maintainers will worry the watchdog interferes with the existing
    SIGTERM, then ``shutdown_state = REQUESTED``, then ``run_busy_loop``
    drain, then ``engine_core.shutdown()`` finally path. This test
    exercises that path directly: parent does ``del llm; sys.exit(0)``
    and the EngineCore must exit via the existing cooperative
    machinery, NOT via the watchdog.
    """
    wrapper, engine_pid = _spawn_wrapper("cooperative", tmp_path)
    wrapper.wait(timeout=30)
    assert wrapper.returncode == 0
    assert _wait_for_pid_exit(engine_pid, REAP_TIMEOUT_S), (
        f"EngineCore (pid={engine_pid}) survived cooperative shutdown; "
        "this would indicate the watchdog broke the existing path."
    )


# ---------------------------------------------------------------------------
# Watchdog primitive: exercises parent_death.install_parent_death_watchdog
# in isolation. Runs cross-platform without GPU, suitable for CI matrices
# that don't have CUDA.
# ---------------------------------------------------------------------------


@pytest.mark.timeout(60)
@pytest.mark.skipif(
    sys.platform not in ("linux", "darwin", "win32"),
    reason="parent-death watchdog supports linux/macos/windows",
)
def test_watchdog_primitive_kills_child_on_parent_sigkill(tmp_path):
    """Spawn a tiny child that installs the watchdog and sleeps. SIGKILL
    its parent, assert the child terminates within the reap timeout.

    No model load, no GPU. Exercises only the OS-level primitive.
    """
    pid_file = tmp_path / "child_pid.txt"
    parent_script = textwrap.dedent(f"""
        import os, sys, subprocess, time
        child_script = '''
import os, sys, time
sys.path.insert(0, {str(Path(__file__).resolve().parents[3])!r})
from vllm.v1.engine.parent_death import install_parent_death_watchdog
install_parent_death_watchdog()
with open({str(pid_file)!r}, "w") as f:
    f.write(str(os.getpid()))
time.sleep(60)
'''
        child = subprocess.Popen([sys.executable, "-c", child_script])
        # Block forever. Parent will be SIGKILLed externally.
        time.sleep(60)
    """)
    parent = subprocess.Popen([sys.executable, "-c", parent_script])
    # Wait for child to write its PID.
    deadline = time.monotonic() + 30.0
    child_pid = None
    while time.monotonic() < deadline:
        if pid_file.exists():
            try:
                child_pid = int(pid_file.read_text().strip())
                break
            except ValueError:
                pass
        time.sleep(0.05)
    if child_pid is None:
        parent.kill()
        pytest.fail("watchdog primitive child did not write PID within 30s")
    try:
        parent.send_signal(signal.SIGKILL)
        parent.wait(timeout=5)
        assert _wait_for_pid_exit(child_pid, REAP_TIMEOUT_S), (
            f"child (pid={child_pid}) survived parent SIGKILL; "
            "parent-death watchdog primitive is not firing."
        )
    finally:
        try:
            os.kill(child_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
