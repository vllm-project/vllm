# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-OS parent-death watchdog for the EngineCore subprocess.

The EngineCore child is spawned via ``multiprocessing`` and ordinarily
relies on a cooperative shutdown path. The parent calls
``proc.terminate()`` (via :func:`vllm.v1.utils.shutdown` from a
``weakref.finalize`` or an explicit ``LLM.shutdown()``), which delivers
SIGTERM to the child. The child's existing SIGTERM handler in
``EngineCoreProc.run_engine_core`` flips ``shutdown_state`` and the
busy loop drains and exits cleanly.

That cooperative path requires the parent to be alive and well-behaved.
When the parent dies abnormally (SIGKILL, OOM-kill, segfault, or
``os._exit()`` from an embedding host), atexit handlers and weakref
finalizers never run. ``proc.terminate()`` is never called. The
EngineCore child orphans, holds GPU memory, and blocks subsequent
``LLM()`` instantiations. See vllm-project/vllm#19849, #17273, #1908.

This module installs an OS-level watchdog inside the EngineCore child
that fires when the parent's task struct is reaped, regardless of how
the parent died. The default handler raises SIGTERM to ``self``, which
reuses the existing cooperative SIGTERM handler. No new shutdown
decision is made by the child; only the existing path is reached.

Per-OS mechanism:

* **Linux**: ``prctl(PR_SET_PDEATHSIG, SIGTERM)``. The kernel delivers
  SIGTERM to this process the moment the parent exits. Effectively
  zero latency.
* **macOS**: ``kqueue`` registration on ``EVFILT_PROC | NOTE_EXIT``
  watching the parent PID, polled by a daemon thread. Latency is the
  thread's wakeup time, typically <50ms.
* **Windows**: ``CreateJobObject`` with
  ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` is the canonical primitive,
  but it must be set up by the parent before the child is assigned.
  As a self-installable child-side fallback we open a handle to the
  parent process and wait on it from a daemon thread. When the wait
  returns, the parent has exited and the watchdog fires. Latency is
  the wait granularity, typically <100ms.

The watchdog is idempotent: calling :func:`install_parent_death_watchdog`
more than once is a no-op after the first successful installation.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
from collections.abc import Callable

from vllm.logger import init_logger

logger = init_logger(__name__)

_INSTALLED = False
_LOCK = threading.Lock()


def _default_on_parent_death() -> None:
    """Default watchdog handler that raises SIGTERM to self.

    Reuses the EngineCore's existing SIGTERM handler (installed in
    ``EngineCoreProc.run_engine_core``) so the cooperative shutdown
    path runs: ``shutdown_state = REQUESTED`` then busy loop drains
    then ``finally`` calls ``engine_core.shutdown()`` for resource
    cleanup.

    No new shutdown decision is made by the child; the watchdog only
    routes "parent is verifiably dead" into the same path the parent
    would have used had it been able to call ``proc.terminate()``.
    """
    logger.warning(
        "EngineCore parent process died; initiating cooperative shutdown "
        "via SIGTERM to self (pid=%d).",
        os.getpid(),
    )
    try:
        os.kill(os.getpid(), signal.SIGTERM)
    except OSError as e:
        # Last resort: if signaling self fails, hard-exit so we don't
        # leak the GPU. This branch is not expected to fire in
        # practice; kill(self, SIGTERM) succeeds whenever the process
        # is alive.
        logger.error(
            "Self-SIGTERM failed (%s); exiting hard to avoid GPU leak.", e
        )
        os._exit(1)


def install_parent_death_watchdog(
    on_parent_death: Callable[[], None] | None = None,
) -> bool:
    """Install a watchdog that triggers when this process's parent dies.

    The watchdog fires when the parent process is reaped by the kernel,
    regardless of how the parent died (graceful exit, SIGKILL, OOM-kill,
    segfault, ``os._exit()`` from an embedding host).

    Args:
        on_parent_death: Callback invoked in a watchdog-owned context
            when the parent is detected dead. Defaults to
            :func:`_default_on_parent_death`, which raises SIGTERM to
            this process to reuse the existing cooperative shutdown
            path. A custom handler is useful for tests.

    Returns:
        ``True`` if the watchdog was installed by this call. ``False``
        if a watchdog had already been installed (idempotent), or if
        the OS does not support a parent-death primitive.

    Notes:
        Safe to call from any thread. Should be called from the
        EngineCore subprocess's main as early as possible, before
        model load, so the watchdog covers the long startup window
        during which a parent crash would otherwise orphan the child.

        On Linux, ``prctl(PR_SET_PDEATHSIG)`` is reset on
        ``execve()``. EngineCore does not exec, so this is not a
        concern in the current code path. If a future refactor adds
        an exec, the watchdog must be re-installed in the post-exec
        process.
    """
    global _INSTALLED
    handler = on_parent_death or _default_on_parent_death

    with _LOCK:
        if _INSTALLED:
            logger.debug("Parent-death watchdog already installed; skipping.")
            return False

        platform = sys.platform
        try:
            if platform == "linux":
                _install_linux(handler)
            elif platform == "darwin":
                _install_macos(handler)
            elif platform == "win32":
                _install_windows(handler)
            else:
                logger.warning(
                    "Parent-death watchdog not implemented for platform "
                    "%s; EngineCore may orphan if the parent dies "
                    "abnormally.",
                    platform,
                )
                return False
        except Exception as e:
            # Never block EngineCore startup on watchdog install failure.
            # Log and continue without the watchdog; caller can still
            # rely on the cooperative shutdown path.
            logger.warning(
                "Failed to install parent-death watchdog (%s); EngineCore "
                "may orphan if the parent dies abnormally.",
                e,
            )
            return False

        _INSTALLED = True
        logger.debug("Parent-death watchdog installed (platform=%s).", platform)
        return True


# ---------------------------------------------------------------------------
# Linux: prctl(PR_SET_PDEATHSIG, SIGTERM)
# ---------------------------------------------------------------------------

# From <linux/prctl.h>. Stable since 2.6.0; safe to hardcode.
_PR_SET_PDEATHSIG = 1


def _install_linux(handler: Callable[[], None]) -> None:
    """Linux: ask the kernel to send SIGTERM when the parent exits.

    The kernel delivers SIGTERM the instant the parent's task struct
    is reaped; no userspace polling required. The signal is then
    handled by whatever SIGTERM handler is installed at the time the
    parent dies (typically EngineCore's cooperative handler in
    ``run_engine_core``, installed shortly after this watchdog).

    On Linux, custom ``on_parent_death`` callbacks are not supported:
    the kernel-delivered SIGTERM is handled by whatever signal handler
    is current when the parent dies, not by anything this function
    installs at registration time. Use the default handler
    (SIGTERM-to-self) on Linux. Custom handlers continue to work via
    the daemon-thread mechanism on macOS and Windows.
    """
    import ctypes
    import ctypes.util

    libc_name = ctypes.util.find_library("c") or "libc.so.6"
    libc = ctypes.CDLL(libc_name, use_errno=True)
    rc = libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    if rc != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, f"prctl(PR_SET_PDEATHSIG, SIGTERM) failed: errno={errno}")

    # Edge case: if the parent had already exited by the time prctl
    # ran (race: spawn, parent dies, child gets to prctl), the kernel
    # will not back-fire the signal. Check now.
    try:
        if os.getppid() == 1:
            # Parent already reaped (we've been re-parented to init).
            handler()
            return
    except OSError:
        pass


# ---------------------------------------------------------------------------
# macOS: kqueue EVFILT_PROC | NOTE_EXIT on parent pid
# ---------------------------------------------------------------------------


def _install_macos(handler: Callable[[], None]) -> None:
    """macOS: register kqueue NOTE_EXIT and poll from a daemon thread.

    macOS has no equivalent of ``PR_SET_PDEATHSIG``. The supported
    pattern is to register an ``EVFILT_PROC`` filter for the parent's
    PID and wait for ``NOTE_EXIT``. We do this from a daemon thread
    so the EngineCore main loop is unaffected.

    The thread exits naturally when ``handler()`` returns (the default
    handler raises SIGTERM to self, which causes the process to exit
    cleanly, taking the daemon thread with it).
    """
    import select

    parent_pid = os.getppid()
    if parent_pid == 1:
        # Already orphaned at install time.
        handler()
        return

    kq = select.kqueue()
    # NOTE_EXIT = 0x80000000, EVFILT_PROC = -5
    # select.KQ_NOTE_EXIT and select.KQ_FILTER_PROC are exposed by stdlib.
    event = select.kevent(
        ident=parent_pid,
        filter=select.KQ_FILTER_PROC,
        flags=select.KQ_EV_ADD | select.KQ_EV_ENABLE,
        fflags=select.KQ_NOTE_EXIT,
    )
    # Register; control event will be returned on parent exit.
    kq.control([event], 0)

    def _watch():
        try:
            # Block until the parent exits. Long timeout chunks let the
            # thread wake periodically so it can notice if the kqueue
            # has been closed by a separate cleanup path (defensive).
            while True:
                events = kq.control(None, 1, 5.0)
                if events:
                    break
                # Defensive: if PPID has changed to 1, the parent was
                # reaped and the kqueue may have missed it (unlikely
                # but cheap to check).
                if os.getppid() == 1:
                    break
        except Exception as e:
            logger.warning("kqueue watcher errored: %s", e)
            return
        finally:
            try:
                kq.close()
            except OSError:
                pass
        try:
            handler()
        except Exception:
            logger.exception("parent-death handler raised; exiting hard.")
            os._exit(1)

    t = threading.Thread(
        target=_watch,
        name="vllm-parent-death-watchdog",
        daemon=True,
    )
    t.start()


# ---------------------------------------------------------------------------
# Windows: wait on parent process handle from daemon thread
# ---------------------------------------------------------------------------


def _install_windows(handler: Callable[[], None]) -> None:
    """Windows: open a handle to the parent and wait on it.

    The cleanest Windows primitive is a Job Object with
    ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``, but that must be set up
    by the parent before the child is assigned. As a self-installable
    fallback the child opens ``PROCESS_SYNCHRONIZE`` access on the
    parent and waits on the handle from a daemon thread. When the
    wait returns, the parent has exited.

    For a parent-side Job Object integration (better latency, tree
    propagation), see the follow-up noted in PR_DESCRIPTION.md.
    """
    import ctypes
    from ctypes import wintypes

    parent_pid = os.getppid()
    if parent_pid in (0, 1):
        handler()
        return

    PROCESS_SYNCHRONIZE = 0x00100000
    SYNCHRONIZE = 0x00100000
    INFINITE = 0xFFFFFFFF
    WAIT_OBJECT_0 = 0x00000000

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    OpenProcess = kernel32.OpenProcess
    OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    OpenProcess.restype = wintypes.HANDLE
    WaitForSingleObject = kernel32.WaitForSingleObject
    WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    WaitForSingleObject.restype = wintypes.DWORD
    CloseHandle = kernel32.CloseHandle
    CloseHandle.argtypes = [wintypes.HANDLE]
    CloseHandle.restype = wintypes.BOOL

    handle = OpenProcess(
        PROCESS_SYNCHRONIZE | SYNCHRONIZE, False, parent_pid
    )
    if not handle:
        err = ctypes.get_last_error()
        raise OSError(err, f"OpenProcess(parent={parent_pid}) failed: {err}")

    def _watch():
        try:
            rc = WaitForSingleObject(handle, INFINITE)
            if rc != WAIT_OBJECT_0:
                logger.warning(
                    "WaitForSingleObject on parent returned 0x%x; firing "
                    "watchdog handler anyway.",
                    rc,
                )
        finally:
            CloseHandle(handle)
        try:
            handler()
        except Exception:
            logger.exception("parent-death handler raised; exiting hard.")
            os._exit(1)

    t = threading.Thread(
        target=_watch,
        name="vllm-parent-death-watchdog",
        daemon=True,
    )
    t.start()


def _reset_for_testing() -> None:
    """Reset module state. Test-only."""
    global _INSTALLED
    with _LOCK:
        _INSTALLED = False
