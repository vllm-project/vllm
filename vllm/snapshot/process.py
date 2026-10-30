# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import ctypes
import errno
import os
from pathlib import Path

# pidfd_getfd(2) is syscall 438 on the generic Linux syscall table used by
# x86_64 and aarch64. Some CPython builds omit os.pidfd_getfd, so use the
# raw syscall as a fallback when the wrapper is unavailable.
_SYS_PIDFD_GETFD = 438


def process_starttime(pid: int) -> int:
    stat = (Path("/proc") / str(pid) / "stat").read_text()
    fields = stat[stat.rfind(")") + 2 :].split()
    if len(fields) <= 19:
        raise RuntimeError(f"invalid process stat for PID {pid}")
    return int(fields[19])


def _pidfd_getfd(pidfd: int, target_fd: int) -> int:
    getter = getattr(os, "pidfd_getfd", None)
    if getter is not None:
        return getter(pidfd, target_fd)
    if os.uname().machine not in ("x86_64", "aarch64"):
        raise OSError(errno.ENOSYS, "pidfd_getfd syscall is unavailable")
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.syscall(_SYS_PIDFD_GETFD, pidfd, target_fd, 0)
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, f"pidfd_getfd: {os.strerror(error)}")
    return result


def _duplicate_fd(source_pid: int, target_fd: int) -> int:
    # /proc/<pid>/fd open creates a new file description, which drops O_APPEND
    # and fails with ENXIO for socket-backed stdio (systemd journal). pidfd_getfd
    # duplicates the real description when ptrace access is available.
    pidfd_open = getattr(os, "pidfd_open", None)
    if pidfd_open is not None:
        try:
            pidfd = pidfd_open(source_pid)
            try:
                return _pidfd_getfd(pidfd, target_fd)
            finally:
                os.close(pidfd)
        except OSError:
            pass
    return os.open(
        f"/proc/{source_pid}/fd/{target_fd}",
        os.O_WRONLY | os.O_CLOEXEC,
    )


def rebind_stdio(source_pid: int, expected_starttime: int | None = None) -> None:
    source_starttime = process_starttime(source_pid)
    if expected_starttime is not None and source_starttime != expected_starttime:
        raise RuntimeError("stdio source process identity changed")
    source_fds = []
    try:
        for target_fd in (1, 2):
            source_fds.append(_duplicate_fd(source_pid, target_fd))
        if process_starttime(source_pid) != source_starttime:
            raise RuntimeError("stdio source process identity changed")
        for source_fd, target_fd in zip(source_fds, (1, 2)):
            os.dup2(source_fd, target_fd, inheritable=True)
    finally:
        for source_fd in source_fds:
            os.close(source_fd)
