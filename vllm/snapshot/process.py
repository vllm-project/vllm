# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
from pathlib import Path


def process_starttime(pid: int) -> int:
    stat = (Path("/proc") / str(pid) / "stat").read_text()
    fields = stat[stat.rfind(")") + 2 :].split()
    if len(fields) <= 19:
        raise RuntimeError(f"invalid process stat for PID {pid}")
    return int(fields[19])


def _duplicate_fd(source_pid: int, target_fd: int) -> int:
    # /proc/<pid>/fd open creates a new file description, which drops O_APPEND
    # and fails with ENXIO for socket-backed stdio (systemd journal). pidfd_getfd
    # duplicates the real description when ptrace access is available.
    pidfd_open = getattr(os, "pidfd_open", None)
    pidfd_getfd = getattr(os, "pidfd_getfd", None)
    if pidfd_open is not None and pidfd_getfd is not None:
        try:
            pidfd = pidfd_open(source_pid)
            try:
                return pidfd_getfd(pidfd, target_fd)
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
