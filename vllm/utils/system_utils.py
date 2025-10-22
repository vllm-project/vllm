# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import contextlib
import os
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TextIO

try:
    import setproctitle
except ImportError:
    setproctitle = None  # type: ignore[assignment]

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

CYAN = "\033[1;36m"
RESET = "\033[0;0m"


# Environment variable utilities


def update_environment_variables(envs_dict: dict[str, str]):
    """Update multiple environment variables with logging."""
    for k, v in envs_dict.items():
        if k in os.environ and os.environ[k] != v:
            logger.warning(
                "Overwriting environment variable %s from '%s' to '%s'",
                k,
                os.environ[k],
                v,
            )
        os.environ[k] = v


@contextlib.contextmanager
def set_env_var(key: str, value: str) -> Iterator[None]:
    """Temporarily set an environment variable."""
    old = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


# File path utilities


def unique_filepath(fn: Callable[[int], Path]) -> Path:
    """Generate a unique file path by trying incrementing integers.

    Note: This function has a TOCTOU race condition.
    Caller should use atomic operations (e.g., open with 'x' mode)
    when creating the file to ensure thread safety.
    """
    i = 0
    while True:
        p = fn(i)
        if not p.exists():
            return p
        i += 1


# Process management utilities


def set_process_title(
    name: str, suffix: str = "", prefix: str = envs.VLLM_PROCESS_NAME_PREFIX
) -> None:
    """Set the current process title with optional suffix."""
    if setproctitle is None:
        return
    if suffix:
        name = f"{name}_{suffix}"
    setproctitle.setproctitle(f"{prefix}::{name}")


def _add_prefix(file: TextIO, worker_name: str, pid: int) -> None:
    """Add colored prefix to file output for log decoration."""
    prefix = f"{CYAN}({worker_name} pid={pid}){RESET} "
    file_write = file.write

    def write_with_prefix(s: str):
        if not s:
            return
        if file.start_new_line:  # type: ignore[attr-defined]
            file_write(prefix)
        idx = 0
        while (next_idx := s.find("\n", idx)) != -1:
            next_idx += 1
            file_write(s[idx:next_idx])
            if next_idx == len(s):
                file.start_new_line = True  # type: ignore[attr-defined]
                return
            file_write(prefix)
            idx = next_idx
        file_write(s[idx:])
        file.start_new_line = False  # type: ignore[attr-defined]

    file.start_new_line = True  # type: ignore[attr-defined]
    file.write = write_with_prefix  # type: ignore[method-assign]


def decorate_logs(process_name: str | None = None) -> None:
    """Decorate stdout/stderr with process name and PID prefix."""
    from vllm.utils import get_mp_context

    if process_name is None:
        process_name = get_mp_context().current_process().name
    pid = os.getpid()
    _add_prefix(sys.stdout, process_name, pid)
    _add_prefix(sys.stderr, process_name, pid)
