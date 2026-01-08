# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import contextlib
import multiprocessing
import os
import signal
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TextIO

import psutil

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.ray.lazy_utils import is_in_ray_actor

from .platform_utils import cuda_is_initialized, xpu_is_initialized

logger = init_logger(__name__)

CYAN = "\033[0;36m"
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


@contextlib.contextmanager
def suppress_stdout():
    """
    Suppress stdout from C libraries at the file descriptor level.

    Only suppresses stdout, not stderr, to preserve error messages.
    Suppression is disabled when VLLM_LOGGING_LEVEL is set to DEBUG.

    Example:
        with suppress_stdout():
            # C library calls that would normally print to stdout
            torch.distributed.new_group(ranks, backend="gloo")
    """
    # Don't suppress if logging level is DEBUG
    if envs.VLLM_LOGGING_LEVEL == "DEBUG":
        yield
        return

    stdout_fd = sys.stdout.fileno()
    stdout_dup = os.dup(stdout_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)

    try:
        sys.stdout.flush()
        os.dup2(devnull_fd, stdout_fd)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(stdout_dup, stdout_fd)
        os.close(stdout_dup)
        os.close(devnull_fd)


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


def _maybe_force_spawn():
    """Check if we need to force the use of the `spawn` multiprocessing start
    method.
    """
    if os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") == "spawn":
        return

    reasons = []
    if is_in_ray_actor():
        # even if we choose to spawn, we need to pass the ray address
        # to the subprocess so that it knows how to connect to the ray cluster.
        # env vars are inherited by subprocesses, even if we use spawn.
        import ray

        os.environ["RAY_ADDRESS"] = ray.get_runtime_context().gcs_address
        reasons.append("In a Ray actor and can only be spawned")

    if cuda_is_initialized():
        reasons.append("CUDA is initialized")
    elif xpu_is_initialized():
        reasons.append("XPU is initialized")

    if reasons:
        logger.warning(
            "We must use the `spawn` multiprocessing start method. "
            "Overriding VLLM_WORKER_MULTIPROC_METHOD to 'spawn'. "
            "See https://docs.vllm.ai/en/latest/usage/"
            "troubleshooting.html#python-multiprocessing "
            "for more information. Reasons: %s",
            "; ".join(reasons),
        )
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def get_mp_context():
    """Get a multiprocessing context with a particular method (spawn or fork).
    By default we follow the value of the VLLM_WORKER_MULTIPROC_METHOD to
    determine the multiprocessing method (default is fork). However, under
    certain conditions, we may enforce spawn and override the value of
    VLLM_WORKER_MULTIPROC_METHOD.
    """
    _maybe_force_spawn()
    mp_method = envs.VLLM_WORKER_MULTIPROC_METHOD
    return multiprocessing.get_context(mp_method)


def set_process_title(
    name: str,
    suffix: str = "",
    prefix: str = envs.VLLM_PROCESS_NAME_PREFIX,
) -> None:
    """Set the current process title with optional suffix."""
    try:
        import setproctitle
    except ImportError:
        return

    if suffix:
        name = f"{name}_{suffix}"

    setproctitle.setproctitle(f"{prefix}::{name}")


def _add_prefix(file: TextIO, worker_name: str, pid: int) -> None:
    """Add colored prefix to file output for log decoration."""
    if envs.NO_COLOR:
        prefix = f"({worker_name} pid={pid}) "
    else:
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
    # Respect VLLM_CONFIGURE_LOGGING environment variable
    if not envs.VLLM_CONFIGURE_LOGGING:
        return

    if process_name is None:
        process_name = get_mp_context().current_process().name

    pid = os.getpid()
    _add_prefix(sys.stdout, process_name, pid)
    _add_prefix(sys.stderr, process_name, pid)


def kill_process_tree(pid: int):
    """
    Kills all descendant processes of the given pid by sending SIGKILL.

    Args:
        pid (int): Process ID of the parent process
    """
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    # Get all children recursively
    children = parent.children(recursive=True)

    # Send SIGKILL to all children first
    for child in children:
        with contextlib.suppress(ProcessLookupError):
            os.kill(child.pid, signal.SIGKILL)

    # Finally kill the parent
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGKILL)


# Resource utilities


# Adapted from: https://github.com/sgl-project/sglang/blob/v0.4.1/python/sglang/srt/utils.py#L630
def set_ulimit(target_soft_limit: int = 65535):
    if sys.platform.startswith("win"):
        logger.info("Windows detected, skipping ulimit adjustment.")
        return

    import resource

    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)

    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type, (target_soft_limit, current_hard))
        except ValueError as e:
            logger.warning(
                "Found ulimit of %s and failed to automatically increase "
                "with error %s. This can cause fd limit errors like "
                "`OSError: [Errno 24] Too many open files`. Consider "
                "increasing with ulimit -n",
                current_soft,
                e,
            )


def find_loaded_library(lib_name: str) -> str | None:
    """
    According to according to https://man7.org/linux/man-pages/man5/proc_pid_maps.5.html,
    the file `/proc/self/maps` contains the memory maps of the process, which includes the
    shared libraries loaded by the process. We can use this file to find the path of the
    a loaded library.
    """  # noqa
    found_line = None
    with open("/proc/self/maps") as f:
        for line in f:
            if lib_name in line:
                found_line = line
                break
    if found_line is None:
        # the library is not loaded in the current process
        return None
    # if lib_name is libcudart, we need to match a line with:
    # address /path/to/libcudart-hash.so.11.0
    start = found_line.index("/")
    path = found_line[start:].strip()
    filename = path.split("/")[-1]
    assert filename.rpartition(".so")[0].startswith(lib_name), (
        f"Unexpected filename: {filename} for library {lib_name}"
    )
    return path
