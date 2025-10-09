# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal helpers for opt-in lightweight timing collection."""
from __future__ import annotations

import atexit
import os
import sys
import threading
import time
from types import TracebackType
from typing import Optional, TextIO

import vllm.envs as envs

_LOG_PATH = envs.VLLM_LITE_PROFILER_LOG_PATH
_THREAD_LOCK = threading.Lock()


def _get_process_rank() -> Optional[int]:
    for env_name in ("VLLM_DP_RANK", "RANK", "LOCAL_RANK"):
        value = os.environ.get(env_name)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                return None
    return None


_SHOULD_LOG = _LOG_PATH is not None and ((rank := _get_process_rank()) is None
                                         or rank == 0)

# Cache for log file handles
_log_file_cache: dict[str, TextIO] = {}


def _write_log_entry(name: str, elapsed_us: int) -> None:
    if not _SHOULD_LOG or _LOG_PATH is None:
        return

    log_line = f"{name}|{elapsed_us}\n"
    with _THREAD_LOCK:
        log_file = _log_file_cache.get(_LOG_PATH)
        if log_file is None:
            directory = os.path.dirname(_LOG_PATH)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(_LOG_PATH, "a", buffering=50000) as log_file:
                _log_file_cache[_LOG_PATH] = log_file
                atexit.register(log_file.close)

        log_file.write(log_line)
        log_file.flush()


class LiteScope:
    """Lite Scope that directly logs function duration"""

    def __init__(self, name: str) -> None:
        self._name = name
        self._start_time: Optional[int] = None

    def __enter__(self) -> None:
        self._start_time = time.perf_counter_ns()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if self._start_time is not None and exc_type is None:
            elapsed_ns = time.perf_counter_ns() - self._start_time
            # Use integer microseconds for better performance
            elapsed_us = elapsed_ns // 1000
            _write_log_entry(self._name, elapsed_us)
        return False


def maybe_emit_lite_profiler_report() -> None:
    """Print a lite-profiler summary when profiling is enabled."""

    log_path = envs.VLLM_LITE_PROFILER_LOG_PATH
    if log_path is None:
        return

    if not os.path.exists(log_path):
        print(
            "Lite profiler log not found. Ensure the profiled process sets "
            "the expected path.",
            file=sys.stderr,
        )
        return

    try:
        from vllm.utils import lite_profiler_report
    except Exception as exc:  # pragma: no cover - import error should not crash
        print(
            f"Failed to import lite profiler report helper: {exc}",
            file=sys.stderr,
        )
        return

    print(f"\nLite profiler summary ({log_path}):")
    try:
        lite_profiler_report.summarize_log(log_path, stream=sys.stdout)
    except Exception as exc:  # pragma: no cover - avoid crashing benchmarks
        print(
            f"Failed to summarize lite profiler log {log_path}: {exc}",
            file=sys.stderr,
        )
