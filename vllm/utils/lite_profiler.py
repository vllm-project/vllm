# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal helpers for opt-in lightweight timing collection."""
from __future__ import annotations

import atexit
import multiprocessing
import os
import time
from contextlib import suppress
from types import TracebackType
from typing import TextIO

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)


def _should_log_results() -> int | None:
    """Check if the current process should log results.
    For multiprocessing, only the main process and the first worker 
    should log results, as other worker processes duplicate the same
    work.
    """
    process = multiprocessing.current_process()
    return process.name in ("EngineCore_DP0", "VllmWorker-0")


# Cache for log file handles
_log_file_cache: dict[str, TextIO] = {}


def _write_log_entry(name: str, elapsed_us: int) -> None:
    """Write a profiler entry using cached file handles for optimal performance.

    This function implements an efficient caching approach where file handles
    are opened once per log path and reused for all subsequent writes. This
    eliminates the significant overhead of opening/closing files for every
    profiler entry, which is crucial for maintaining the lightweight nature
    of the profiler.

    The cached file handles are automatically closed on program exit via atexit.
    """
    global _log_file_cache
    _LOG_PATH = envs.VLLM_LITE_PROFILER_LOG_PATH

    if not _should_log_results() or _LOG_PATH is None:
        return

    # Handle case where file handle was opened in parent but we're in child
    # The file descriptor may be invalid after fork
    if _log_file_cache.get(_LOG_PATH) is not None:
        try:
            # Test if the file handle is still valid
            _log_file_cache[_LOG_PATH].tell()
        except (OSError, ValueError):
            # File handle is stale, clear and reopen
            _log_file_cache.clear()

    log_line = f"{name}|{elapsed_us}\n"
    log_file = _log_file_cache.get(_LOG_PATH)
    if log_file is None:
        directory = os.path.dirname(_LOG_PATH)
        if directory:
            os.makedirs(directory, exist_ok=True)
        # ruff: noqa: SIM115 - intentionally keeping file handle cached globally
        log_file = open(_LOG_PATH, "a", buffering=50000)
        _log_file_cache[_LOG_PATH] = log_file
        atexit.register(log_file.close)

    log_file.write(log_line)


class LiteScope:
    """Lightweight context manager for timing code blocks with minimal overhead.

    This class provides a simple way to measure and log the execution time of
    code blocks using Python's context manager protocol (with statement). It's
    designed for high-frequency profiling with minimal performance impact.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._start_time: int | None = None

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
    """Generate and display a summary report of profiling data if available.

    This function serves as the main entry point for analyzing and displaying
    profiling results. It checks if profiling was enabled and a log file exists,
    then delegates to the lite_profiler_report module to generate statistics
    like function call counts, timing distributions, and performance insights.
    """

    log_path = envs.VLLM_LITE_PROFILER_LOG_PATH
    if log_path is None:
        return

    if not os.path.exists(log_path):
        logger.warning(
            "Lite profiler log not found. Ensure the profiled process sets "
            "the expected path.")
        return

    try:
        from vllm.utils import lite_profiler_report
    except Exception as exc:  # pragma: no cover - import error should not crash
        logger.error("Failed to import lite profiler report helper: %s", exc)
        return

    logger.info("")
    logger.info("Lite profiler summary (%s):", log_path)
    try:
        # Generate and display the summary report
        lite_profiler_report.summarize_log(log_path)

        # Clear the log file to avoid accumulating data from multiple runs
        with suppress(OSError):
            directory = os.path.dirname(log_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(log_path, "w"):
                pass
    except Exception as exc:  # pragma: no cover - avoid crashing benchmarks
        logger.error("Failed to summarize lite profiler log %s: %s", log_path,
                     exc)
