# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal helpers for opt-in lightweight timing collection."""

from __future__ import annotations

import atexit
import multiprocessing
import os
import time
from types import TracebackType
from typing import TextIO

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)


def _should_log_results() -> bool:
    """Check if the current process should log results.
    Only the data-parallel rank 0 engine core and worker 0 should emit logs in
    multi-process deployments so that we avoid duplicating identical timing
    data.
    """
    process = multiprocessing.current_process()
    return process.name in ("EngineCore_DP0", "VllmWorker-0")


# Cache for log file handle
_log_file: TextIO | None = None


def _write_log_entry(name: str, elapsed_ns: int) -> None:
    """Write a profiler entry using cached file handle for optimal performance.

    This function implements an efficient caching approach where the file handle
    is opened once and reused for all subsequent writes. This eliminates the
    significant overhead of opening/closing files for every profiler entry,
    which is crucial for maintaining the lightweight nature of the profiler.

    The cached file handle is automatically closed on program exit via atexit.
    """
    global _log_file
    _LOG_PATH = envs.VLLM_LITE_PROFILER_LOG_PATH
    assert _LOG_PATH is not None

    if not _should_log_results():
        return

    # Handle case where file handle was opened in parent but we're in the
    # child process. The file descriptor may become invalid after fork
    if _log_file is not None:
        try:
            # Verify if the file handle is still valid
            _log_file.flush()
            _log_file.tell()
        except (OSError, ValueError):
            # File handle is stale, clear and reopen
            _log_file = None

    # Write the log entry
    log_line = f"{name}|{elapsed_ns}\n"
    if _log_file is None:
        directory = os.path.dirname(_LOG_PATH)
        if directory:
            os.makedirs(directory, exist_ok=True)
        # ruff: noqa: SIM115 - intentionally keeping file handle cached globally
        _log_file = open(_LOG_PATH, "a", buffering=1)
        atexit.register(_log_file.close)

    _log_file.write(log_line)


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
    ) -> None:
        if self._start_time is not None and exc_type is None:
            elapsed_ns = time.perf_counter_ns() - self._start_time
            _write_log_entry(self._name, elapsed_ns)


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

    # Ensure the log file is flushed and closed before generating report
    global _log_file
    if _log_file is not None:
        _log_file.close()
        _log_file = None

    if not os.path.exists(log_path):
        logger.warning(
            "Lite profiler log not found. Ensure the profiled process sets "
            "the expected path."
        )
        return

    from vllm.utils.lite_profiler import lite_profiler_report

    logger.info("")
    logger.info("Lite profiler summary (%s):", log_path)
    try:
        # Generate and display the summary report
        lite_profiler_report.summarize_log(log_path)
        os.remove(log_path)
    except Exception as exc:  # pragma: no cover - avoid crashing benchmarks
        logger.error("Failed to summarize lite profiler log %s: %s", log_path, exc)
