# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal helpers for opt-in lightweight timing collection."""
from __future__ import annotations

import atexit
import json
import logging
import os
import sys
import time
from logging.handlers import MemoryHandler
from types import TracebackType
from typing import Optional

import vllm.envs as envs
from vllm.logger import init_logger


class LiteScope:
    """Lite Scope that directly logs function duration"""

    def __init__(self, profiler: LiteProfiler, name: str) -> None:
        self._profiler = profiler
        self._name = name
        self._start_time: Optional[float] = None

    def __enter__(self) -> None:
        self._start_time = time.perf_counter()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if self._start_time is not None and exc_type is None:
            elapsed_seconds = time.perf_counter() - self._start_time
            elapsed_ns = int(elapsed_seconds * 1e9)
            self._profiler._log_function_duration(self._name, elapsed_ns)
        return False


class LiteProfiler:
    """Lite profiler that logs function durations directly."""

    def __init__(self) -> None:
        self._logger = init_logger("vllm.lite_profiler")
        log_path = envs.VLLM_LITE_PROFILER_LOG_PATH
        if log_path:
            file_handler = logging.FileHandler(log_path, mode="w")
            file_handler.setFormatter(logging.Formatter("%(message)s"))
            self._handler = MemoryHandler(capacity=2048,
                                          flushLevel=logging.ERROR,
                                          target=file_handler)
            self._logger.addHandler(self._handler)
            self._logger.propagate = False
            atexit.register(self._handler.flush)

    def scope(self, name: str):
        return LiteScope(self, name)

    def _log_function_duration(self, name: str, elapsed_ns: int) -> None:
        """Log function duration in the json format"""
        metrics = {name: {"ns": elapsed_ns}}
        payload = {
            "ts": time.time(),
            "metrics": metrics,
        }

        message = json.dumps(payload, separators=(",", ":"))
        self._logger.info(
            message)  # Will be buffered and flushed by MemoryHandler

    def flush(self) -> None:
        """Explicitly flush the buffer to the file."""
        if hasattr(self, "_handler"):
            self._handler.flush()


# Global variable to store the single LiteProfiler instance
_lite_profiler: Optional[LiteProfiler] = None


def scope_function(name: str):
    """Create a scope context manager for timing a function.

    Initializes LiteProfiler only once and reuses the same instance
    for all subsequent calls
    """
    global _lite_profiler

    if _lite_profiler is None:
        _lite_profiler = LiteProfiler()

    return _lite_profiler.scope(name)


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
