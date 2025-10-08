# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal helpers for opt-in lightweight timing collection."""
from __future__ import annotations

import atexit
import logging
import os
import sys
import time
from logging.handlers import MemoryHandler
from types import TracebackType
from typing import Optional

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

# Setup buffered file logging using MemoryHandler
_log_path = envs.VLLM_LITE_PROFILER_LOG_PATH
if _log_path:
    # Create file handler for the profiler output
    file_handler = logging.FileHandler(_log_path, mode="w")
    file_handler.setFormatter(logging.Formatter("%(message)s"))

    # Create memory handler that buffers logs and flushes periodically
    memory_handler = MemoryHandler(
        capacity=1000,  # Buffer up to 1000 log entries
        flushLevel=logging.CRITICAL,  # Don't auto-flush unless critical
        target=file_handler,
    )

    logger.addHandler(memory_handler)
    logger.propagate = False

    # Register cleanup to flush buffer on exit
    atexit.register(memory_handler.flush)


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
            # Simple format: "<name>|<elapsed_time_us>"
            logger.info("%s|%s", self._name, elapsed_us)
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
