# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lightweight profiler for timing code execution with minimal overhead."""

from vllm.utils.lite_profiler.lite_profiler import (
    LiteProfilerScope,
    maybe_emit_lite_profiler_report,
)

__all__ = [
    "LiteProfilerScope",
    "maybe_emit_lite_profiler_report",
]
