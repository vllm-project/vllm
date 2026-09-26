# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging


class TraceContextFilter(logging.Filter):
    """Attach the emitting thread/task's OpenTelemetry context to a record."""

    def __init__(self) -> None:
        super().__init__()
        try:
            from opentelemetry.trace import get_current_span
        except ImportError as exc:
            raise ImportError(
                "TraceContextFilter requires opentelemetry-api. "
                "Install it before enabling VLLM_LOGGING_TRACE_CONTEXT."
            ) from exc
        self._get_current_span = get_current_span

    def filter(self, record: logging.LogRecord) -> bool:
        context = self._get_current_span().get_span_context()
        record.trace_id = format(context.trace_id if context.is_valid else 0, "032x")
        record.span_id = format(context.span_id if context.is_valid else 0, "016x")
        record.trace_sampled = bool(context.is_valid and context.trace_flags.sampled)
        record.trace_context = (
            f"[trace_id={record.trace_id} span_id={record.span_id}] "
            if context.is_valid
            else ""
        )
        return True
