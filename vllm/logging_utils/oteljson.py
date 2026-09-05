# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OpenTelemetry Logs Data Model JSON formatter."""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

_SEVERITY = {
    logging.DEBUG: ("DEBUG", 5),
    logging.INFO: ("INFO", 9),
    logging.WARNING: ("WARN", 13),
    logging.ERROR: ("ERROR", 17),
    logging.CRITICAL: ("FATAL", 21),
}


def _severity(levelno: int) -> tuple[str, int]:
    if levelno >= logging.CRITICAL:
        return _SEVERITY[logging.CRITICAL]
    if levelno >= logging.ERROR:
        return _SEVERITY[logging.ERROR]
    if levelno >= logging.WARNING:
        return _SEVERITY[logging.WARNING]
    if levelno >= logging.INFO:
        return _SEVERITY[logging.INFO]
    return _SEVERITY[logging.DEBUG]


def _trace_fields() -> dict[str, str]:
    try:
        from opentelemetry import trace
    except ImportError:
        return {}
    ctx = trace.get_current_span().get_span_context()
    if not ctx.is_valid:
        return {}
    return {
        "trace_id": format(ctx.trace_id, "032x"),
        "span_id": format(ctx.span_id, "016x"),
    }


class OTelJSONFormatter(logging.Formatter):
    """Emit one JSON object per log record using OTel field names."""

    def format(self, record: logging.LogRecord) -> str:
        severity_text, severity_number = _severity(record.levelno)
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "severity_text": severity_text,
            "severity_number": severity_number,
            "body": record.getMessage(),
            "logger": record.name,
            "service.name": os.getenv("OTEL_SERVICE_NAME", "vllm"),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        payload.update(_trace_fields())
        return json.dumps(payload, default=str)
