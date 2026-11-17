# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers for tests that exercise parser metrics."""

from vllm.parser import metrics
from vllm.v1.metrics.prometheus import unregister_vllm_metrics


def reset_parser_metrics() -> None:
    """Detach the registered parser metrics and reset the module state."""
    unregister_vllm_metrics()
    metrics._metrics.clear()
    metrics._tool_call_children.clear()
    metrics._collectors.clear()
    metrics._enabled = False
    metrics._slow_log_enabled = False
    metrics._slow_call_seconds = 0.0
    metrics._slow_request_seconds = 0.0
    metrics._log_request_ids = False
