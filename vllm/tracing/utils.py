# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping

from vllm.logger import init_logger
from vllm.utils.func_utils import run_once

logger = init_logger(__name__)

try:
    from opentelemetry import trace

    _IS_OTEL_AVAILABLE = True
except ImportError:
    trace = None  # type: ignore
    _IS_OTEL_AVAILABLE = False

# Standard W3C headers used for context propagation
TRACE_HEADERS = ["traceparent", "tracestate"]


class SpanAttributes:
    """
    Standard attributes for spans.

    These are largely based on OpenTelemetry Semantic Conventions but are defined
    here as constants so they can be used by any backend or logger.
    """

    # Attribute names copied from OTel semantic conventions to avoid version conflicts
    GEN_AI_USAGE_COMPLETION_TOKENS = "gen_ai.usage.completion_tokens"
    GEN_AI_USAGE_PROMPT_TOKENS = "gen_ai.usage.prompt_tokens"
    GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
    GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"

    # Custom attributes added until they are standardized
    GEN_AI_REQUEST_ID = "gen_ai.request.id"
    GEN_AI_REQUEST_N = "gen_ai.request.n"
    GEN_AI_USAGE_NUM_SEQUENCES = "gen_ai.usage.num_sequences"
    GEN_AI_LATENCY_TIME_IN_QUEUE = "gen_ai.latency.time_in_queue"
    GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN = "gen_ai.latency.time_to_first_token"
    GEN_AI_LATENCY_E2E = "gen_ai.latency.e2e"
    GEN_AI_LATENCY_TIME_IN_SCHEDULER = "gen_ai.latency.time_in_scheduler"

    # Latency breakdowns
    GEN_AI_LATENCY_TIME_IN_MODEL_FORWARD = "gen_ai.latency.time_in_model_forward"
    GEN_AI_LATENCY_TIME_IN_MODEL_EXECUTE = "gen_ai.latency.time_in_model_execute"
    GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL = "gen_ai.latency.time_in_model_prefill"
    GEN_AI_LATENCY_TIME_IN_MODEL_DECODE = "gen_ai.latency.time_in_model_decode"
    GEN_AI_LATENCY_TIME_IN_MODEL_INFERENCE = "gen_ai.latency.time_in_model_inference"


class LoadingSpanAttributes:
    """Custom attributes for code-level tracing (file, line number)."""

    CODE_NAMESPACE = "code.namespace"
    CODE_FUNCTION = "code.function"
    CODE_FILEPATH = "code.filepath"
    CODE_LINENO = "code.lineno"


def contains_trace_headers(headers: Mapping[str, str]) -> bool:
    """Check if the provided headers dictionary contains trace context."""
    return any(h in headers for h in TRACE_HEADERS)


def extract_trace_headers(headers: Mapping[str, str]) -> Mapping[str, str]:
    """
    Build trace headers for downstream propagation.

    Prefer the active in-process span so downstream spans become children of
    the current ASGI/FastAPI span. Fall back to any inbound trace headers when
    there is no active span.
    """
    if _IS_OTEL_AVAILABLE:
        current_span = trace.get_current_span()
        span_context = current_span.get_span_context()
        if span_context.is_valid:
            trace_headers = {
                "traceparent": (
                    f"00-{span_context.trace_id:032x}-"
                    f"{span_context.span_id:016x}-"
                    f"{int(span_context.trace_flags):02x}"
                ),
            }
            if "tracestate" in headers:
                trace_headers["tracestate"] = headers["tracestate"]
            return trace_headers
    return {h: headers[h] for h in TRACE_HEADERS if h in headers}


@run_once
def log_tracing_disabled_warning() -> None:
    logger.warning("Received a request with trace context but tracing is disabled")
