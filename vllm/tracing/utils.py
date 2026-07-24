# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Mapping

from vllm.logger import init_logger
from vllm.utils.func_utils import run_once

logger = init_logger(__name__)

# Standard W3C headers used for context propagation
TRACE_HEADERS = ["traceparent", "tracestate"]

# OTel semantic-convention stability opt-in (comma-separated env var). When it
# contains this token, instrumentations emit the current (experimental) GenAI
# conventions instead of / in addition to the legacy ones.
# See: https://opentelemetry.io/docs/specs/semconv/gen-ai/
OTEL_SEMCONV_STABILITY_OPT_IN = "OTEL_SEMCONV_STABILITY_OPT_IN"
GEN_AI_LATEST_EXPERIMENTAL = "gen_ai_latest_experimental"


def is_gen_ai_latest_semconv_enabled() -> bool:
    """Whether to emit the current (experimental) OTel GenAI semconv attributes.

    Gated by the standard OTel opt-in env var ``OTEL_SEMCONV_STABILITY_OPT_IN``
    (a comma-separated list). When it contains ``gen_ai_latest_experimental``,
    vLLM additionally emits the current spec attribute names and span name;
    otherwise only the legacy attributes are emitted, so default behavior is
    unchanged.
    """
    opt_in = os.environ.get(OTEL_SEMCONV_STABILITY_OPT_IN, "")
    return GEN_AI_LATEST_EXPERIMENTAL in (token.strip() for token in opt_in.split(","))


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

    # Current OTel GenAI semconv identity attributes. Emitted only when
    # OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental is set.
    # `operation.name` is Required by the spec; `request.model` conditionally
    # required. (`response.model` above remains defined but is never emitted.)
    GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
    GEN_AI_REQUEST_MODEL = "gen_ai.request.model"

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


def latest_gen_ai_semconv_attributes(
    operation_name: str | None, model_name: str | None
) -> tuple[dict[str, str], str | None]:
    """Current OTel GenAI semconv identity attributes + span name.

    Pure helper for the request span. Returns ``(attributes, span_name)`` where
    ``span_name`` is the spec-recommended ``"{operation} {model}"`` or ``None``
    when it can't be formed (operation or model unknown) and the caller should
    keep its default span name.

    Args:
        operation_name: `gen_ai.operation.name` (e.g. "chat"), or None.
        model_name: `gen_ai.request.model` (served model name), or None.
    """
    attributes: dict[str, str] = {}
    if model_name is not None:
        attributes[SpanAttributes.GEN_AI_REQUEST_MODEL] = model_name
    if operation_name is not None:
        attributes[SpanAttributes.GEN_AI_OPERATION_NAME] = operation_name
    span_name = None
    if operation_name is not None and model_name is not None:
        span_name = f"{operation_name} {model_name}"
    return attributes, span_name


def contains_trace_headers(headers: Mapping[str, str]) -> bool:
    """Check if the provided headers dictionary contains trace context."""
    return any(h in headers for h in TRACE_HEADERS)


def extract_trace_headers(headers: Mapping[str, str]) -> Mapping[str, str]:
    """
    Extract only trace-related headers from a larger header dictionary.
    Useful for logging or passing context to a non-OTel client.
    """
    return {h: headers[h] for h in TRACE_HEADERS if h in headers}


@run_once
def log_tracing_disabled_warning() -> None:
    logger.warning("Received a request with trace context but tracing is disabled")
