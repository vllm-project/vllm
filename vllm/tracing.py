# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Iterator, Mapping
from typing import Optional, Union

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils import run_once
# Moving `TraceHeaders` to tracing.py leads to a circular import
# because tracing.py imports SchedulerOutput
from vllm.v1.core.sched.output import (NewRequestData, SchedulerOutput,
                                       TraceHeaders)
from vllm.v1.utils import join_context_managers
from vllm.v1.worker.gpu_input_batch import CachedRequestState

TRACE_HEADERS = ["traceparent", "tracestate"]

logger = init_logger(__name__)

_is_otel_imported = False
otel_import_error_traceback: Optional[str] = None
try:
    from opentelemetry.context.context import Context
    from opentelemetry.sdk.environment_variables import (
        OTEL_EXPORTER_OTLP_TRACES_PROTOCOL)
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import Span, SpanKind, Tracer, set_tracer_provider
    from opentelemetry.trace.propagation.tracecontext import (
        TraceContextTextMapPropagator)
    _is_otel_imported = True
except ImportError:
    # Capture and format traceback to provide detailed context for the import
    # error. Only the string representation of the error is retained to avoid
    # memory leaks.
    # See https://github.com/vllm-project/vllm/pull/7266#discussion_r1707395458
    import traceback
    otel_import_error_traceback = traceback.format_exc()

    class Context:  # type: ignore
        pass

    class BaseSpanAttributes:  # type: ignore
        pass

    class SpanKind:  # type: ignore
        pass

    class Tracer:  # type: ignore
        pass

    class Span:  # type: ignore
        pass


def is_otel_available() -> bool:
    return _is_otel_imported


def init_tracer(instrumenting_module_name: str,
                otlp_traces_endpoint: str) -> Optional[Tracer]:
    if not is_otel_available():
        raise ValueError(
            "OpenTelemetry is not available. Unable to initialize "
            "a tracer. Ensure OpenTelemetry packages are installed. "
            f"Original error:\n{otel_import_error_traceback}")
    trace_provider = TracerProvider()

    span_exporter = get_span_exporter(otlp_traces_endpoint)
    trace_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    set_tracer_provider(trace_provider)

    tracer = trace_provider.get_tracer(instrumenting_module_name)
    return tracer


def get_span_exporter(endpoint):
    protocol = os.environ.get(OTEL_EXPORTER_OTLP_TRACES_PROTOCOL, "grpc")
    if protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter)
    elif protocol == "http/protobuf":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter)  # type: ignore
    else:
        raise ValueError(
            f"Unsupported OTLP protocol '{protocol}' is configured")

    return OTLPSpanExporter(endpoint=endpoint)


def extract_trace_context(
        headers: Optional[TraceHeaders]) -> Optional[Context]:
    if is_otel_available():
        headers = headers or {}
        return TraceContextTextMapPropagator().extract(headers)
    else:
        return None


def extract_trace_headers(headers: TraceHeaders) -> TraceHeaders:

    return {h: headers[h] for h in TRACE_HEADERS if h in headers}


class SpanAttributes:
    # Attribute names copied from here to avoid version conflicts:
    # https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md
    GEN_AI_USAGE_COMPLETION_TOKENS = "gen_ai.usage.completion_tokens"
    GEN_AI_USAGE_PROMPT_TOKENS = "gen_ai.usage.prompt_tokens"
    GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
    GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
    GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
    GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
    # Attribute names added until they are added to the semantic conventions:
    GEN_AI_REQUEST_ID = "gen_ai.request.id"
    GEN_AI_REQUEST_N = "gen_ai.request.n"
    GEN_AI_USAGE_NUM_SEQUENCES = "gen_ai.usage.num_sequences"
    GEN_AI_LATENCY_TIME_IN_QUEUE = "gen_ai.latency.time_in_queue"
    GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN = "gen_ai.latency.time_to_first_token"
    GEN_AI_LATENCY_E2E = "gen_ai.latency.e2e"
    GEN_AI_LATENCY_TIME_IN_SCHEDULER = "gen_ai.latency.time_in_scheduler"
    # Time taken in the forward pass for this across all workers
    GEN_AI_LATENCY_TIME_IN_MODEL_FORWARD = (
        "gen_ai.latency.time_in_model_forward")
    # Time taken in the model execute function. This will include model
    # forward, block/sync across workers, cpu-gpu sync time and sampling time.
    GEN_AI_LATENCY_TIME_IN_MODEL_EXECUTE = (
        "gen_ai.latency.time_in_model_execute")
    GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL = \
        "gen_ai.latency.time_in_model_prefill"
    GEN_AI_LATENCY_TIME_IN_MODEL_DECODE = "gen_ai.latency.time_in_model_decode"
    GEN_AI_LATENCY_TIME_IN_MODEL_INFERENCE = \
        "gen_ai.latency.time_in_model_inference"


def contains_trace_headers(headers: TraceHeaders) -> bool:
    return any(h in headers for h in TRACE_HEADERS)


@run_once
def log_tracing_disabled_warning() -> None:
    logger.warning(
        "Received a request with trace context but tracing is disabled")


def maybe_create_tracer(vllm_config: VllmConfig,
                        instrumenting_module_name: str) -> Optional[Tracer]:
    if vllm_config.observability_config is None:
        return None
    endpoint = vllm_config.observability_config.otlp_traces_endpoint
    if endpoint is None:
        return None
    return init_tracer(instrumenting_module_name, endpoint)


def trace_in_every_processed_request(tracer: Tracer, span_name: str,
                                     requests: Mapping[str,
                                                       CachedRequestState],
                                     scheduler_output: SchedulerOutput):
    """
    The processed requests are either new scheduled requests or cached requests.
    Starts an opentelemetry span for every request processed in this step.
    Returns a context manager.
    """
    new_reqs = scheduler_output.scheduled_new_reqs
    cached_reqs = [
        requests[id] for id in scheduler_output.scheduled_cached_reqs.req_ids
    ]
    all_reqs = cached_reqs + new_reqs

    span_context_managers = [
        new_span_ctx_manager(tracer, req, span_name) for req in all_reqs
    ]
    single_context_manager = join_context_managers(
        span_context_managers)  # type: ignore
    return single_context_manager


def new_span_ctx_manager(tracer: Tracer, req_data: Union[NewRequestData,
                                                         CachedRequestState],
                         name: str) -> Iterator[Span]:
    """
    Creates an OpenTelemetry span which can be used as a
    context manager. E.g.

    ```python
    with new_span_ctx_manager(tracer, req_data, "Some Work") as span:
        # do work
    ```
    """
    assert isinstance(req_data, (NewRequestData, CachedRequestState))
    context: Context = extract_trace_context(req_data.trace_headers)
    span_context_manager = tracer.start_as_current_span(
        name=name,
        kind=SpanKind.INTERNAL,
        context=context,
    )
    return span_context_manager
