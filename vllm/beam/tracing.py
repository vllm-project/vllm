import logging
from functools import wraps
from typing import Optional, Union, AsyncGenerator
import time

from vllm.entrypoints.openai.protocol import CompletionRequest, ErrorResponse
from vllm.tracing import extract_trace_context, SpanAttributes, init_tracer
from vllm.v1.request import Request
from opentelemetry import trace

try:
    tracer = init_tracer(
    "vllm.entrypoints.openai.serving_completion",
    "http://localhost:4317")
except Exception as e:
    tracer = None
    logging.getLogger(__name__).warning(f"Failed to initialize tracer: {e}. OpenTelemetry tracing will not be available.")

def trace_streaming_completion(tracer_attr='tracer'):
    """
    Decorator specifically for tracing streaming completion functions.
    Handles both the initial processing and the async generator.
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(self, request: CompletionRequest, raw_request: Request | None = None):
            ctx = extract_trace_context(dict(raw_request.headers)) if raw_request else None

            with tracer.start_span("chunkwise_beam_completion", context=ctx) as parent_span:
                try:
                    parent_span.set_attribute(SpanAttributes.GEN_AI_REQUEST_MAX_TOKENS, request.max_tokens)
                    parent_span.set_attribute(SpanAttributes.GEN_AI_REQUEST_N, request.n)
                    if hasattr(request, "request_id"):
                        parent_span.set_attribute(SpanAttributes.GEN_AI_REQUEST_ID, request.request_id)

                    gen = await func(self, request, raw_request)

                    # If it's an error response, return it immediately
                    if isinstance(gen, ErrorResponse):
                        return gen

                    async def traced_generator():
                        """Wrapped generator that handles errors properly"""
                        try:
                            with trace.use_span(parent_span, end_on_exit=False):
                                with tracer.start_as_current_span("chunk_generation") as chunk_span:
                                    chunk_count = 0
                                    async for item in gen:
                                        chunk_count += 1
                                        yield item
                                    chunk_span.set_attribute("chunks_generated", chunk_count)
                        except Exception as e:
                            # Record the exception in the parent span
                            parent_span.record_exception(e)
                            parent_span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                            raise
                        finally:
                            # Ensure any cleanup in the original generator happens
                            if hasattr(gen, 'aclose'):
                                await gen.aclose()

                    return traced_generator()

                except Exception as e:
                    parent_span.record_exception(e)
                    parent_span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    raise

        return wrapper

    return decorator


def trace_async_method(span_name: Optional[str] = None, tracer_attr='tracer'):
    """
    Simple decorator for tracing regular async methods.
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            name = span_name or func.__name__

            with tracer.start_as_current_span(name) as span:
                start_time = time.time()
                try:
                    result = await func(self, *args, **kwargs)
                    span.set_attribute("execution_time_ms", (time.time() - start_time) * 1000)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    raise

        return wrapper

    return decorator