from functools import wraps
from typing import Optional, Union, AsyncGenerator
import time

from vllm.entrypoints.openai.protocol import CompletionRequest, ErrorResponse
from vllm.tracing import extract_trace_context, SpanAttributes, init_tracer
from vllm.v1.request import Request
from opentelemetry import trace

tracer = init_tracer(
                "vllm.entrypoints.openai.serving_completion",
                "http://localhost:4317")

def trace_streaming_completion(tracer_attr='tracer'):
    """
    Decorator specifically for tracing streaming completion functions.
    Handles both the initial processing and the async generator.
    """

    def decorator(func):
        async def wrapper(self, request: CompletionRequest, raw_request: Request | None = None):
            ctx = extract_trace_context(dict(raw_request.headers)) if raw_request else None
            parent_span = tracer.start_span("chunkwise_beam_completion", context=ctx)

            # keep the span current until we’re done
            scope = trace.use_span(parent_span, end_on_exit=False)

            try:
                parent_span.set_attribute(SpanAttributes.GEN_AI_REQUEST_MAX_TOKENS, request.max_tokens)
                parent_span.set_attribute(SpanAttributes.GEN_AI_REQUEST_N, request.n)
                if hasattr(request, "request_id"):
                    parent_span.set_attribute(SpanAttributes.GEN_AI_REQUEST_ID, request.request_id)

                gen = await func(self, request, raw_request)
                if isinstance(gen, ErrorResponse):
                    parent_span.end()
                    scope.__exit__(None, None, None)
                    return gen

                async def traced_generator():
                    with trace.use_span(parent_span, end_on_exit=False):
                        with tracer.start_as_current_span("chunk_generation"):
                            async for item in gen:
                                yield item


                    # now it’s safe to close the parent
                    parent_span.end()
                    scope.__exit__(None, None, None)

                return traced_generator()

            except Exception as e:
                parent_span.record_exception(e)
                parent_span.end()
                scope.__exit__(type(e), e, e.__traceback__)
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
                    raise

        return wrapper

    return decorator