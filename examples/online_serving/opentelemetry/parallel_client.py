# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio

import aiohttp
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import SpanKind, Tracer, set_tracer_provider
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator


async def main():
    """Send multiple concurrent requests to test batched tracing"""

    prompts = [
        "What is the capital of France?",
        "What is the largest city in the world?",
        "Why does the sky look blue?",
    ]

    tracer = create_tracer()
    with tracer.start_as_current_span("parallel-client", kind=SpanKind.CLIENT):
        async with aiohttp.ClientSession() as session:
            # Send all requests concurrently
            tasks = [
                send_request(session, tracer, i, prompt)
                for i, prompt in enumerate(prompts)
            ]
            results = await asyncio.gather(*tasks)

    for r in results:
        print(r, end="\n\n")


def create_tracer() -> Tracer:
    provider = TracerProvider()
    set_tracer_provider(provider)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    tracer = provider.get_tracer("dummy-client")
    return tracer


async def send_request(
    session,
    tracer: Tracer,
    request_num: int,
    prompt: str,
) -> dict:
    """Send a single traced request"""
    url = "http://localhost:8000/v1/completions"

    with tracer.start_as_current_span(
        f"req-num-{request_num}", kind=SpanKind.CLIENT
    ) as span:
        # Example of setting span attributes
        span.set_attribute("prompt", prompt)

        headers = {"Content-Type": "application/json"}
        TraceContextTextMapPropagator().inject(headers)

        payload = {
            "model": "facebook/opt-125m",
            "max_tokens": 10,
            "prompt": prompt,
        }

        async with session.post(url, headers=headers, json=payload) as response:
            response.raise_for_status()
            return await response.json()


if __name__ == "__main__":
    asyncio.run(main())
