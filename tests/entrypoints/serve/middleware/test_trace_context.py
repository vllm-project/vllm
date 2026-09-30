# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import logging

import pytest
from opentelemetry import trace

from vllm.entrypoints.serve.middleware.trace_context import TraceContextMiddleware
from vllm.logging_utils.trace_context import TraceContextFilter

TRACE_ID = "11111111111111111111111111111111"
SPAN_ID = "2222222222222222"
TRACEPARENT = f"00-{TRACE_ID}-{SPAN_ID}-01"


@pytest.mark.parametrize("header", [TRACEPARENT, "invalid", None])
@pytest.mark.parametrize("fail", [False, True])
def test_request_logs_bind_context_until_stream_finishes(header, fail):
    """Header context covers response sends and is reset even on failure."""
    records = []
    context_filter = TraceContextFilter()

    async def send(message):
        record = logging.makeLogRecord({"msg": "response chunk"})
        context_filter.filter(record)
        records.append(record)

    async def receive():
        return {"type": "http.request", "body": b""}

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await asyncio.sleep(0)
        await send({"type": "http.response.body", "body": b"chunk"})
        if fail:
            raise RuntimeError("request failed")

    async def run():
        original = trace.get_current_span()
        scope = {
            "type": "http",
            "headers": [(b"traceparent", header.encode())] if header else [],
        }
        if fail:
            with pytest.raises(RuntimeError, match="request failed"):
                await TraceContextMiddleware(app)(scope, receive, send)
        else:
            await TraceContextMiddleware(app)(scope, receive, send)
        assert trace.get_current_span() is original

    asyncio.run(run())
    assert len(records) == 2
    for record in records:
        if header == TRACEPARENT:
            assert record.trace_id == TRACE_ID
            assert record.span_id == SPAN_ID
        else:
            assert record.trace_context == ""


def test_overlapping_requests_keep_separate_contexts():
    """An await must not expose another request's incoming span."""

    async def app(scope, receive, send):
        await asyncio.sleep(0)
        assert trace.get_current_span().get_span_context().trace_id == scope["id"]

    async def unused(*args):
        pass

    async def run():
        middleware = TraceContextMiddleware(app)
        await asyncio.gather(
            *(
                middleware(
                    {
                        "type": "http",
                        "id": i,
                        "headers": [
                            (b"traceparent", f"00-{i:032x}-{SPAN_ID}-01".encode())
                        ],
                    },
                    unused,
                    unused,
                )
                for i in (1, 2)
            )
        )

    asyncio.run(run())


def test_existing_span_takes_precedence():
    """Keep a server span already established by external instrumentation."""
    span = trace.NonRecordingSpan(trace.SpanContext(1, 2, is_remote=False))

    async def app(scope, receive, send):
        assert trace.get_current_span() is span

    async def unused(*args):
        pass

    with trace.use_span(span):
        asyncio.run(
            TraceContextMiddleware(app)(
                {"type": "http", "headers": [(b"traceparent", TRACEPARENT.encode())]},
                unused,
                unused,
            )
        )
