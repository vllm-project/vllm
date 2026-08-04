# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio

import pytest
from starlette.responses import StreamingResponse

from vllm.entrypoints.openai.sse_keep_alive import (
    SSE_KEEP_ALIVE_COMMENT,
    with_sse_keep_alive,
)

# --- Disabled path ----------------------------------------------------------


def test_disabled_returns_same_generator():
    """For 0, negative, and NaN intervals the exact same generator object must
    be returned (identity), proving no wrapper or per-chunk overhead when
    disabled."""

    async def gen():
        yield "x"

    for disabled in (0.0, -5.0, float("nan"), float("inf")):
        g = gen()
        assert with_sse_keep_alive(g, disabled) is g


# --- Keep-alive emission ----------------------------------------------------


@pytest.mark.asyncio
async def test_keep_alive_during_silent_start():
    """A silent start (queue wait / prefill) emits keep-alive comments before
    the first real chunk."""
    release = asyncio.Event()

    async def gen():
        await release.wait()  # simulates queue wait / prefill
        yield "data: first\n\n"

    out: list[str] = []
    async for item in with_sse_keep_alive(gen(), 0.01):
        out.append(item)
        if item == SSE_KEEP_ALIVE_COMMENT:
            release.set()
        if item == "data: first\n\n":
            break

    assert SSE_KEEP_ALIVE_COMMENT in out
    assert out.index(SSE_KEEP_ALIVE_COMMENT) < out.index("data: first\n\n")


@pytest.mark.asyncio
async def test_no_keep_alive_when_streaming_is_fast():
    """If chunks arrive faster than the interval, no comment is emitted and all
    data is forwarded."""

    async def gen():
        for chunk in ["a", "b", "c"]:
            yield chunk

    out = [c async for c in with_sse_keep_alive(gen(), 0.5)]
    assert out == ["a", "b", "c"]
    assert SSE_KEEP_ALIVE_COMMENT not in out


@pytest.mark.asyncio
async def test_keep_alive_between_chunks_does_not_drop_data():
    """Keep-alives interleaved with real chunks must never drop or reorder
    data."""
    release_b = asyncio.Event()
    release_c = asyncio.Event()

    async def gen():
        yield "data: a\n\n"
        await release_b.wait()
        yield "data: b\n\n"
        await release_c.wait()
        yield "data: c\n\n"

    out: list[str] = []
    async for item in with_sse_keep_alive(gen(), 0.01):
        out.append(item)
        if item == SSE_KEEP_ALIVE_COMMENT and not release_b.is_set():
            release_b.set()
        elif item == "data: b\n\n" and not release_c.is_set():
            release_c.set()
        elif item == "data: c\n\n":
            break

    assert SSE_KEEP_ALIVE_COMMENT in out
    data_chunks = [c for c in out if c != SSE_KEEP_ALIVE_COMMENT]
    assert data_chunks == ["data: a\n\n", "data: b\n\n", "data: c\n\n"]


@pytest.mark.asyncio
async def test_empty_generator_returns_immediately():
    async def gen():
        if False:
            yield "never"

    out = [c async for c in with_sse_keep_alive(gen(), 0.01)]
    assert out == []


# --- Errors and cancellation ------------------------------------------------


@pytest.mark.asyncio
async def test_exception_propagates():
    async def gen():
        yield "data: before\n\n"
        raise RuntimeError("boom")

    wrapped = with_sse_keep_alive(gen(), 0.01)
    assert await anext(wrapped) == "data: before\n\n"
    with pytest.raises(RuntimeError, match="boom"):
        await anext(wrapped)


@pytest.mark.asyncio
async def test_upstream_cancelled_error_propagates():
    async def gen():
        yield "data: before\n\n"
        raise asyncio.CancelledError("upstream cancelled")

    wrapped = with_sse_keep_alive(gen(), 0.01)
    assert await anext(wrapped) == "data: before\n\n"
    with pytest.raises(asyncio.CancelledError):
        await anext(wrapped)


@pytest.mark.asyncio
async def test_aclose_closes_upstream():
    """Closing a wrapper that is mid-stream must complete promptly and close
    the upstream generator (regression for the earlier leaked-task hang)."""
    finalized = asyncio.Event()

    async def gen():
        try:
            yield "chunk-0"
            await asyncio.Event().wait()  # block forever
        finally:
            finalized.set()

    wrapped = with_sse_keep_alive(gen(), 10.0)
    assert await anext(wrapped) == "chunk-0"
    await asyncio.wait_for(wrapped.aclose(), timeout=2.0)
    assert finalized.is_set()


@pytest.mark.asyncio
async def test_client_disconnect_cancels_upstream():
    """Cancelling the consumer mid-stream must cancel the pending anext and
    close the upstream generator so its cleanup runs."""
    finalized = asyncio.Event()

    async def gen():
        try:
            for i in range(1000):
                yield f"chunk-{i}"
            await asyncio.Event().wait()
        finally:
            finalized.set()

    wrapped = with_sse_keep_alive(gen(), 10.0)
    assert await anext(wrapped) == "chunk-0"
    task = asyncio.current_task()
    assert task is not None
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await anext(wrapped)
    assert finalized.is_set()


# --- Response-level integration (the shipped StreamingResponse path) --------


@pytest.mark.asyncio
async def test_streaming_response_emits_comment():
    """A keep-alive comment must reach the client through the real
    StreamingResponse used by the routers."""
    release = asyncio.Event()

    async def gen():
        await release.wait()
        yield "data: hi\n\n"

    response = StreamingResponse(
        content=with_sse_keep_alive(gen(), 0.01),
        media_type="text/event-stream",
    )
    body: list[bytes] = []
    started = asyncio.Event()

    async def send(message):
        if message["type"] == "http.response.start":
            started.set()
        elif message["type"] == "http.response.body":
            body.append(message["body"])
            if SSE_KEEP_ALIVE_COMMENT.encode() in message["body"]:
                release.set()

    async def receive():
        await asyncio.Event().wait()

    await response(
        {"type": "http", "asgi": {"spec_version": "2.3"}},
        receive,
        send,
    )
    data = b"".join(body).decode()
    assert SSE_KEEP_ALIVE_COMMENT in data
    assert "data: hi" in data


@pytest.mark.asyncio
async def test_streaming_response_disconnect_closes_upstream():
    """On the ASGI 2.3 disconnect path used by the pinned Uvicorn, Starlette
    cancels streaming and the wrapper's finally must close the upstream
    generator."""
    finalized = asyncio.Event()

    async def gen():
        try:
            await asyncio.Event().wait()
            yield "never"
        finally:
            finalized.set()

    response = StreamingResponse(
        content=with_sse_keep_alive(gen(), 10.0),
        media_type="text/event-stream",
    )
    started = asyncio.Event()

    async def receive():
        await started.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.start":
            started.set()

    await response(
        {"type": "http", "asgi": {"spec_version": "2.3"}},
        receive,
        send,
    )
    assert finalized.is_set()
    pending = [
        t
        for t in asyncio.all_tasks()
        if t is not asyncio.current_task() and not t.done()
    ]
    assert pending == []
