# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the ``openai-responses`` backend of ``vllm bench serve``.

The Responses stream differs from Chat Completions in ways the benchmark client
has to get right: every SSE message is labelled with an ``event:`` line,
reasoning and output text arrive as separate token-bearing events, and a
request ends with ``response.completed`` rather than with its last token. These
tests drive the real request function against a local SSE server so the
resulting latency and usage numbers are covered end to end.
"""

import asyncio
import json
from collections.abc import Sequence

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from vllm.benchmarks.lib.endpoint_request_func import (
    ASYNC_REQUEST_FUNCS,
    OPENAI_COMPATIBLE_BACKENDS,
    RequestFuncInput,
    RequestFuncOutput,
    async_request_openai_responses,
)

# These tests never touch torch, so skip the GPU cleanup fixture.
pytestmark = pytest.mark.skip_global_cleanup

MODEL = "openai/gpt-oss-20b"

# Gaps between scripted events. GAP_TOKEN is much larger than GAP_PREFILL so
# that a time to first token measured at the wrong event is unambiguous.
GAP_PREFILL = 0.15
GAP_TOKEN = 0.5
GAP_FINALIZE = 0.3


def sse(event_type: str, payload: dict) -> bytes:
    """Frame one event the way ``/v1/responses`` does, with an ``event:`` line."""
    data = json.dumps({"type": event_type, **payload})
    return f"event: {event_type}\ndata: {data}\n\n".encode()


def usage_event(input_tokens: int, output_tokens: int, status: str = "completed"):
    return sse(
        f"response.{status}",
        {
            "response": {
                "status": status,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
            }
        },
    )


def text_delta(delta: str) -> bytes:
    return sse("response.output_text.delta", {"delta": delta})


def reasoning_delta(delta: str) -> bytes:
    return sse("response.reasoning_text.delta", {"delta": delta})


PREAMBLE = (
    sse("response.created", {"response": {"status": "in_progress"}})
    + sse("response.in_progress", {"response": {"status": "in_progress"}})
    + sse("response.output_item.added", {"output_index": 0})
    + sse("response.content_part.added", {"output_index": 0})
)


async def _serve(
    script: Sequence[bytes | float],
    *,
    status: int = 200,
    chunk_size: int | None = None,
    seen: dict | None = None,
    **request_kwargs,
) -> RequestFuncOutput:
    """Run the request function against a server replaying ``script``.

    Args:
        script: Byte strings to write, interleaved with floats to sleep for.
        status: HTTP status to respond with. Non-200 skips the script.
        chunk_size: If set, split every write into chunks of this many bytes to
            emulate fragmented transport.
        seen: If given, populated with the request payload.
        request_kwargs: Extra fields for the ``RequestFuncInput``.
    """

    async def handler(request: web.Request) -> web.StreamResponse:
        if seen is not None:
            seen["payload"] = await request.json()
        if status != 200:
            return web.Response(status=status, reason="Bad Request")

        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for item in script:
            if isinstance(item, float):
                await asyncio.sleep(item)
            elif chunk_size is None:
                await response.write(item)
            else:
                for i in range(0, len(item), chunk_size):
                    await response.write(item[i : i + chunk_size])
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v1/responses", handler)
    server = TestServer(app)
    await server.start_server()
    try:
        request_input = RequestFuncInput(
            prompt="Why is the sky blue?",
            api_url=str(server.make_url("/v1/responses")),
            prompt_len=5,
            output_len=16,
            model=MODEL,
            **request_kwargs,
        )
        async with aiohttp.ClientSession() as session:
            return await async_request_openai_responses(request_input, session)
    finally:
        await server.close()


def run(script: Sequence[bytes | float], **kwargs) -> RequestFuncOutput:
    return asyncio.run(_serve(script, **kwargs))


def test_backend_is_registered_as_openai_compatible():
    assert ASYNC_REQUEST_FUNCS["openai-responses"] is async_request_openai_responses
    # Membership enables --ignore-eos on random datasets and the sampling
    # parameter flags, all of which ResponsesRequest accepts.
    assert "openai-responses" in OPENAI_COMPATIBLE_BACKENDS


def test_request_payload_is_accepted_by_the_server_schema():
    """Every field sent must be a real ResponsesRequest field.

    This is what makes the backend safe to list in OPENAI_COMPATIBLE_BACKENDS:
    the flags that membership enables have to reach the model, and the server
    drops unknown fields without complaining.
    """
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

    sampling = {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 20,
        "frequency_penalty": 0.2,
        "presence_penalty": 0.1,
        "repetition_penalty": 1.05,
    }
    seen: dict = {}
    output = run(
        [PREAMBLE, text_delta("hi"), usage_event(5, 1)],
        seen=seen,
        extra_body=sampling,
        ignore_eos=True,
    )

    assert output.success
    assert seen["payload"] == {
        "model": MODEL,
        "input": "Why is the sky blue?",
        "max_output_tokens": 16,
        "stream": True,
        "ignore_eos": True,
        **sampling,
    }
    assert set(seen["payload"]) <= set(ResponsesRequest.model_fields)
    ResponsesRequest.model_validate(seen["payload"])

    # `vllm bench serve` rejects --min-p for this backend for the same reason.
    assert "min_p" not in ResponsesRequest.model_fields


def test_latency_is_measured_from_token_events_to_completion():
    output = run(
        [
            PREAMBLE,
            GAP_PREFILL / 2,
            # A keep-alive comment sent during prefill is not a token.
            b": keep-alive\n\n",
            GAP_PREFILL / 2,
            reasoning_delta("The user asks "),
            GAP_TOKEN,
            text_delta("Rayleigh"),
            text_delta(" scattering."),
            GAP_FINALIZE,
            usage_event(5, 8),
        ]
    )

    assert output.success
    # Reasoning is timed but not returned, matching how `openai-chat` treats
    # `DeltaMessage.reasoning`.
    assert output.generated_text == "Rayleigh scattering."

    # Neither the response.created/in_progress/output_item.added preamble nor
    # the keep-alive comment may start the clock, and the reasoning delta must,
    # so TTFT lands on the prefill gap rather than near zero or out at the
    # first output text delta.
    assert GAP_PREFILL * 0.8 <= output.ttft < GAP_PREFILL + GAP_TOKEN * 0.5

    # One ITL per token event after the first: the gap before "Rayleigh" and
    # the back-to-back delta that follows it.
    assert len(output.itl) == 2
    assert output.itl[0] >= GAP_TOKEN * 0.8
    assert output.itl[1] < GAP_TOKEN * 0.5

    # E2EL runs to response.completed, not to the last token.
    last_token_at = output.ttft + sum(output.itl)
    assert output.latency >= last_token_at + GAP_FINALIZE * 0.8


def test_usage_is_taken_from_the_terminal_event():
    output = run([PREAMBLE, text_delta("blue"), usage_event(17, 4)])

    assert output.success
    assert output.prompt_len == 17
    assert output.output_tokens == 4


def test_incomplete_response_is_a_successful_request():
    """Hitting max_output_tokens is the Responses form of finish_reason=length."""
    output = run(
        [PREAMBLE, text_delta("truncated"), usage_event(17, 16, status="incomplete")]
    )

    assert output.success
    assert output.generated_text == "truncated"
    assert output.output_tokens == 16


@pytest.mark.parametrize("chunk_size", [1, 3, 7, 64])
def test_output_survives_transport_chunk_boundaries(chunk_size: int):
    """Splits inside the event line, the JSON, or a multibyte character."""
    text = " leading 中 trailing "
    output = run(
        [PREAMBLE, text_delta(text), usage_event(5, 6)],
        chunk_size=chunk_size,
    )

    assert output.success
    assert output.generated_text == text
    assert output.output_tokens == 6


def test_http_error_is_not_reported_as_success():
    output = run([], status=400)

    assert not output.success
    assert output.error


def test_stream_without_tokens_is_not_reported_as_success():
    """A stream that finishes without generating anything has no TTFT."""
    output = run([PREAMBLE, usage_event(5, 0)])

    assert not output.success
    assert "TTFT" in output.error


def test_truncated_stream_is_not_reported_as_success():
    """A stream cut short after a token has no usage and an understated E2EL."""
    output = run([PREAMBLE, text_delta("partial")])

    assert not output.success
    assert "terminal event" in output.error


def test_mid_stream_error_is_not_reported_as_success():
    """vLLM reports a generation failure as an error payload mid-stream."""
    output = run(
        [
            PREAMBLE,
            text_delta("partial"),
            b'event: error\ndata: {"error": {"message": "boom", "code": 500}}\n\n',
        ]
    )

    assert not output.success
    assert "boom" in output.error


def test_data_only_framing_is_still_parsed():
    """Some proxies drop the ``event:`` line and forward only ``data:``."""
    script: list[bytes | float] = [
        b'data: {"type": "response.output_text.delta", "delta": "ok"}\n\n',
        usage_event(5, 1),
    ]
    output = run(script)

    assert output.success
    assert output.generated_text == "ok"


def test_endpoint_path_is_validated():
    request_input = RequestFuncInput(
        prompt="hi",
        api_url="http://localhost:8000/v1/chat/completions",
        prompt_len=1,
        output_len=1,
        model=MODEL,
    )

    async def call() -> None:
        async with aiohttp.ClientSession() as session:
            await async_request_openai_responses(request_input, session)

    with pytest.raises(ValueError, match="responses"):
        asyncio.run(call())
