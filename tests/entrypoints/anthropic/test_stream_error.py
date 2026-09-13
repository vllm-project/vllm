# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for mid-stream error handling in the Anthropic messages converter.

Regression for issue #46028: an upstream engine failure is delivered to
``message_stream_converter`` as an ``ErrorResponse`` payload
(``{"error": {...}}``), not a ``ChatCompletionStreamResponse``. The converter
must forward the real upstream message as an Anthropic ``error`` event instead
of masking it with a pydantic schema-validation error.

These tests are CPU-only and do not start a server.
"""

import json

import pytest

from vllm.entrypoints.anthropic.serving import (
    AnthropicServingMessages,
    parse_streaming_error_chunk,
)
from vllm.entrypoints.openai.engine.protocol import ErrorInfo, ErrorResponse


def _error_chunk(message: str) -> str:
    """Build the SSE line an upstream engine emits for a mid-stream failure.

    Mirrors ``create_streaming_error_response`` in the OpenAI serving layer:
    ``json.dumps(ErrorResponse(...).model_dump())`` wrapped in an SSE ``data:``
    frame.
    """
    payload = ErrorResponse(
        error=ErrorInfo(message=message, type="InternalServerError", code=500)
    )
    return f"data: {json.dumps(payload.model_dump())}\n\n"


def test_parse_streaming_error_chunk_detects_error_payload():
    err = parse_streaming_error_chunk(
        '{"error": {"message": "Internal Server Error", '
        '"type": "InternalServerError", "code": 500}}'
    )
    assert err is not None
    assert err.error.message == "Internal Server Error"


def test_parse_streaming_error_chunk_ignores_normal_chunk():
    # A regular (non-error) streaming chunk has no top-level ``error`` field.
    assert (
        parse_streaming_error_chunk(
            '{"id": "x", "object": "chat.completion.chunk", '
            '"created": 0, "model": "m", "choices": []}'
        )
        is None
    )


def test_parse_streaming_error_chunk_ignores_malformed_json():
    assert parse_streaming_error_chunk("not json") is None
    assert parse_streaming_error_chunk('{"foo": 1}') is None


@pytest.mark.asyncio
async def test_converter_forwards_upstream_error_message():
    async def upstream():
        yield _error_chunk("Internal Server Error")
        yield "data: [DONE]\n\n"

    # The error path never touches initialized instance state, so a bare
    # instance is sufficient to exercise the converter.
    serving = object.__new__(AnthropicServingMessages)
    events = [event async for event in serving.message_stream_converter(upstream())]

    joined = "".join(events)
    assert "event: error" in joined
    assert "Internal Server Error" in joined
    # The real message must not be masked by a schema-validation error.
    assert "validation error" not in joined.lower()

    # The error event must carry the upstream message in the Anthropic shape.
    error_lines = [e for e in events if "event: error" in e]
    assert len(error_lines) == 1
    data_str = error_lines[0].split("data:", 1)[1].strip()
    parsed = json.loads(data_str)
    assert parsed["type"] == "error"
    assert parsed["error"]["message"] == "Internal Server Error"
