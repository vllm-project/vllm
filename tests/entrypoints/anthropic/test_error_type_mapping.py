# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for Anthropic ``error.type`` translation (issue #57997)."""

import json

import pytest

from vllm.entrypoints.anthropic.api_router import translate_error_response
from vllm.entrypoints.serve.engine.protocol import ErrorInfo, ErrorResponse

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    ("status_code", "expected_type"),
    [
        (400, "invalid_request_error"),
        (401, "authentication_error"),
        (403, "permission_error"),
        (404, "not_found_error"),
        (413, "request_too_large"),
        (418, "invalid_request_error"),
        (422, "invalid_request_error"),
        (429, "rate_limit_error"),
        (500, "api_error"),
        (501, "api_error"),
        # vLLM overload (e.g. QueueOverflowError) is 503; Anthropic overload
        # vocabulary is overloaded_error, which it documents at 529. Map both.
        (503, "overloaded_error"),
        (529, "overloaded_error"),
    ],
)
def test_error_type_is_translated_to_anthropic_vocabulary(
    status_code: int, expected_type: str
):
    response = translate_error_response(
        ErrorResponse(
            error=ErrorInfo(
                message="boom",
                # Deliberately use an internal type name to prove it is ignored.
                type="NotFoundError",
                code=status_code,
            )
        )
    )
    assert response.status_code == status_code
    assert json.loads(response.body) == {
        "type": "error",
        "error": {"type": expected_type, "message": "boom"},
    }


@pytest.mark.parametrize(
    ("status_code", "expected_type"),
    [
        (404, "not_found_error"),
        (429, "rate_limit_error"),
        (503, "overloaded_error"),
        (500, "api_error"),
    ],
)
@pytest.mark.asyncio
async def test_stream_error_event_uses_translated_type(
    status_code: int, expected_type: str
):
    """The SSE error path must translate ``error.type`` the same way as the
    non-streaming path, preserving the HTTP status from ErrorResponse."""
    from vllm.entrypoints.anthropic.serving import AnthropicServingMessages

    async def fake_generator():
        yield ErrorResponse(
            error=ErrorInfo(
                message="boom",
                type="InternalError",
                code=status_code,
            )
        )

    converter = AnthropicServingMessages.__new__(AnthropicServingMessages)
    events = [e async for e in converter.message_stream_converter(fake_generator())]
    error_events = [e for e in events if e.startswith("event: error")]
    assert len(error_events) == 1
    payload_line = [
        line for line in error_events[0].splitlines() if line.startswith("data: ")
    ][0]
    payload = json.loads(payload_line[len("data: "):])
    assert payload["type"] == "error"
    assert payload["error"]["type"] == expected_type
    assert payload["error"]["message"] == "boom"
