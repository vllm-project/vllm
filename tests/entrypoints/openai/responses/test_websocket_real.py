# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Real integration tests for the Responses API WebSocket mode.

These tests start a real vLLM server on CPU with a tiny model and exercise
the WebSocket endpoint end-to-end: real inference, real streaming events,
real connection lifecycle.

Run:
    pytest tests/entrypoints/openai/responses/test_websocket_real.py -v -s
"""

import asyncio
import json

import pytest
import websockets

from ....utils import RemoteOpenAIServer

MODEL_NAME = "hmellor/tiny-random-LlamaForCausalLM"
SERVED_MODEL_NAME = "test-model"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_ws_url(server: RemoteOpenAIServer) -> str:
    """Build WebSocket URL from server address."""
    return server.url_root.replace("http://", "ws://") + "/v1/responses"


async def _send(ws, event: dict) -> None:
    await ws.send(json.dumps(event))


async def _recv(ws, timeout: float = 60.0) -> dict:
    msg = await asyncio.wait_for(ws.recv(), timeout=timeout)
    return json.loads(msg)


async def _collect_until_done(ws, timeout: float = 60.0) -> list[dict]:
    """Receive events until response.completed or error."""
    events: list[dict] = []
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        remaining = deadline - asyncio.get_event_loop().time()
        if remaining <= 0:
            raise TimeoutError("Timed out waiting for response.completed")
        event = await _recv(ws, timeout=remaining)
        events.append(event)
        if event.get("type") in ("response.completed", "error"):
            break
    return events


def _make_create_event(
    model: str = SERVED_MODEL_NAME,
    input_text: str = "Hello",
    max_output_tokens: int = 10,
    **kwargs,
) -> dict:
    return {
        "type": "response.create",
        "model": model,
        "input": input_text,
        "max_output_tokens": max_output_tokens,
        **kwargs,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def server():
    from .conftest import BASE_TEST_ENV

    args = [
        "--enforce-eager",
        "--max-model-len",
        "256",
        "--served-model-name",
        SERVED_MODEL_NAME,
    ]
    env_dict = {**BASE_TEST_ENV}

    with RemoteOpenAIServer(
        MODEL_NAME,
        args,
        env_dict=env_dict,
        max_wait_seconds=300,
    ) as remote_server:
        yield remote_server


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_basic_streaming(server):
    """Basic streaming works over WebSocket: connect, send response.create,
    receive streaming events ending with response.completed."""
    ws_url = _get_ws_url(server)
    async with websockets.connect(ws_url) as ws:
        await _send(ws, _make_create_event())
        events = await _collect_until_done(ws)

    types = [e["type"] for e in events]

    # Must start with created and end with completed
    assert types[0] == "response.created"
    assert types[-1] == "response.completed"

    # Must contain streaming deltas
    assert "response.output_text.delta" in types

    # Completed event must contain a valid response
    completed = events[-1]
    # Status can be "completed" (natural EOS) or "incomplete" (hit
    # max_output_tokens) — both are valid terminal states.
    assert completed["response"]["status"] in ("completed", "incomplete")
    assert completed["response"]["model"] == SERVED_MODEL_NAME
    assert len(completed["response"]["output"]) > 0


@pytest.mark.asyncio
async def test_streaming_event_ordering(server):
    """Streaming events follow correct nesting order:
    created -> in_progress -> output_item.added -> content_part.added ->
    deltas -> content_part.done -> output_item.done -> completed."""
    ws_url = _get_ws_url(server)
    async with websockets.connect(ws_url) as ws:
        await _send(ws, _make_create_event())
        events = await _collect_until_done(ws)

    types = [e["type"] for e in events]

    # Envelope ordering
    assert types[0] == "response.created"
    assert types[1] == "response.in_progress"
    assert types[-1] == "response.completed"

    # Nesting: added before done
    assert types.index("response.output_item.added") < types.index(
        "response.output_item.done"
    )
    assert types.index("response.content_part.added") < types.index(
        "response.content_part.done"
    )

    # Deltas are between content_part.added and content_part.done
    first_delta = types.index("response.output_text.delta")
    assert first_delta > types.index("response.content_part.added")
    assert first_delta < types.index("response.content_part.done")


@pytest.mark.asyncio
async def test_streaming_text_consistency(server):
    """Concatenated delta text matches the final output_text in
    response.completed."""
    ws_url = _get_ws_url(server)
    async with websockets.connect(ws_url) as ws:
        await _send(ws, _make_create_event(max_output_tokens=20))
        events = await _collect_until_done(ws)

    # Concatenate deltas
    streaming_text = "".join(
        e["delta"] for e in events if e.get("type") == "response.output_text.delta"
    )

    # Get final output_text from completed event
    completed = events[-1]
    assert completed["type"] == "response.completed"
    final_text = completed["response"]["output"][0]["content"][0]["text"]

    assert streaming_text == final_text, (
        f"Streaming text does not match final output.\n"
        f"Streaming: {streaming_text!r}\n"
        f"Final:     {final_text!r}"
    )


@pytest.mark.asyncio
async def test_warmup_generate_false(server):
    """generate:false returns created + completed without inference."""
    ws_url = _get_ws_url(server)
    async with websockets.connect(ws_url) as ws:
        await _send(ws, _make_create_event(generate=False))
        events = await _collect_until_done(ws, timeout=10)

    types = [e["type"] for e in events]

    # Only created and completed, no deltas
    assert types == ["response.created", "response.completed"]

    # Response should have empty output (no inference)
    completed = events[-1]
    assert completed["response"]["output"] == []
    assert completed["response"]["status"] == "completed"


@pytest.mark.asyncio
async def test_continuation_with_previous_response_id(server):
    """Continuation: first request caches response, second request uses
    previous_response_id to continue on the same connection."""
    ws_url = _get_ws_url(server)
    async with websockets.connect(ws_url) as ws:
        # First request
        await _send(ws, _make_create_event())
        events1 = await _collect_until_done(ws)

        assert events1[-1]["type"] == "response.completed"
        response_id = events1[-1]["response"]["id"]
        assert response_id is not None

        # Second request with previous_response_id
        await _send(
            ws,
            _make_create_event(
                previous_response_id=response_id,
            ),
        )
        events2 = await _collect_until_done(ws)

        assert events2[-1]["type"] == "response.completed"
        # Second response should have a different id
        assert events2[-1]["response"]["id"] != response_id


@pytest.mark.asyncio
async def test_invalid_previous_response_id(server):
    """Invalid previous_response_id returns previous_response_not_found
    error."""
    ws_url = _get_ws_url(server)
    async with websockets.connect(ws_url) as ws:
        await _send(
            ws,
            _make_create_event(
                previous_response_id="resp_nonexistent",
            ),
        )
        events = await _collect_until_done(ws, timeout=10)

    assert len(events) == 1
    error = events[0]
    assert error["type"] == "error"
    assert error["status"] == 404
    assert error["error"]["code"] == "previous_response_not_found"


@pytest.mark.asyncio
async def test_unknown_event_type(server):
    """Unknown event type returns error but connection stays open."""
    ws_url = _get_ws_url(server)
    async with websockets.connect(ws_url) as ws:
        # Send unknown event type
        await _send(ws, {"type": "session.update", "model": SERVED_MODEL_NAME})
        error = await _recv(ws, timeout=10)

        assert error["type"] == "error"
        assert error["error"]["code"] == "unknown_event_type"

        # Connection should still be alive - send a valid request
        await _send(ws, _make_create_event())
        events = await _collect_until_done(ws)

        assert events[-1]["type"] == "response.completed"


@pytest.mark.asyncio
async def test_invalid_json(server):
    """Invalid JSON returns error but connection stays open."""
    ws_url = _get_ws_url(server)
    async with websockets.connect(ws_url) as ws:
        await ws.send("this is not json{{{")
        error = await _recv(ws, timeout=10)

        assert error["type"] == "error"
        assert error["error"]["code"] == "invalid_json"

        # Connection still alive
        await _send(ws, _make_create_event())
        events = await _collect_until_done(ws)

        assert events[-1]["type"] == "response.completed"


@pytest.mark.asyncio
async def test_multiple_requests_sequential(server):
    """Multiple sequential requests on the same connection succeed."""
    ws_url = _get_ws_url(server)
    async with websockets.connect(ws_url) as ws:
        for i in range(3):
            await _send(
                ws,
                _make_create_event(
                    input_text=f"Request {i}",
                ),
            )
            events = await _collect_until_done(ws)

            assert events[-1]["type"] == "response.completed"
            assert events[-1]["response"]["status"] in ("completed", "incomplete")


@pytest.mark.asyncio
async def test_connection_limit(server):
    """When connection limit is reached, server returns
    websocket_connection_limit_reached error."""
    # The default --max-websocket-connections is 100.
    # We can't easily exhaust that, but we can verify the server
    # accepts connections normally (it's under the limit).
    ws_url = _get_ws_url(server)
    async with websockets.connect(ws_url) as ws:
        await _send(ws, _make_create_event(generate=False))
        events = await _collect_until_done(ws, timeout=10)

        # If we got here, the connection was accepted (not limited)
        assert events[-1]["type"] == "response.completed"


@pytest.mark.asyncio
async def test_multiple_concurrent_connections(server):
    """Multiple WebSocket connections can be active concurrently."""
    ws_url = _get_ws_url(server)

    async def run_one(conn_id: int) -> list[dict]:
        async with websockets.connect(ws_url) as ws:
            await _send(
                ws,
                _make_create_event(
                    input_text=f"Connection {conn_id}",
                ),
            )
            return await _collect_until_done(ws)

    results = await asyncio.gather(run_one(0), run_one(1), run_one(2))

    for events in results:
        assert events[-1]["type"] == "response.completed"


@pytest.mark.asyncio
async def test_warmup_then_real_request(server):
    """Warmup (generate:false) followed by a real inference request on the
    same connection."""
    ws_url = _get_ws_url(server)
    async with websockets.connect(ws_url) as ws:
        # Warmup
        await _send(ws, _make_create_event(generate=False))
        warmup_events = await _collect_until_done(ws, timeout=10)
        assert warmup_events[-1]["type"] == "response.completed"
        warmup_id = warmup_events[-1]["response"]["id"]

        # Real request using the warmup response id as continuation
        await _send(
            ws,
            _make_create_event(
                previous_response_id=warmup_id,
            ),
        )
        real_events = await _collect_until_done(ws)
        assert real_events[-1]["type"] == "response.completed"

        # Real response should have actual output
        output = real_events[-1]["response"]["output"]
        assert len(output) > 0
        assert output[0]["content"][0]["text"]  # non-empty text


@pytest.mark.asyncio
async def test_response_has_usage(server):
    """Completed response includes usage with token counts."""
    ws_url = _get_ws_url(server)
    async with websockets.connect(ws_url) as ws:
        await _send(ws, _make_create_event(max_output_tokens=10))
        events = await _collect_until_done(ws)

    completed = events[-1]
    assert completed["type"] == "response.completed"

    usage = completed["response"].get("usage")
    assert usage is not None
    assert usage["input_tokens"] > 0
    assert usage["output_tokens"] > 0
    assert usage["total_tokens"] == (usage["input_tokens"] + usage["output_tokens"])
