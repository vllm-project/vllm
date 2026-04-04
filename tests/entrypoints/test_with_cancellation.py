# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for the with_cancellation decorator in vllm.entrypoints.utils.

Verifies that the decorator never returns None (which would cause FastAPI
to send HTTP 200 with an empty body) and properly raises CancelledError
when the client disconnects.
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.utils import with_cancellation


def _make_mock_request(disconnect_event: asyncio.Event | None = None):
    """Create a mock Request whose receive() never sends disconnect
    unless disconnect_event is set."""
    request = MagicMock()

    async def mock_receive():
        if disconnect_event is not None:
            await disconnect_event.wait()
            return {"type": "http.disconnect"}
        # Never disconnect - wait forever
        await asyncio.Future()

    request.receive = mock_receive
    request.app = MagicMock()
    request.app.state = MagicMock()
    request.app.state.enable_server_load_tracking = False
    return request


@pytest.mark.asyncio
async def test_with_cancellation_returns_handler_result():
    """Test that the decorator returns the handler's result normally."""
    expected = {"status": "ok", "data": [1, 2, 3]}

    @with_cancellation
    async def handler(self, raw_request):
        return expected

    request = _make_mock_request()
    result = await handler(None, raw_request=request)
    assert result == expected


@pytest.mark.asyncio
async def test_with_cancellation_raises_on_disconnect():
    """Test that disconnection raises CancelledError instead of returning None."""
    disconnect_event = asyncio.Event()

    @with_cancellation
    async def handler(self, raw_request):
        # Simulate a slow handler that takes longer than disconnect
        await asyncio.sleep(10)
        return {"should": "not reach"}

    request = _make_mock_request(disconnect_event)

    # Trigger disconnect shortly after starting
    async def trigger_disconnect():
        await asyncio.sleep(0.05)
        disconnect_event.set()

    asyncio.create_task(trigger_disconnect())

    with pytest.raises(asyncio.CancelledError):
        await handler(None, raw_request=request)


@pytest.mark.asyncio
async def test_with_cancellation_no_false_none_under_load():
    """Test that under high concurrent load, the decorator never returns None.

    This is the primary regression test: previously, under concurrent load,
    the ASGI layer could deliver false http.disconnect messages, causing
    the decorator to return None instead of the handler result.
    """
    call_count = 0

    @with_cancellation
    async def handler(self, raw_request):
        nonlocal call_count
        call_count += 1
        # Simulate some async work
        await asyncio.sleep(0.01)
        return {"result": call_count}

    num_concurrent = 200

    async def run_one(idx: int):
        request = _make_mock_request()
        result = await handler(None, raw_request=request)
        return idx, result

    tasks = [asyncio.create_task(run_one(i)) for i in range(num_concurrent)]
    done_tasks = await asyncio.gather(*tasks)

    for idx, result in done_tasks:
        assert result is not None, (
            f"Request {idx}: with_cancellation returned None "
            "(would cause empty HTTP response body)"
        )
        assert "result" in result, f"Request {idx}: unexpected result format: {result}"


@pytest.mark.asyncio
async def test_with_cancellation_handler_exception_propagates():
    """Test that exceptions from the handler propagate correctly."""

    @with_cancellation
    async def handler(self, raw_request):
        raise ValueError("test error")

    request = _make_mock_request()

    with pytest.raises(ValueError, match="test error"):
        await handler(None, raw_request=request)


@pytest.mark.asyncio
async def test_with_cancellation_streaming_response_passthrough():
    """Test that StreamingResponse is returned without interference."""
    from fastapi.responses import StreamingResponse

    async def fake_generator():
        yield b"chunk1"
        yield b"chunk2"

    expected_response = StreamingResponse(fake_generator())

    @with_cancellation
    async def handler(self, raw_request):
        return expected_response

    request = _make_mock_request()
    result = await handler(None, raw_request=request)
    assert result is expected_response
