# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the Realtime API WebSocket connection handler."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vllm.entrypoints.openai.realtime.connection import RealtimeConnection


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket."""
    ws = MagicMock()
    ws.send_text = AsyncMock()
    return ws


@pytest.fixture
def mock_serving():
    """Create a mock OpenAIServingRealtime."""
    serving = MagicMock()
    serving._is_model_supported = MagicMock(return_value=True)
    return serving


@pytest.fixture
def connection(mock_websocket, mock_serving):
    """Create a RealtimeConnection with mocked dependencies."""
    return RealtimeConnection(mock_websocket, mock_serving)


@pytest.mark.asyncio
async def test_commit_without_model_validation_sends_error(connection, mock_websocket):
    """Commit before session.update should send an error and not start generation.

    Regression test: previously, the missing `return` after send_error
    meant execution would continue and potentially start generation
    without a validated model.
    """
    with patch.object(connection, "start_generation", new=AsyncMock()) as mock_gen:
        await connection.handle_event({"type": "input_audio_buffer.commit"})

        # Should have sent an error event
        assert mock_websocket.send_text.called
        sent_data = json.loads(mock_websocket.send_text.call_args[0][0])
        assert sent_data["type"] == "error"
        assert "model_not_validated" in sent_data.get("code", "")

        # Should NOT have started generation after the error
        mock_gen.assert_not_called()


@pytest.mark.asyncio
async def test_commit_final_without_model_validation_sends_error(
    connection, mock_websocket
):
    """Final commit before session.update should send an error and not queue None.

    Regression test: without the return statement, a final commit before
    model validation would still put None into the audio queue.
    """
    initial_queue_size = connection.audio_queue.qsize()

    await connection.handle_event({"type": "input_audio_buffer.commit", "final": True})

    # Should have sent an error event
    assert mock_websocket.send_text.called
    sent_data = json.loads(mock_websocket.send_text.call_args[0][0])
    assert sent_data["type"] == "error"

    # Audio queue should be unchanged (no None sentinel added)
    assert connection.audio_queue.qsize() == initial_queue_size


@pytest.mark.asyncio
async def test_commit_with_validated_model_starts_generation(
    connection, mock_websocket
):
    """Non-final commit after session.update should start generation."""
    # First validate the model
    await connection.handle_event({"type": "session.update", "model": "test-model"})
    assert connection._is_model_validated

    with patch.object(connection, "start_generation", new=AsyncMock()) as mock_gen:
        await connection.handle_event({"type": "input_audio_buffer.commit"})
        mock_gen.assert_called_once()


@pytest.mark.asyncio
async def test_final_commit_with_validated_model_queues_sentinel(
    connection, mock_websocket
):
    """Final commit after session.update should put None sentinel in audio queue."""
    # First validate the model
    await connection.handle_event({"type": "session.update", "model": "test-model"})

    assert connection.audio_queue.empty()
    await connection.handle_event({"type": "input_audio_buffer.commit", "final": True})

    # None sentinel should be in the audio queue
    assert not connection.audio_queue.empty()
    sentinel = connection.audio_queue.get_nowait()
    assert sentinel is None
