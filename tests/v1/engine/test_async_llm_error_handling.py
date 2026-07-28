# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for ValueError/VLLMValidationError abort behaviour in AsyncLLM.

These tests verify that when a ValueError (or its subclass VLLMValidationError)
is raised *after* add_request() has already returned a queue — meaning the
request is live in both the OutputProcessor and EngineCore scheduler — the
generate() and encode() methods call abort() to clean up that state.

Prior to the fix, the ``except ValueError`` handlers in both methods did not
call abort(), unlike every other exception handler.  This left requests
permanently stranded in the scheduler, eventually exhausting the KV cache and
deadlocking all subsequent requests.

No CUDA / real model is required; all engine internals are mocked.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm.exceptions import VLLMValidationError
from vllm.v1.engine.async_llm import AsyncLLM


def _make_engine() -> AsyncLLM:
    """Return an AsyncLLM instance with all internals mocked out."""
    engine = object.__new__(AsyncLLM)
    engine.log_requests = False
    # output_handler is checked only inside add_request/_run_output_handler,
    # both of which are mocked below, so the value here doesn't matter.
    engine.output_handler = MagicMock()
    return engine


def _make_queue(request_id: str, side_effect: Exception) -> MagicMock:
    """Return a mock RequestOutputCollector whose get() raises side_effect."""
    q = MagicMock()
    q.request_id = request_id
    q.get_nowait.return_value = None
    q.get = AsyncMock(side_effect=side_effect)
    q.close = MagicMock()
    return q


# ---------------------------------------------------------------------------
# generate() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_aborts_on_valuerror_after_queue_created():
    """
    If a plain ValueError propagates into the generate() loop via the queue
    (e.g. from output_handler.propagate_error), abort() must be called so the
    scheduler doesn't retain a zombie request.
    """
    engine = _make_engine()
    mock_queue = _make_queue("req-1", ValueError("test error"))

    engine.add_request = AsyncMock(return_value=mock_queue)
    engine.abort = AsyncMock()

    with pytest.raises(ValueError, match="test error"):
        async for _ in engine.generate(
            prompt={"prompt_token_ids": [1, 2, 3]},
            sampling_params=MagicMock(),
            request_id="req-1",
        ):
            pass

    engine.abort.assert_called_once_with("req-1", internal=True)


@pytest.mark.asyncio
async def test_generate_aborts_on_vllm_validation_error_after_queue_created():
    """
    VLLMValidationError is a ValueError subclass. If it is raised after the
    request is already in the scheduler (add_request succeeded), generate()
    must still abort the request.
    """
    engine = _make_engine()
    error = VLLMValidationError(
        "max_tokens must be at least 1, got 0.",
        parameter="max_tokens",
        value=0,
    )
    mock_queue = _make_queue("req-2", error)

    engine.add_request = AsyncMock(return_value=mock_queue)
    engine.abort = AsyncMock()

    with pytest.raises(VLLMValidationError):
        async for _ in engine.generate(
            prompt={"prompt_token_ids": [1, 2, 3]},
            sampling_params=MagicMock(),
            request_id="req-2",
        ):
            pass

    engine.abort.assert_called_once_with("req-2", internal=True)


@pytest.mark.asyncio
async def test_generate_no_abort_when_add_request_raises():
    """
    When the ValueError comes from add_request() itself (before the queue is
    created), q is None and abort() must NOT be called — there is nothing to
    clean up.
    """
    engine = _make_engine()
    engine.add_request = AsyncMock(
        side_effect=ValueError("prompt too long")
    )
    engine.abort = AsyncMock()

    with pytest.raises(ValueError, match="prompt too long"):
        async for _ in engine.generate(
            prompt={"prompt_token_ids": [1, 2, 3]},
            sampling_params=MagicMock(),
            request_id="req-3",
        ):
            pass

    engine.abort.assert_not_called()


# ---------------------------------------------------------------------------
# encode() tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_encode_aborts_on_valuerror_after_queue_created():
    """Same invariant as generate(), but for encode()."""
    engine = _make_engine()
    mock_queue = _make_queue("req-4", ValueError("test error"))

    engine.add_request = AsyncMock(return_value=mock_queue)
    engine.abort = AsyncMock()

    with pytest.raises(ValueError, match="test error"):
        async for _ in engine.encode(
            prompt={"prompt_token_ids": [1, 2, 3]},
            pooling_params=MagicMock(),
            request_id="req-4",
        ):
            pass

    engine.abort.assert_called_once_with("req-4", internal=True)


@pytest.mark.asyncio
async def test_encode_no_abort_when_add_request_raises():
    """When add_request() itself raises, abort() must NOT be called."""
    engine = _make_engine()
    engine.add_request = AsyncMock(
        side_effect=ValueError("unsupported pooling task")
    )
    engine.abort = AsyncMock()

    with pytest.raises(ValueError, match="unsupported pooling task"):
        async for _ in engine.encode(
            prompt={"prompt_token_ids": [1, 2, 3]},
            pooling_params=MagicMock(),
            request_id="req-5",
        ):
            pass

    engine.abort.assert_not_called()
