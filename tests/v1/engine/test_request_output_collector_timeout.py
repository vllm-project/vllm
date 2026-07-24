# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Pure-async unit tests for the request watchdog (issue #33099).
No GPU or vLLM engine required.

Covers:
  1. RequestOutputCollector.get_with_timeout() — timeout fires when nothing
     is ever put.
  2. Output is returned immediately once put() is called.
  3. An exception put via put() is propagated via get_with_timeout().
  4. Queue remains usable after get_with_timeout() times out.
  5. Timestamps (created_at / last_put_at) are accurate.
  6. AsyncLLM.generate() watchdog path — raises EngineRequestTimeoutError,
     calls abort(), and produces the correct error message.
"""

import asyncio

import pytest

from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.output_processor import RequestOutputCollector


def _make_collector() -> RequestOutputCollector:
    return RequestOutputCollector(
        output_kind=RequestOutputKind.DELTA,
        request_id="test-req-0",
    )


@pytest.mark.asyncio
async def test_get_with_timeout_fires_when_nothing_put():
    """get_with_timeout() must raise asyncio.TimeoutError if nothing arrives."""
    q = _make_collector()
    with pytest.raises(asyncio.TimeoutError):
        await q.get_with_timeout(timeout_s=0.05)


@pytest.mark.asyncio
async def test_get_with_timeout_returns_after_put():
    """get_with_timeout() returns the item put by a concurrent producer."""
    from unittest.mock import MagicMock

    from vllm.outputs import RequestOutput

    q = _make_collector()

    # Build a minimal RequestOutput mock that satisfies isinstance checks.
    mock_output = MagicMock(spec=RequestOutput)
    mock_output.finished = True

    async def producer():
        await asyncio.sleep(0.02)
        q.put(mock_output)

    asyncio.create_task(producer())
    result = await q.get_with_timeout(timeout_s=2.0)
    assert result is mock_output


@pytest.mark.asyncio
async def test_get_with_timeout_propagates_exception():
    """get_with_timeout() re-raises an exception put into the queue."""
    q = _make_collector()

    async def error_producer():
        await asyncio.sleep(0.02)
        q.put(RuntimeError("simulated engine error"))

    asyncio.create_task(error_producer())
    with pytest.raises(RuntimeError, match="simulated engine error"):
        await q.get_with_timeout(timeout_s=2.0)


@pytest.mark.asyncio
async def test_get_with_timeout_does_not_consume_existing_nowait_items():
    """
    After get_with_timeout() times out, a subsequent get_nowait() or get()
    should still be able to retrieve an item placed later.
    """
    from unittest.mock import MagicMock

    from vllm.outputs import RequestOutput

    q = _make_collector()

    # First call times out (nothing has been put).
    with pytest.raises(asyncio.TimeoutError):
        await q.get_with_timeout(timeout_s=0.02)

    # Now put an item; get_nowait() should see it.
    mock_output = MagicMock(spec=RequestOutput)
    mock_output.finished = False
    q.put(mock_output)
    result = q.get_nowait()
    assert result is mock_output


@pytest.mark.asyncio
async def test_created_at_and_last_put_at_updated():
    """created_at and last_put_at timestamps are set correctly."""
    import time
    from unittest.mock import MagicMock

    from vllm.outputs import RequestOutput

    before = time.monotonic()
    q = _make_collector()
    after = time.monotonic()

    assert before <= q.created_at <= after
    assert q.last_put_at == q.created_at  # unchanged before any put

    mock_output = MagicMock(spec=RequestOutput)
    mock_output.finished = False
    before_put = time.monotonic()
    q.put(mock_output)
    after_put = time.monotonic()

    assert before_put <= q.last_put_at <= after_put


# ---------------------------------------------------------------------------
# Integration test: AsyncLLM.generate() watchdog path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_watchdog_raises_timeout_and_calls_abort():
    """
    AsyncLLM.generate() must raise EngineRequestTimeoutError and call
    abort() when request_timeout_s fires. No GPU needed — we stub
    out add_request() and abort() on the engine object.
    """
    import unittest.mock as mock

    from vllm.sampling_params import RequestOutputKind, SamplingParams
    from vllm.v1.engine.exceptions import EngineRequestTimeoutError
    from vllm.v1.engine.output_processor import RequestOutputCollector

    # Build a collector that never produces output (simulates a stalled engine).
    stalled_q = RequestOutputCollector(
        output_kind=RequestOutputKind.DELTA,
        request_id="test-watchdog-req",
    )

    # Minimal AsyncLLM stand-in — only the methods generate() actually calls.
    engine = mock.AsyncMock()
    engine.add_request = mock.AsyncMock(return_value=stalled_q)
    engine.abort = mock.AsyncMock()
    engine.log_requests = False

    # Set up scheduler_config so the watchdog fires after 50 ms.
    sched_cfg = mock.MagicMock()
    sched_cfg.request_timeout_s = 0.05
    sched_cfg.request_stall_timeout_s = 0.0
    engine.vllm_config.scheduler_config = sched_cfg

    from vllm.v1.engine.async_llm import AsyncLLM

    gen = AsyncLLM.generate(
        engine,
        prompt="hello",
        sampling_params=SamplingParams(),
        request_id="test-watchdog-req",
    )

    with pytest.raises(EngineRequestTimeoutError) as exc_info:
        async for _ in gen:
            pass

    # abort() must have been called exactly once with the request id.
    engine.abort.assert_awaited_once()
    call_args = engine.abort.call_args
    assert call_args.args[0] == "test-watchdog-req"
    assert call_args.kwargs.get("internal") is True

    # Error message should mention the request id and config field names.
    assert "test-watchdog-req" in str(exc_info.value)
    assert "request_timeout_s" in str(exc_info.value)

    # Must be a subclass of EngineGenerateError (existing handlers match).
    from vllm.v1.engine.exceptions import EngineGenerateError

    assert isinstance(exc_info.value, EngineGenerateError)
