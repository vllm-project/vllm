# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

import vllm.benchmarks.lib.ready_checker as ready_checker
from vllm.benchmarks.lib.endpoint_request_func import (
    RequestFuncInput,
    RequestFuncOutput,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.skip_global_cleanup]


@pytest.fixture
def test_input() -> RequestFuncInput:
    return RequestFuncInput(
        prompt="hello",
        api_url="http://localhost:8000/v1/completions",
        prompt_len=1,
        output_len=1,
        model="test-model",
    )


async def test_readiness_timeout_cancels_stalled_request(test_input):
    cancelled = asyncio.Event()

    async def stalled_request(**kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    output = await asyncio.wait_for(
        ready_checker.wait_for_endpoint(
            stalled_request, test_input, session=None, timeout_seconds=1
        ),
        timeout=5,
    )
    assert not output.success
    assert output.error == "Endpoint readiness timed out after 1s."
    assert cancelled.is_set()


async def test_readiness_timeout_preserves_last_failure(test_input):
    failure = RequestFuncOutput(success=False, error="Model is still loading")
    calls = 0

    async def request(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return failure
        await asyncio.Event().wait()

    output = await asyncio.wait_for(
        ready_checker.wait_for_endpoint(
            request, test_input, session=None, timeout_seconds=1, retry_interval=0
        ),
        timeout=5,
    )
    assert output is failure
    assert output.error == "Model is still loading"
    assert calls == 2


@pytest.mark.parametrize("failures", [0, 1])
async def test_readiness_returns_success(test_input, failures):
    success = RequestFuncOutput(success=True)
    failure = RequestFuncOutput(success=False, error="Model is still loading")
    request = AsyncMock(side_effect=[failure] * failures + [success])

    output = await ready_checker.wait_for_endpoint(
        request, test_input, session=None, timeout_seconds=1, retry_interval=0
    )

    assert output is success
    assert request.await_count == failures + 1


async def test_readiness_does_not_sleep_after_deadline(test_input, monkeypatch):
    failure = RequestFuncOutput(success=False, error="Model is still loading")
    clock = Mock(return_value=0.0)
    sleep = AsyncMock()

    async def request(**kwargs):
        clock.return_value = 1.0
        return failure

    monkeypatch.setattr(ready_checker.time, "perf_counter", clock)
    monkeypatch.setattr(ready_checker.asyncio, "sleep", sleep)
    output = await ready_checker.wait_for_endpoint(
        request, test_input, session=None, timeout_seconds=1, retry_interval=5
    )

    assert output is failure
    sleep.assert_not_awaited()


async def test_readiness_propagates_external_cancellation(test_input):
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def request(**kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    task = asyncio.create_task(
        ready_checker.wait_for_endpoint(
            request, test_input, session=None, timeout_seconds=60
        )
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
    finally:
        task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)
    assert cancelled.is_set()
