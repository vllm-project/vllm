# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import gc
from unittest.mock import AsyncMock, Mock

import aiohttp
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


@pytest.mark.parametrize("prior_failure", [False, True])
async def test_readiness_timeout_cancels_request_and_preserves_failure(
    test_input, prior_failure
):
    failure = RequestFuncOutput(success=False, error="Model is still loading")
    cancelled = asyncio.Event()
    calls = 0

    async def request(**kwargs):
        nonlocal calls
        calls += 1
        if prior_failure and calls == 1:
            return failure
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    output = await asyncio.wait_for(
        ready_checker.wait_for_endpoint(
            request, test_input, session=None, timeout_seconds=1, retry_interval=0
        ),
        timeout=5,
    )
    assert not output.success
    await asyncio.wait_for(cancelled.wait(), timeout=5)
    assert calls == 1 + prior_failure
    if prior_failure:
        assert output is failure
        assert output.error == "Model is still loading"
    else:
        assert output.error == "Endpoint readiness timed out after 1s."


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


@pytest.mark.parametrize("failure_kind", ["empty", "connector", "timeout", "zero"])
async def test_readiness_always_reports_timeout(test_input, monkeypatch, failure_kind):
    clock = Mock(return_value=0.0)

    async def request(**kwargs):
        clock.return_value = 1.0
        if failure_kind == "connector":
            raise aiohttp.ClientConnectorError(Mock(), OSError("Connection refused"))
        if failure_kind == "timeout":
            raise asyncio.TimeoutError
        return RequestFuncOutput(success=False)

    monkeypatch.setattr(ready_checker.time, "perf_counter", clock)
    timeout = 0 if failure_kind == "zero" else 1
    output = await ready_checker.wait_for_endpoint(
        request, test_input, session=None, timeout_seconds=timeout, retry_interval=0
    )

    assert not output.success
    assert output.error == f"Endpoint readiness timed out after {timeout}s."


@pytest.mark.parametrize("external_cancel", [False, True])
@pytest.mark.parametrize("cleanup_raises", [False, True])
async def test_readiness_does_not_wait_for_probe_cleanup(
    test_input, external_cancel, cleanup_raises
):
    started = asyncio.Event()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    cleanup_finished = asyncio.Event()
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    unhandled = Mock()
    loop.set_exception_handler(unhandled)

    async def request(**kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleanup_started.set()
            await release_cleanup.wait()
            cleanup_finished.set()
            if cleanup_raises:
                raise RuntimeError("Probe cleanup failed") from None
            return RequestFuncOutput(success=True)

    task = asyncio.create_task(
        ready_checker.wait_for_endpoint(
            request,
            test_input,
            session=None,
            timeout_seconds=60 if external_cancel else 1,
        )
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=5)
        if external_cancel:
            task.cancel()
        done, _ = await asyncio.wait({task}, timeout=3)
        assert task in done, "Readiness waited for probe cancellation cleanup"
        if external_cancel:
            with pytest.raises(asyncio.CancelledError):
                task.result()
        else:
            output = task.result()
            assert not output.success
            assert output.error == "Endpoint readiness timed out after 1s."
        await asyncio.wait_for(cleanup_started.wait(), timeout=5)
        assert not cleanup_finished.is_set()
    finally:
        release_cleanup.set()
        try:
            await asyncio.wait_for(cleanup_finished.wait(), timeout=5)
            await asyncio.gather(task, return_exceptions=True)
            # Let completion callbacks run, then expose any unretrieved exception.
            await asyncio.sleep(0)
            gc.collect()
            unhandled.assert_not_called()
        finally:
            loop.set_exception_handler(previous_handler)
