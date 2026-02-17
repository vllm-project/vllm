# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests for shutdown timeout and signal handling."""

import asyncio
import signal
import time
from dataclasses import dataclass, field

import httpx
import openai
import psutil
import pytest

from tests.utils import RemoteOpenAIServer

MODEL_NAME = "hmellor/tiny-random-LlamaForCausalLM"

_SHUTDOWN_DETECTION_TIMEOUT = 10
_CHILD_CLEANUP_TIMEOUT = 10


def _get_child_pids(parent_pid: int) -> list[int]:
    try:
        parent = psutil.Process(parent_pid)
        return [c.pid for c in parent.children(recursive=True)]
    except psutil.NoSuchProcess:
        return []


async def _assert_children_cleaned_up(
    child_pids: list[int],
    timeout: float = _CHILD_CLEANUP_TIMEOUT,
):
    """Wait for child processes to exit and fail if any remain."""
    if not child_pids:
        return

    deadline = time.time() + timeout
    while time.time() < deadline:
        still_alive = []
        for pid in child_pids:
            try:
                p = psutil.Process(pid)
                if p.is_running() and p.status() != psutil.STATUS_ZOMBIE:
                    still_alive.append(pid)
            except psutil.NoSuchProcess:
                pass
        if not still_alive:
            return
        await asyncio.sleep(0.5)

    pytest.fail(
        f"Child processes {still_alive} still alive after {timeout}s. "
        f"Process cleanup may not be working correctly."
    )


@dataclass
class ShutdownState:
    got_503: bool = False
    got_500: bool = False
    requests_after_sigterm: int = 0
    aborted_requests: int = 0
    connection_errors: int = 0
    stop_requesting: bool = False
    errors: list[str] = field(default_factory=list)


async def _concurrent_request_loop(
    client: openai.AsyncOpenAI,
    state: ShutdownState,
    sigterm_sent: asyncio.Event | None = None,
    concurrency: int = 10,
):
    """Run multiple concurrent requests to keep the server busy."""

    async def single_request():
        while not state.stop_requesting:
            try:
                response = await client.completions.create(
                    model=MODEL_NAME,
                    prompt="Write a story: ",
                    max_tokens=200,
                )
                if sigterm_sent is not None and sigterm_sent.is_set():
                    state.requests_after_sigterm += 1
                # Check if any choice has finish_reason='abort'
                if any(choice.finish_reason == "abort" for choice in response.choices):
                    state.aborted_requests += 1
            except openai.APIStatusError as e:
                if e.status_code == 503:
                    state.got_503 = True
                elif e.status_code == 500:
                    state.got_500 = True
                else:
                    state.errors.append(f"API error: {e}")
            except (openai.APIConnectionError, httpx.RemoteProtocolError):
                state.connection_errors += 1
                if sigterm_sent is not None and sigterm_sent.is_set():
                    break
            except Exception as e:
                state.errors.append(f"Unexpected error: {e}")
                break
            await asyncio.sleep(0.01)

    tasks = [asyncio.create_task(single_request()) for _ in range(concurrency)]
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()


@pytest.mark.asyncio
async def test_wait_timeout_completes_requests():
    """Verify wait timeout: new requests rejected, in-flight requests complete."""
    server_args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "256",
        "--enforce-eager",
        "--gpu-memory-utilization",
        "0.05",
        "--max-num-seqs",
        "4",
        "--shutdown-timeout",
        "30",
    ]

    with RemoteOpenAIServer(MODEL_NAME, server_args) as remote_server:
        client = remote_server.get_async_client()
        proc = remote_server.proc
        child_pids = _get_child_pids(proc.pid)

        state = ShutdownState()
        sigterm_sent = asyncio.Event()

        request_task = asyncio.create_task(
            _concurrent_request_loop(client, state, sigterm_sent, concurrency=10)
        )

        await asyncio.sleep(0.5)
        proc.send_signal(signal.SIGTERM)
        sigterm_sent.set()

        try:
            await asyncio.wait_for(request_task, timeout=_SHUTDOWN_DETECTION_TIMEOUT)
        except asyncio.TimeoutError:
            pass
        finally:
            state.stop_requesting = True
            if not request_task.done():
                request_task.cancel()
            await asyncio.gather(request_task, return_exceptions=True)

        # wait timeout should complete in-flight requests
        assert state.requests_after_sigterm > 0, (
            f"Wait timeout should complete in-flight requests. "
            f"503: {state.got_503}, 500: {state.got_500}, "
            f"conn_errors: {state.connection_errors}, errors: {state.errors}"
        )
        # server must stop accepting new requests (503, 500, or connection close)
        assert state.got_503 or state.got_500 or state.connection_errors > 0, (
            f"Server should stop accepting requests. "
            f"completed: {state.requests_after_sigterm}, errors: {state.errors}"
        )

        await _assert_children_cleaned_up(child_pids)


@pytest.mark.asyncio
@pytest.mark.parametrize("wait_for_engine_idle", [0.0, 2.0])
async def test_abort_timeout_exits_quickly(wait_for_engine_idle: float):
    server_args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "256",
        "--enforce-eager",
        "--gpu-memory-utilization",
        "0.05",
        "--max-num-seqs",
        "4",
        "--shutdown-timeout",
        "0",
    ]

    with RemoteOpenAIServer(MODEL_NAME, server_args) as remote_server:
        proc = remote_server.proc
        child_pids = _get_child_pids(proc.pid)

        if wait_for_engine_idle > 0:
            client = remote_server.get_async_client()
            # Send requests to ensure engine is fully initialized
            for _ in range(2):
                await client.completions.create(
                    model=MODEL_NAME,
                    prompt="Test request: ",
                    max_tokens=10,
                )
            # Wait for engine to become idle
            await asyncio.sleep(wait_for_engine_idle)

        start_time = time.time()
        proc.send_signal(signal.SIGTERM)

        # abort timeout (0) should exit promptly
        for _ in range(20):
            if proc.poll() is not None:
                break
            time.sleep(0.1)

        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
            pytest.fail("Process did not exit after SIGTERM with abort timeout")

        exit_time = time.time() - start_time
        assert exit_time < 2, f"Default shutdown took too long: {exit_time:.1f}s"
        assert proc.returncode in (0, -15, None), f"Unexpected: {proc.returncode}"

        await _assert_children_cleaned_up(child_pids)


@pytest.mark.asyncio
async def test_wait_timeout_with_short_duration():
    """Verify server exits cleanly with a short wait timeout."""
    wait_timeout = 3
    server_args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "256",
        "--enforce-eager",
        "--gpu-memory-utilization",
        "0.05",
        "--max-num-seqs",
        "4",
        "--shutdown-timeout",
        str(wait_timeout),
    ]

    with RemoteOpenAIServer(MODEL_NAME, server_args) as remote_server:
        client = remote_server.get_async_client()
        proc = remote_server.proc
        child_pids = _get_child_pids(proc.pid)

        state = ShutdownState()
        request_task = asyncio.create_task(
            _concurrent_request_loop(client, state, concurrency=3)
        )

        await asyncio.sleep(0.5)

        start_time = time.time()
        proc.send_signal(signal.SIGTERM)

        # server should exit within wait_timeout + buffer
        max_wait = wait_timeout + 15
        for _ in range(int(max_wait * 10)):
            if proc.poll() is not None:
                break
            time.sleep(0.1)

        exit_time = time.time() - start_time

        state.stop_requesting = True
        if not request_task.done():
            request_task.cancel()
        await asyncio.gather(request_task, return_exceptions=True)

        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
            pytest.fail(f"Process did not exit within {max_wait}s after SIGTERM")

        assert exit_time < wait_timeout + 10, (
            f"Took too long to exit ({exit_time:.1f}s), expected <{wait_timeout + 10}s"
        )
        assert proc.returncode in (0, -15, None), f"Unexpected: {proc.returncode}"

        await _assert_children_cleaned_up(child_pids)


@pytest.mark.asyncio
async def test_abort_timeout_fails_inflight_requests():
    """Verify abort timeout (0) immediately aborts in-flight requests."""
    server_args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "256",
        "--enforce-eager",
        "--gpu-memory-utilization",
        "0.05",
        "--max-num-seqs",
        "4",
        "--shutdown-timeout",
        "0",
    ]

    with RemoteOpenAIServer(MODEL_NAME, server_args) as remote_server:
        client = remote_server.get_async_client()
        proc = remote_server.proc
        child_pids = _get_child_pids(proc.pid)

        state = ShutdownState()
        sigterm_sent = asyncio.Event()

        request_task = asyncio.create_task(
            _concurrent_request_loop(client, state, sigterm_sent, concurrency=10)
        )

        await asyncio.sleep(0.5)

        proc.send_signal(signal.SIGTERM)
        sigterm_sent.set()

        try:
            await asyncio.wait_for(request_task, timeout=5)
        except asyncio.TimeoutError:
            pass
        finally:
            state.stop_requesting = True
            if not request_task.done():
                request_task.cancel()
            await asyncio.gather(request_task, return_exceptions=True)

        # With abort timeout (0), requests should be aborted (finish_reason='abort')
        # or rejected (connection errors or API errors)
        assert (
            state.aborted_requests > 0
            or state.connection_errors > 0
            or state.got_500
            or state.got_503
        ), (
            f"Abort timeout should cause request aborts or failures. "
            f"aborted: {state.aborted_requests}, "
            f"503: {state.got_503}, 500: {state.got_500}, "
            f"conn_errors: {state.connection_errors}, "
            f"completed: {state.requests_after_sigterm}"
        )

        # Verify fast shutdown
        start_time = time.time()
        for _ in range(100):
            if proc.poll() is not None:
                break
            time.sleep(0.1)

        exit_time = time.time() - start_time
        assert exit_time < 10, f"Abort timeout shutdown took too long: {exit_time:.1f}s"

        await _assert_children_cleaned_up(child_pids)


@pytest.mark.asyncio
async def test_request_rejection_during_shutdown():
    """Verify new requests are rejected with error during shutdown."""
    server_args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "256",
        "--enforce-eager",
        "--gpu-memory-utilization",
        "0.05",
        "--max-num-seqs",
        "4",
        "--shutdown-timeout",
        "30",
    ]

    with RemoteOpenAIServer(MODEL_NAME, server_args) as remote_server:
        client = remote_server.get_async_client()
        proc = remote_server.proc
        child_pids = _get_child_pids(proc.pid)

        proc.send_signal(signal.SIGTERM)

        await asyncio.sleep(1.0)

        # Try to send new requests - they should be rejected
        rejected_count = 0
        for _ in range(10):
            try:
                await client.completions.create(
                    model=MODEL_NAME, prompt="Hello", max_tokens=10
                )
            except (
                openai.APIStatusError,
                openai.APIConnectionError,
                httpx.RemoteProtocolError,
            ):
                rejected_count += 1
            await asyncio.sleep(0.1)

        assert rejected_count > 0, (
            f"Expected requests to be rejected during shutdown, "
            f"but {rejected_count} were rejected out of 10"
        )

        await _assert_children_cleaned_up(child_pids)


@pytest.mark.asyncio
async def test_multi_api_server_shutdown():
    """Verify shutdown works with multiple API servers."""
    server_args = [
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "256",
        "--enforce-eager",
        "--gpu-memory-utilization",
        "0.05",
        "--max-num-seqs",
        "4",
        "--shutdown-timeout",
        "30",
        "--api-server-count",
        "2",
    ]

    with RemoteOpenAIServer(MODEL_NAME, server_args, auto_port=True) as remote_server:
        client = remote_server.get_async_client()
        proc = remote_server.proc
        child_pids = _get_child_pids(proc.pid)

        assert len(child_pids) >= 2, (
            f"Expected at least 2 child processes, got {len(child_pids)}"
        )

        state = ShutdownState()
        sigterm_sent = asyncio.Event()

        # Start concurrent requests across both API servers
        request_task = asyncio.create_task(
            _concurrent_request_loop(client, state, sigterm_sent, concurrency=8)
        )

        await asyncio.sleep(0.5)

        # Send SIGTERM to parent - should propagate to all children
        proc.send_signal(signal.SIGTERM)
        sigterm_sent.set()

        try:
            await asyncio.wait_for(request_task, timeout=_SHUTDOWN_DETECTION_TIMEOUT)
        except asyncio.TimeoutError:
            pass
        finally:
            state.stop_requesting = True
            if not request_task.done():
                request_task.cancel()
            await asyncio.gather(request_task, return_exceptions=True)

        for _ in range(300):  # up to 30 seconds
            if proc.poll() is not None:
                break
            time.sleep(0.1)

        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
            pytest.fail("Process did not exit after SIGTERM")

        await _assert_children_cleaned_up(child_pids)
