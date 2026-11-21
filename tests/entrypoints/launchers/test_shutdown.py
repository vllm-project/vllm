# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests for shutdown behavior, timeout, and signal handling."""

import asyncio
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import openai
import psutil
import pytest
from prometheus_client.parser import text_string_to_metric_families

from tests.utils import RemoteOpenAIServer
from vllm.utils.network_utils import get_open_port

MODEL_NAME = "vllm-project/tiny-random-LlamaForCausalLM"

# GPU initialization might take take longer
_SERVER_STARTUP_TIMEOUT = 120
_PROCESS_EXIT_TIMEOUT = 15
_SHUTDOWN_DETECTION_TIMEOUT = 10
_CHILD_CLEANUP_TIMEOUT = 10
_INFLIGHT_REQUEST_START_TIMEOUT = 5
_INFLIGHT_REQUEST_POLL_INTERVAL = 0.1
_ABORT_CLIENT_TIMEOUT = 3


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


def _wait_for_server_exit(proc: subprocess.Popen, timeout: float) -> None:
    """Children such as multiprocessing's resource_tracker only exit after the
    server process does, so wait for it before checking them."""
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pytest.fail(f"Server did not exit within {timeout}s")


@dataclass(slots=True)
class ShutdownState:
    got_503: bool = False
    got_500: bool = False
    requests_after_sigterm: int = 0
    aborted_requests: int = 0
    connection_errors: int = 0
    inflight_requests: int = 0
    stop_requesting: bool = False
    errors: list[str] = field(default_factory=list)


async def _concurrent_request_loop(
    client: openai.AsyncOpenAI,
    state: ShutdownState,
    sigterm_sent: asyncio.Event | None = None,
    concurrency: int = 10,
    *,
    pooling: bool = False,
    stop_on_sigterm: bool = False,
):
    """Run multiple concurrent requests to keep the server busy."""

    async def single_request():
        while not state.stop_requesting and not (
            stop_on_sigterm and sigterm_sent is not None and sigterm_sent.is_set()
        ):
            try:
                state.inflight_requests += 1
                if pooling:
                    await client.embeddings.create(
                        model=MODEL_NAME,
                        input="hello " * 1500,
                    )
                else:
                    response = await client.completions.create(
                        model=MODEL_NAME,
                        prompt="Write a story: ",
                        max_tokens=200,
                    )
                    # Check if any choice has finish_reason='abort'
                    if any(
                        choice.finish_reason == "abort" for choice in response.choices
                    ):
                        state.aborted_requests += 1
                if sigterm_sent is not None and sigterm_sent.is_set():
                    state.requests_after_sigterm += 1
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
            finally:
                state.inflight_requests -= 1
            await asyncio.sleep(0.01)

    tasks = [asyncio.create_task(single_request()) for _ in range(concurrency)]
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()


def _metric_total(metrics_text: str, name: str) -> float:
    """Sum one Prometheus gauge across the server's label sets."""
    total = 0.0
    for family in text_string_to_metric_families(metrics_text):
        if family.name == name:
            for sample in family.samples:
                if sample.name == name:
                    total += sample.value
    return total


def _num_running_requests(metrics_text: str) -> float:
    """Sum `vllm:num_requests_running` across the server's label sets."""
    return _metric_total(metrics_text, "vllm:num_requests_running")


def _num_admitted_requests(metrics_text: str) -> float:
    """Requests the scheduler already has: running plus waiting."""
    return _num_running_requests(metrics_text) + _metric_total(
        metrics_text, "vllm:num_requests_waiting"
    )


async def _wait_for_metric(
    metrics_url: str,
    predicate,
    timeout: float,
    failure: str,
) -> float:
    """Poll server metrics until `predicate` accepts the text."""
    start = time.monotonic()
    deadline = start + timeout
    async with httpx.AsyncClient() as metrics_client:
        while time.monotonic() < deadline:
            response = await metrics_client.get(metrics_url)
            response.raise_for_status()
            if predicate(response.text):
                return time.monotonic() - start
            await asyncio.sleep(_INFLIGHT_REQUEST_POLL_INTERVAL)
    raise AssertionError(failure)


async def _wait_for_running_request(
    metrics_url: str,
    timeout: float = _INFLIGHT_REQUEST_START_TIMEOUT,
) -> float:
    """Wait until the server reports a running request, returning the delay.

    `ShutdownState.inflight_requests` is incremented before the HTTP call is
    awaited, so it only shows that a client coroutine started. Draining can
    only complete work the engine has already admitted, so synchronize on the
    server's own scheduler gauge instead.
    """
    return await _wait_for_metric(
        metrics_url,
        lambda text: _num_running_requests(text) > 0,
        timeout,
        f"No request reached the running state within {timeout}s",
    )


@pytest.mark.asyncio
async def test_shutdown_on_engine_failure(tmp_path: Path):
    """Verify that API returns connection error when server process is killed.

    Starts a vLLM server, kills it to simulate a crash, then verifies that
    subsequent API calls fail appropriately.
    """
    port = get_open_port()

    # Redirect server output to a file rather than a pipe: nothing drains the
    # pipes while we poll for startup, so a server that writes more than the
    # pipe buffer holds would block on write and never become ready.
    server_log = tmp_path / "server.log"
    with server_log.open("wb") as log_file:
        proc = subprocess.Popen(
            [
                # dtype, max-len etc set so that this can run in CI
                sys.executable,
                "-m",
                "vllm.entrypoints.launchers.api_server.entry",
                "--model",
                MODEL_NAME,
                "--dtype",
                "bfloat16",
                "--max-model-len",
                "128",
                "--enforce-eager",
                "--port",
                str(port),
                "--gpu-memory-utilization",
                "0.05",
                "--max-num-seqs",
                "2",
            ],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN),
        )

    # Wait for server startup
    start_time = time.time()
    client = openai.AsyncOpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key="dummy",
        max_retries=0,
        timeout=10,
    )

    # Poll until server is ready
    while time.time() - start_time < _SERVER_STARTUP_TIMEOUT:
        try:
            await client.completions.create(
                model=MODEL_NAME, prompt="Hello", max_tokens=1
            )
            break
        except Exception:
            time.sleep(0.5)
            if proc.poll() is not None:
                pytest.fail(
                    f"Server died during startup: {proc.returncode}\n"
                    f"{server_log.read_text(errors='replace')}"
                )
    else:
        proc.terminate()
        proc.wait(timeout=_PROCESS_EXIT_TIMEOUT)
        pytest.fail(
            f"Server failed to start in {_SERVER_STARTUP_TIMEOUT} seconds\n"
            f"{server_log.read_text(errors='replace')}"
        )

    # Kill server to simulate crash
    proc.terminate()
    time.sleep(1)

    # Verify API calls now fail
    with pytest.raises((openai.APIConnectionError, openai.APIStatusError)):
        await client.completions.create(
            model=MODEL_NAME, prompt="This should fail", max_tokens=1
        )

    return_code = proc.wait(timeout=_PROCESS_EXIT_TIMEOUT)
    assert return_code is not None


@pytest.mark.asyncio
async def test_wait_timeout_completes_requests():
    """Verify wait timeout: new requests rejected, in-flight requests complete."""
    shutdown_timeout = 30
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
        str(shutdown_timeout),
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

        try:
            # Drain can only complete requests the engine already admitted, so
            # wait for the server to report one instead of guessing with a sleep.
            admission_delay = await _wait_for_running_request(
                remote_server.url_for("metrics")
            )
            proc.send_signal(signal.SIGTERM)
            sigterm_sent.set()

            await asyncio.wait_for(request_task, timeout=_SHUTDOWN_DETECTION_TIMEOUT)
        except TimeoutError:
            pass
        finally:
            state.stop_requesting = True
            if not request_task.done():
                request_task.cancel()
            await asyncio.gather(request_task, return_exceptions=True)

        # wait timeout should complete in-flight requests
        assert state.requests_after_sigterm > 0, (
            f"Wait timeout should complete in-flight requests. "
            f"admission_delay: {admission_delay:.3f}s, "
            f"503: {state.got_503}, 500: {state.got_500}, "
            f"conn_errors: {state.connection_errors}, errors: {state.errors}"
        )
        # server must stop accepting new requests (503, 500, or connection close)
        assert state.got_503 or state.got_500 or state.connection_errors > 0, (
            f"Server should stop accepting requests. "
            f"completed: {state.requests_after_sigterm}, errors: {state.errors}"
        )

        _wait_for_server_exit(proc, shutdown_timeout)
        await _assert_children_cleaned_up(child_pids)


async def _pooling_request_rejected_with_503(
    client: openai.AsyncOpenAI,
    deadline_s: float,
) -> str | None:
    """Poll embeddings until shutdown rejects one.

    SIGTERM and EngineCore entering shutdown are not the same instant, so the
    first call after the signal can still be admitted. Returns None on 503,
    or the unexpected failure that ended the poll.
    """
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        try:
            await client.embeddings.create(model=MODEL_NAME, input="ping")
        except openai.APIStatusError as e:
            if e.status_code == 503:
                return None
            return f"HTTP {e.status_code}: {e}"
        except (openai.APIConnectionError, httpx.RemoteProtocolError) as e:
            return f"connection closed before 503: {e}"
        await asyncio.sleep(0.05)
    return "timed out before a new embeddings call returned 503"


# One embeddings call of this length finishes in well under a second on a
# fast GPU once the scheduler has it (see the 5090 log: SIGTERM, HTTP 200,
# and "request processing complete" in the same second). A single in-flight
# request is therefore gone before the post-SIGTERM probe is tokenized.
# Ten serial chunked prefills, each ~1400 tokens at 16 tokens per step,
# keep EngineCore inside the drain loop for that probe. Prefix caching is
# disabled and each prompt differs: identical text would hit the cache after
# the first request, and the drain would end before the probe.
_POOLING_DRAIN_REQUESTS = 10


def _pooling_prompt(i: int) -> str:
    return f"{i} " + ("hello " * 1400)


@pytest.mark.asyncio
async def test_wait_timeout_completes_pooling_requests():
    """In-flight /v1/embeddings finish during drain; a later one gets 503.

    Several long embeddings are admitted before SIGTERM, behind a one-sequence
    budget and a small token chunk, so the engine is still draining when the
    probe runs. Those calls return 200. The probe returns 503, not 500.
    """
    shutdown_timeout = 60
    server_args = [
        "--runner",
        "pooling",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--max-num-batched-tokens",
        "16",
        "--enforce-eager",
        "--gpu-memory-utilization",
        "0.05",
        "--max-num-seqs",
        "1",
        "--no-enable-prefix-caching",
        "--shutdown-timeout",
        str(shutdown_timeout),
    ]

    with RemoteOpenAIServer(
        MODEL_NAME,
        server_args,
        env_dict={"VLLM_USE_RUST_FRONTEND": "0"},
    ) as remote_server:
        # In-flight calls must be allowed to outlive the drain. The probe
        # client fails in 10s so a missed drain window cannot sit until the
        # default 600s client timeout.
        client = remote_server.get_async_client(timeout=shutdown_timeout + 30)
        probe = remote_server.get_async_client(timeout=10)
        proc = remote_server.proc
        child_pids = _get_child_pids(proc.pid)
        metrics_url = remote_server.url_for("metrics")

        inflight = [
            asyncio.create_task(
                client.embeddings.create(model=MODEL_NAME, input=_pooling_prompt(i))
            )
            for i in range(_POOLING_DRAIN_REQUESTS)
        ]
        try:
            # Drain only finishes work the scheduler already holds. Wait until
            # every call is running or waiting, not merely until the first
            # one has started (that one may already be on its last chunk).
            admission_delay = await _wait_for_metric(
                metrics_url,
                lambda text: _num_admitted_requests(text) >= _POOLING_DRAIN_REQUESTS,
                timeout=60,
                failure=(
                    "Pooling requests were not all admitted before SIGTERM "
                    f"(wanted {_POOLING_DRAIN_REQUESTS})"
                ),
            )
            proc.send_signal(signal.SIGTERM)
            rejection = await _pooling_request_rejected_with_503(
                probe, _SHUTDOWN_DETECTION_TIMEOUT
            )
            done, pending = await asyncio.wait(inflight, timeout=shutdown_timeout)
            assert not pending, (
                f"{len(pending)} in-flight embeddings did not finish during drain"
            )
            for task in done:
                result = task.result()
                assert result.data, (
                    "In-flight pooling request returned an empty embedding"
                )
            assert rejection is None, (
                "A new embeddings call after SIGTERM should return 503. "
                f"admission_delay: {admission_delay:.3f}s, failure: {rejection}"
            )
            _wait_for_server_exit(proc, shutdown_timeout)
            await _assert_children_cleaned_up(child_pids)
        finally:
            for task in inflight:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*inflight, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("wait_for_engine_idle", [0.0, 2.0])
async def test_abort_timeout_exits_within_cleanup_grace(
    wait_for_engine_idle: float,
):
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

        proc.send_signal(signal.SIGTERM)

        # A zero request timeout aborts requests immediately, but process
        # teardown may still use the platform-specific resource cleanup grace.
        termination_timeout = remote_server._get_process_termination_timeout()
        try:
            proc.wait(timeout=termination_timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
            pytest.fail(
                "Process did not exit within the resource cleanup grace "
                f"({termination_timeout}s)"
            )

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
        client = remote_server.get_async_client(timeout=_ABORT_CLIENT_TIMEOUT)
        proc = remote_server.proc
        child_pids = _get_child_pids(proc.pid)

        state = ShutdownState()
        sigterm_sent = asyncio.Event()

        request_task = asyncio.create_task(
            _concurrent_request_loop(client, state, sigterm_sent, concurrency=10)
        )

        deadline = time.time() + _INFLIGHT_REQUEST_START_TIMEOUT
        while state.inflight_requests == 0 and time.time() < deadline:
            await asyncio.sleep(_INFLIGHT_REQUEST_POLL_INTERVAL)
        assert state.inflight_requests > 0

        proc.send_signal(signal.SIGTERM)
        sigterm_sent.set()

        try:
            await asyncio.wait_for(request_task, timeout=5)
        except TimeoutError:
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
    shutdown_timeout = 30
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
        str(shutdown_timeout),
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

        _wait_for_server_exit(proc, shutdown_timeout)
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
        except TimeoutError:
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
