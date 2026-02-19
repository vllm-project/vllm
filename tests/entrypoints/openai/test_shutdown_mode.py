# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests for shutdown modes and signal handling."""

import asyncio
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field

import httpx
import openai
import pytest

from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port

MODEL_NAME = "hmellor/tiny-random-LlamaForCausalLM"

_IS_ROCM = current_platform.is_rocm()
_SERVER_STARTUP_TIMEOUT = 120
_PROCESS_EXIT_TIMEOUT = 30
_SHUTDOWN_DETECTION_TIMEOUT = 10
_CHILD_CLEANUP_TIMEOUT = 10

try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def _cleanup_orphaned_vllm_processes():
    """Kill any orphaned vLLM processes from previous test runs."""
    if not _HAS_PSUTIL:
        return
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            cmdline_str = " ".join(cmdline)
            # kill orphaned api_server or EngineCore processes using our test model
            if MODEL_NAME in cmdline_str or "VLLM::EngineCore" in proc.info.get(
                "name", ""
            ):
                proc.kill()
                proc.wait(timeout=5)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
            pass


@pytest.fixture(autouse=True)
def cleanup_before_test():
    """Ensure no orphaned processes from previous runs interfere."""
    _cleanup_orphaned_vllm_processes()
    yield
    # also cleanup after in case test fails mid-run
    _cleanup_orphaned_vllm_processes()


def _get_child_pids(parent_pid: int) -> list[int]:
    """Get all child process PIDs recursively."""
    if not _HAS_PSUTIL:
        return []
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
    if not _HAS_PSUTIL or not child_pids:
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
    connection_errors: int = 0
    stop_requesting: bool = False
    errors: list[str] = field(default_factory=list)


def _start_server(
    port: int,
    shutdown_mode: str = "abort",
    capture_output: bool = False,
    wait_timeout: int = 30,
    api_server_count: int = 1,
):
    args = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        MODEL_NAME,
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "256",
        "--enforce-eager",
        "--port",
        str(port),
        "--gpu-memory-utilization",
        "0.05",
        "--max-num-seqs",
        "4",
        "--disable-frontend-multiprocessing",
        "--shutdown-mode",
        shutdown_mode,
        "--shutdown-wait-timeout",
        str(wait_timeout),
    ]

    if api_server_count > 1:
        args.extend(["--api-server-count", str(api_server_count)])

    return subprocess.Popen(
        args,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.STDOUT if capture_output else None,
        start_new_session=True,
        preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN),
    )


async def _wait_for_server_ready(client: openai.AsyncOpenAI, timeout: float):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            await client.completions.create(
                model=MODEL_NAME, prompt="Hello", max_tokens=1
            )
            return True
        except Exception:
            await asyncio.sleep(0.5)
    return False


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
                await client.completions.create(
                    model=MODEL_NAME,
                    prompt="Write a story: ",
                    max_tokens=200,
                )
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
            await asyncio.sleep(0.01)

    tasks = [asyncio.create_task(single_request()) for _ in range(concurrency)]
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()


@pytest.mark.asyncio
async def test_wait_mode_completes_requests():
    """Verify wait mode: new requests rejected, in-flight requests complete."""
    port = get_open_port()
    proc = _start_server(port, shutdown_mode="wait")

    try:
        client = openai.AsyncOpenAI(
            base_url=f"http://localhost:{port}/v1",
            api_key="dummy",
            max_retries=0,
            timeout=30,
        )

        if not await _wait_for_server_ready(client, _SERVER_STARTUP_TIMEOUT):
            proc.terminate()
            proc.wait(timeout=_PROCESS_EXIT_TIMEOUT)
            pytest.fail(f"Server failed to start in {_SERVER_STARTUP_TIMEOUT}s")

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

        # wait mode must complete in-flight requests
        assert state.requests_after_sigterm > 0, (
            f"Wait mode should complete in-flight requests. "
            f"503: {state.got_503}, 500: {state.got_500}, "
            f"conn_errors: {state.connection_errors}, errors: {state.errors}"
        )
        # server must stop accepting new requests (503, 500, or connection close)
        assert state.got_503 or state.got_500 or state.connection_errors > 0, (
            f"Server should stop accepting requests. "
            f"completed: {state.requests_after_sigterm}, errors: {state.errors}"
        )

        await _assert_children_cleaned_up(child_pids)

    finally:
        if proc.poll() is None:
            proc.terminate()
        return_code = proc.wait(timeout=_PROCESS_EXIT_TIMEOUT)
        assert return_code in (0, -15, None), f"Unexpected return code: {return_code}"


@pytest.mark.asyncio
async def test_abort_mode_exits_quickly():
    """Verify default (abort) shutdown mode exits promptly on SIGTERM."""
    port = get_open_port()
    # don't specify shutdown_mode to test the default behavior
    proc = _start_server(port)

    try:
        client = openai.AsyncOpenAI(
            base_url=f"http://localhost:{port}/v1",
            api_key="dummy",
            max_retries=0,
            timeout=30,
        )

        if not await _wait_for_server_ready(client, _SERVER_STARTUP_TIMEOUT):
            proc.terminate()
            proc.wait(timeout=_PROCESS_EXIT_TIMEOUT)
            pytest.fail(f"Server failed to start in {_SERVER_STARTUP_TIMEOUT}s")

        child_pids = _get_child_pids(proc.pid)

        start_time = time.time()
        proc.send_signal(signal.SIGTERM)

        # default mode should exit promptly
        for _ in range(100):
            if proc.poll() is not None:
                break
            time.sleep(0.1)

        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
            pytest.fail("Process did not exit after SIGTERM in default mode")

        exit_time = time.time() - start_time
        assert exit_time < 10, f"Default shutdown took too long: {exit_time:.1f}s"
        assert proc.returncode in (0, -15, None), f"Unexpected: {proc.returncode}"

        await _assert_children_cleaned_up(child_pids)

    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


@pytest.mark.asyncio
async def test_wait_mode_with_short_timeout():
    """Verify server exits cleanly with a short wait timeout."""
    port = get_open_port()
    # use a short wait timeout
    wait_timeout = 3
    proc = _start_server(port, shutdown_mode="wait", wait_timeout=wait_timeout)

    try:
        client = openai.AsyncOpenAI(
            base_url=f"http://localhost:{port}/v1",
            api_key="dummy",
            max_retries=0,
            timeout=60,
        )

        if not await _wait_for_server_ready(client, _SERVER_STARTUP_TIMEOUT):
            proc.terminate()
            proc.wait(timeout=_PROCESS_EXIT_TIMEOUT)
            pytest.fail(f"Server failed to start in {_SERVER_STARTUP_TIMEOUT}s")

        child_pids = _get_child_pids(proc.pid)

        # start some requests (they may or may not complete before timeout)
        state = ShutdownState()
        request_task = asyncio.create_task(
            _concurrent_request_loop(client, state, concurrency=3)
        )

        await asyncio.sleep(0.5)

        # send SIGTERM
        start_time = time.time()
        proc.send_signal(signal.SIGTERM)

        # server should exit within wait_timeout + buffer
        max_wait = wait_timeout + 15
        for _ in range(int(max_wait * 10)):
            if proc.poll() is not None:
                break
            time.sleep(0.1)

        exit_time = time.time() - start_time

        # cleanup request task
        state.stop_requesting = True
        if not request_task.done():
            request_task.cancel()
        await asyncio.gather(request_task, return_exceptions=True)

        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
            pytest.fail(f"Process did not exit within {max_wait}s after SIGTERM")

        # server should exit within reasonable time (wait_timeout + overhead)
        assert exit_time < wait_timeout + 10, (
            f"Took too long to exit ({exit_time:.1f}s), expected <{wait_timeout + 10}s"
        )
        assert proc.returncode in (0, -15, None), f"Unexpected: {proc.returncode}"

        await _assert_children_cleaned_up(child_pids)

    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


@pytest.mark.asyncio
async def test_abort_mode_fails_inflight_requests():
    """Verify abort mode immediately aborts in-flight requests."""
    port = get_open_port()
    proc = _start_server(port, shutdown_mode="abort")

    try:
        client = openai.AsyncOpenAI(
            base_url=f"http://localhost:{port}/v1",
            api_key="dummy",
            max_retries=0,
            timeout=30,
        )

        if not await _wait_for_server_ready(client, _SERVER_STARTUP_TIMEOUT):
            proc.terminate()
            proc.wait(timeout=_PROCESS_EXIT_TIMEOUT)
            pytest.fail(f"Server failed to start in {_SERVER_STARTUP_TIMEOUT}s")

        child_pids = _get_child_pids(proc.pid)

        state = ShutdownState()
        sigterm_sent = asyncio.Event()

        request_task = asyncio.create_task(
            _concurrent_request_loop(client, state, sigterm_sent, concurrency=10)
        )

        # Let requests start
        await asyncio.sleep(0.5)

        # Send SIGTERM - should abort in-flight requests
        proc.send_signal(signal.SIGTERM)
        sigterm_sent.set()

        # Give time for shutdown to process
        try:
            await asyncio.wait_for(request_task, timeout=5)
        except asyncio.TimeoutError:
            pass
        finally:
            state.stop_requesting = True
            if not request_task.done():
                request_task.cancel()
            await asyncio.gather(request_task, return_exceptions=True)

        # In abort mode, requests should fail (connection errors or API errors)
        assert state.connection_errors > 0 or state.got_500 or state.got_503, (
            f"Abort mode should cause request failures. "
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
        assert exit_time < 10, f"Abort mode shutdown took too long: {exit_time:.1f}s"

        await _assert_children_cleaned_up(child_pids)

    finally:
        if proc.poll() is None:
            proc.terminate()
        proc.wait(timeout=_PROCESS_EXIT_TIMEOUT)


@pytest.mark.asyncio
async def test_request_rejection_during_shutdown():
    """Verify new requests are rejected with error during shutdown."""
    port = get_open_port()
    proc = _start_server(port, shutdown_mode="wait", wait_timeout=30)

    try:
        client = openai.AsyncOpenAI(
            base_url=f"http://localhost:{port}/v1",
            api_key="dummy",
            max_retries=0,
            timeout=10,
        )

        if not await _wait_for_server_ready(client, _SERVER_STARTUP_TIMEOUT):
            proc.terminate()
            proc.wait(timeout=_PROCESS_EXIT_TIMEOUT)
            pytest.fail(f"Server failed to start in {_SERVER_STARTUP_TIMEOUT}s")

        child_pids = _get_child_pids(proc.pid)

        # Send SIGTERM to initiate shutdown
        proc.send_signal(signal.SIGTERM)

        # Give server a moment to start shutdown
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

        # At least some requests should be rejected
        assert rejected_count > 0, (
            f"Expected requests to be rejected during shutdown, "
            f"but {rejected_count} were rejected out of 10"
        )

        await _assert_children_cleaned_up(child_pids)

    finally:
        if proc.poll() is None:
            proc.terminate()
        proc.wait(timeout=_PROCESS_EXIT_TIMEOUT)


@pytest.mark.asyncio
async def test_multi_api_server_shutdown():
    """Verify shutdown works with multiple API servers."""
    pytest.importorskip("psutil")

    port = get_open_port()
    proc = _start_server(port, shutdown_mode="wait", api_server_count=2)

    try:
        client = openai.AsyncOpenAI(
            base_url=f"http://localhost:{port}/v1",
            api_key="dummy",
            max_retries=0,
            timeout=30,
        )

        if not await _wait_for_server_ready(client, _SERVER_STARTUP_TIMEOUT):
            proc.terminate()
            proc.wait(timeout=_PROCESS_EXIT_TIMEOUT)
            pytest.fail(f"Server failed to start in {_SERVER_STARTUP_TIMEOUT}s")

        child_pids = _get_child_pids(proc.pid)
        # Should have multiple API server processes plus engine cores
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

        # Wait for clean exit
        for _ in range(300):  # up to 30 seconds
            if proc.poll() is not None:
                break
            time.sleep(0.1)

        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
            pytest.fail("Process did not exit after SIGTERM")

        # Verify all children terminated
        await _assert_children_cleaned_up(child_pids)

    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
