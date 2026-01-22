# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests for graceful shutdown and signal handling."""

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
_DRAIN_DETECTION_TIMEOUT = 10


@dataclass
class DrainState:
    got_503: bool = False
    requests_after_sigterm: int = 0
    connection_errors: int = 0
    stop_requesting: bool = False
    errors: list[str] = field(default_factory=list)


def _start_server(
    port: int,
    enable_graceful_shutdown: bool = False,
    capture_output: bool = False,
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
    ]

    if enable_graceful_shutdown:
        args.extend(["--enable-graceful-shutdown", "--drain-timeout", "30"])

    return subprocess.Popen(
        args,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.STDOUT if capture_output else None,
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
    state: DrainState,
    sigterm_sent: asyncio.Event | None = None,
    concurrency: int = 10,
):
    """Run multiple concurrent requests to keep the server busy."""

    async def single_request():
        while not state.stop_requesting:
            try:
                await client.completions.create(
                    model=MODEL_NAME,
                    prompt="Write a long detailed story about a robot: ",
                    max_tokens=500,
                )
                if sigterm_sent is not None and sigterm_sent.is_set():
                    state.requests_after_sigterm += 1
            except openai.APIStatusError as e:
                if e.status_code == 503:
                    state.got_503 = True
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
async def test_graceful_shutdown_drains_requests():
    """Verify graceful shutdown: 503s returned, in-flight requests complete."""
    port = get_open_port()
    proc = _start_server(port, enable_graceful_shutdown=True)

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

        state = DrainState()
        sigterm_sent = asyncio.Event()

        request_task = asyncio.create_task(
            _concurrent_request_loop(client, state, sigterm_sent, concurrency=10)
        )

        await asyncio.sleep(0.5)
        proc.send_signal(signal.SIGTERM)
        sigterm_sent.set()

        try:
            await asyncio.wait_for(request_task, timeout=_DRAIN_DETECTION_TIMEOUT)
        except asyncio.TimeoutError:
            pass
        finally:
            state.stop_requesting = True
            if not request_task.done():
                request_task.cancel()
            await asyncio.gather(request_task, return_exceptions=True)

        assert (
            state.got_503
            or state.requests_after_sigterm > 0
            or state.connection_errors > 0
        ), (
            f"Expected 503, completed requests, or connection close after SIGTERM. "
            f"503: {state.got_503}, completed: {state.requests_after_sigterm}, "
            f"conn_errors: {state.connection_errors}, errors: {state.errors}"
        )

    finally:
        if proc.poll() is None:
            proc.terminate()
        return_code = proc.wait(timeout=_PROCESS_EXIT_TIMEOUT)
        assert return_code in (0, -15, None), f"Unexpected return code: {return_code}"


@pytest.mark.asyncio
async def test_immediate_shutdown_on_second_signal():
    """Verify that sending two signals triggers immediate shutdown."""
    port = get_open_port()
    proc = _start_server(port, enable_graceful_shutdown=True, capture_output=True)

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

        state = DrainState()

        request_task = asyncio.create_task(
            _concurrent_request_loop(client, state, concurrency=20)
        )

        await asyncio.sleep(1.0)
        proc.send_signal(signal.SIGTERM)
        await asyncio.sleep(0.3)
        proc.send_signal(signal.SIGTERM)

        state.stop_requesting = True
        request_task.cancel()
        await asyncio.gather(request_task, return_exceptions=True)

        try:
            stdout, _ = await asyncio.wait_for(
                asyncio.to_thread(proc.communicate),
                timeout=_PROCESS_EXIT_TIMEOUT,
            )
            output = stdout.decode() if stdout else ""
        except asyncio.TimeoutError:
            proc.kill()
            stdout, _ = proc.communicate()
            output = stdout.decode() if stdout else ""
            pytest.fail(
                f"Process did not exit after second signal. Output:\n{output[-3000:]}"
            )

        assert "Received second signal, forcing immediate shutdown" in output, (
            f"Immediate shutdown log message not found. Output:\n{output[-3000:]}"
        )
        assert proc.returncode in (0, -15, None), f"Unexpected: {proc.returncode}"

    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
