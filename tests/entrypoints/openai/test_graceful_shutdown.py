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
import regex as re

from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port

MODEL_NAME = "hmellor/tiny-random-LlamaForCausalLM"

_IS_ROCM = current_platform.is_rocm()
_SERVER_STARTUP_TIMEOUT = 120
_PROCESS_EXIT_TIMEOUT = 30
_DRAIN_DETECTION_TIMEOUT = 10


@dataclass
class DrainState:
    draining_detected: bool = False
    got_503: bool = False
    metric_value: float | None = None
    requests_completed_after_sigterm: int = 0
    connection_errors: int = 0
    stop_requesting: bool = False
    errors: list[str] = field(default_factory=list)


def _parse_draining_metric(metrics_text: str) -> float | None:
    for line in metrics_text.split("\n"):
        if line.startswith("vllm:server_draining{"):
            match = re.search(r"}\s+(\d+(?:\.\d+)?)", line)
            if match:
                return float(match.group(1))
    return None


def _start_server(port: int, enable_graceful_shutdown: bool = False):
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
        stdout=None,
        stderr=None,
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


async def _continuous_request_loop(
    client: openai.AsyncOpenAI,
    state: DrainState,
    sigterm_sent: asyncio.Event,
):
    while not state.stop_requesting:
        try:
            await client.completions.create(
                model=MODEL_NAME,
                prompt="Count:",
                max_tokens=10,
            )
            if sigterm_sent.is_set():
                state.requests_completed_after_sigterm += 1
        except openai.APIStatusError as e:
            if e.status_code == 503:
                state.got_503 = True
                state.draining_detected = True
            else:
                state.errors.append(f"API error: {e}")
        except (openai.APIConnectionError, httpx.RemoteProtocolError):
            state.connection_errors += 1
            if sigterm_sent.is_set():
                break
        except Exception as e:
            state.errors.append(f"Unexpected error: {e}")
            break
        await asyncio.sleep(0.01)


async def _poll_draining_metric(
    port: int,
    state: DrainState,
    sigterm_sent: asyncio.Event,
    timeout: float,
):
    await sigterm_sent.wait()

    start = time.time()
    async with httpx.AsyncClient() as http_client:
        while time.time() - start < timeout and not state.stop_requesting:
            try:
                response = await http_client.get(
                    f"http://localhost:{port}/metrics",
                    timeout=2.0,
                )
                if response.status_code == 200:
                    value = _parse_draining_metric(response.text)
                    state.metric_value = value
                    if value == 1.0:
                        state.draining_detected = True
                        return
            except (httpx.ConnectError, httpx.ReadTimeout):
                break
            await asyncio.sleep(0.05)


@pytest.mark.asyncio
async def test_graceful_shutdown_drains_and_sets_metric():
    """Verify graceful shutdown: 503s, draining metric=1, requests complete."""
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

        async with httpx.AsyncClient() as http_client:
            response = await http_client.get(
                f"http://localhost:{port}/metrics", timeout=5.0
            )
            assert response.status_code == 200
            initial_value = _parse_draining_metric(response.text)
            assert initial_value == 0.0, "Expected metric=0 before shutdown"

        state = DrainState()
        sigterm_sent = asyncio.Event()

        request_task = asyncio.create_task(
            _continuous_request_loop(client, state, sigterm_sent)
        )
        metric_task = asyncio.create_task(
            _poll_draining_metric(port, state, sigterm_sent, _DRAIN_DETECTION_TIMEOUT)
        )

        await asyncio.sleep(0.5)
        proc.send_signal(signal.SIGTERM)
        sigterm_sent.set()

        try:
            await asyncio.wait_for(
                asyncio.gather(request_task, metric_task, return_exceptions=True),
                timeout=_DRAIN_DETECTION_TIMEOUT,
            )
        except asyncio.TimeoutError:
            pass
        finally:
            state.stop_requesting = True
            if not request_task.done():
                request_task.cancel()
            if not metric_task.done():
                metric_task.cancel()
            await asyncio.gather(request_task, metric_task, return_exceptions=True)

        assert state.draining_detected, (
            f"Failed to detect drain. 503: {state.got_503}, "
            f"metric: {state.metric_value}, errors: {state.errors}"
        )
        assert state.got_503 or state.metric_value == 1.0
        assert state.requests_completed_after_sigterm > 0 or state.got_503

    finally:
        if proc.poll() is None:
            proc.terminate()
        return_code = proc.wait(timeout=_PROCESS_EXIT_TIMEOUT)
        assert return_code in (0, -15, None), f"Unexpected return code: {return_code}"
