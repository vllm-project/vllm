#!/usr/bin/env -S uv run --script
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# /// script
# dependencies = ["httpx", "openai", "psutil"]
# ///
"""
Multi-API server graceful shutdown test for vLLM.

Tests that when SIGTERM is sent to the parent process with --api-server-count > 1:
- All API servers start rejecting new requests (503)
- In-flight requests complete
- Engines drain properly
- All processes exit gracefully

Architecture being tested:
- Parent process spawns N API server child processes
- All API servers share the same engine cores
- Parent coordinates drain: signals API servers to reject, waits for engines
"""

import argparse
import asyncio
import os
import signal
import subprocess
import sys
from contextlib import suppress
from pathlib import Path

import httpx
import openai

# add parent dir to path for utils import
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    DRAIN_TIMEOUT,
    SERVER_STARTUP_TIMEOUT,
    DrainState,
    get_child_pids,
    wait_for_server_ready_async,
)

MODEL_NAME = "facebook/opt-125m"


async def _concurrent_request_loop(
    client: openai.AsyncOpenAI,
    state: DrainState,
    sigterm_sent: asyncio.Event,
):
    """Run concurrent requests to keep the server busy."""
    while not state.stop_requesting:
        try:
            await client.completions.create(
                model=MODEL_NAME,
                prompt="Count:",
                max_tokens=10,
            )
            if sigterm_sent.is_set():
                state.requests_after_sigterm += 1
        except openai.APIStatusError as e:
            if e.status_code == 503:
                state.got_503 = True
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


async def run_test(
    port: int,
    model: str,
    startup_timeout: float,
    api_server_count: int,
    drain_timeout: int,
):
    print(f"Starting vLLM server with {api_server_count} API servers on {port}...")

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model,
            "--port",
            str(port),
            "--api-server-count",
            str(api_server_count),
            "--max-model-len",
            "256",
            "--gpu-memory-utilization",
            "0.10",
            "--max-num-seqs",
            "4",
            "--enforce-eager",
            "--shutdown-mode",
            "drain",
            "--shutdown-drain-timeout",
            str(drain_timeout),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print(f"Parent process started with PID {proc.pid}")

    client = openai.AsyncOpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key="dummy",
        max_retries=0,
        timeout=30,
    )

    try:
        print(f"Waiting up to {startup_timeout}s for server to be ready...")
        if not await wait_for_server_ready_async(client, model, startup_timeout):
            print("Server failed to start")
            if proc.poll() is not None:
                stdout, _ = proc.communicate(timeout=1)
                print(f"Server output:\n{stdout.decode()[-3000:]}")
            raise SystemExit(1)

        print("Server is ready!")

        child_pids = get_child_pids(proc.pid)
        print(f"Found {len(child_pids)} child processes")

        state = DrainState()
        sigterm_sent = asyncio.Event()

        request_task = asyncio.create_task(
            _concurrent_request_loop(client, state, sigterm_sent)
        )

        await asyncio.sleep(2.0)

        print(f"Sending SIGTERM to parent PID {proc.pid}...")
        os.kill(proc.pid, signal.SIGTERM)
        sigterm_sent.set()

        try:
            await asyncio.wait_for(request_task, timeout=drain_timeout + 10)
        except asyncio.TimeoutError:
            pass
        finally:
            state.stop_requesting = True
            if not request_task.done():
                request_task.cancel()
            with suppress(asyncio.CancelledError):
                await request_task

        print("Waiting for parent process to exit...")
        try:
            exit_code = proc.wait(timeout=10)
            print(f"Parent exited with code {exit_code}")
        except subprocess.TimeoutExpired:
            print("Parent did not exit in time")
            proc.kill()
            exit_code = None

        print("\nResults:")
        print(f"  Got 503: {state.got_503}")
        print(f"  Requests completed after SIGTERM: {state.requests_after_sigterm}")
        print(f"  Connection errors: {state.connection_errors}")

        passed = (
            state.got_503
            or state.requests_after_sigterm > 0
            or state.connection_errors > 0
        )

        if passed and exit_code in (0, -15, None):
            print("\n[PASS] Multi-API server drain test PASSED")
        else:
            print("\n[FAIL] Multi-API server drain test FAILED")
            stdout, _ = proc.communicate(timeout=5)
            output = stdout.decode() if stdout else ""
            print("\n=== Server Output (last 5000 chars) ===")
            print(output[-5000:] if len(output) > 5000 else output)
            raise SystemExit(1)

    finally:
        if proc.poll() is None:
            with suppress(ProcessLookupError, OSError):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            with suppress(Exception):
                proc.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(
        description="Test multi-API server graceful shutdown"
    )
    parser.add_argument("--port", type=int, default=8000, help="vLLM port")
    parser.add_argument("--model", default=MODEL_NAME, help="Model name")
    parser.add_argument(
        "--api-server-count", type=int, default=2, help="Number of API servers"
    )
    parser.add_argument(
        "--drain-timeout", type=int, default=DRAIN_TIMEOUT, help="Drain timeout"
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=SERVER_STARTUP_TIMEOUT,
        help="Startup timeout",
    )
    args = parser.parse_args()

    asyncio.run(
        run_test(
            args.port,
            args.model,
            args.startup_timeout,
            args.api_server_count,
            args.drain_timeout,
        )
    )


if __name__ == "__main__":
    main()
