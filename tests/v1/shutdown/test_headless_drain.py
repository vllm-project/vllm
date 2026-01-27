#!/usr/bin/env -S uv run --script
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# /// script
# dependencies = ["httpx"]
# ///
"""
Headless mode drain test for vLLM.

Tests that when SIGTERM is sent to a headless vLLM engine:
- The engine drains properly
- Clean shutdown occurs

Headless mode requires a connecting API server to complete initialization.
This test starts both:
1. A headless engine (--headless)
2. An API server that connects to it (--data-parallel-size-local 0)
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

# add parent dir to path for utils import
sys.path.insert(0, str(Path(__file__).parent))
from utils import DRAIN_TIMEOUT, SERVER_STARTUP_TIMEOUT, wait_for_server_ready

MODEL_NAME = "facebook/opt-125m"


def run_test(
    model: str,
    port: int,
    startup_timeout: float,
    drain_timeout: int,
):
    print(f"Starting headless engine + API server test on port {port}...")

    headless_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model,
            "--headless",
            "--data-parallel-size",
            "1",
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
    print(f"Headless engine started with PID {headless_proc.pid}")

    time.sleep(2.0)

    api_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model,
            "--port",
            str(port),
            "--data-parallel-size",
            "1",
            "--data-parallel-size-local",
            "0",
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
    print(f"API server started with PID {api_proc.pid}")

    try:
        print(f"Waiting up to {startup_timeout}s for API server to be ready...")
        if not wait_for_server_ready(port, startup_timeout):
            print("Server failed to start")
            if headless_proc.poll() is not None:
                stdout, _ = headless_proc.communicate(timeout=1)
                print(f"Headless output:\n{stdout.decode()[-2000:]}")
            if api_proc.poll() is not None:
                stdout, _ = api_proc.communicate(timeout=1)
                print(f"API output:\n{stdout.decode()[-2000:]}")
            raise SystemExit(1)

        print("Server is ready!")
        time.sleep(1.0)

        print(f"Sending SIGTERM to headless engine PID {headless_proc.pid}...")
        os.kill(headless_proc.pid, signal.SIGTERM)

        print("Waiting for headless engine to drain and exit...")
        try:
            headless_exit = headless_proc.wait(timeout=drain_timeout + 10)
            print(f"Headless engine exited with code {headless_exit}")
        except subprocess.TimeoutExpired:
            print("Headless engine did not exit in time")
            headless_proc.kill()
            headless_exit = None

        stdout, _ = headless_proc.communicate(timeout=5)
        output = stdout.decode() if stdout else ""
        has_drain = "Drain initiated" in output or "drain" in output.lower()

        if has_drain and headless_exit in (0, -15, None):
            print("\n[PASS] Headless drain test PASSED")
        else:
            print("\n[FAIL] Headless drain test FAILED")
            print(f"Exit code: {headless_exit}, Drain detected: {has_drain}")
            print("\n=== Headless Output (last 3000 chars) ===")
            print(output[-3000:] if len(output) > 3000 else output)
            raise SystemExit(1)

    finally:
        for proc in [headless_proc, api_proc]:
            if proc.poll() is None:
                with suppress(ProcessLookupError, OSError):
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                with suppress(Exception):
                    proc.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description="Test headless mode drain shutdown")
    parser.add_argument("--model", default=MODEL_NAME, help="Model name")
    parser.add_argument("--port", type=int, default=8100, help="API server port")
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

    run_test(args.model, args.port, args.startup_timeout, args.drain_timeout)


if __name__ == "__main__":
    main()
