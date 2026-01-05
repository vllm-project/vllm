# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test cold start latency comparison: Worker Controller vs Standard vLLM.

This test demonstrates the cold start latency reduction achieved by
the Worker Controller's pre-initialized worker pool, especially when
loading multiple models sequentially.

The test compares (both using API servers for fair comparison):
1. Worker Controller: Workers are already initialized, CUDA context reused
2. Standard vLLM API Server: Full initialization each time including CUDA context

Key insight: The Worker Controller shines when you need to load/unload
multiple models sequentially because it reuses the CUDA context and
distributed setup, while standard vLLM must reinitialize everything each time.

Models tested:
- facebook/opt-125m
- Qwen/Qwen3-0.6B
- Qwen/Qwen3-1.7B
"""

import atexit
import os
import subprocess
import sys
import threading
import time

import requests
import uvicorn

import vllm.worker_controller.worker_controller_server as wc_server

# Import WorkerController
from vllm.worker_controller.worker_controller import WorkerController
from vllm.worker_controller.worker_controller_server import app

BASE_URL = "http://localhost:8000"

# Global reference to server thread
_server_thread = None
_server = None

# Test configuration
MODELS = [
    {"name": "facebook/opt-125m", "uuid": "opt-125m"},
    {"name": "Qwen/Qwen3-0.6B", "uuid": "qwen-0.6b"},
    {"name": "Qwen/Qwen3-1.7B", "uuid": "qwen-1.7b"},
]

TEST_PROMPT = "Hello, my name is"


def start_worker_controller():
    """Start the worker controller server in a background thread."""
    global _server_thread, _server

    print("Starting Worker Controller...")

    # Initialize the WorkerController
    wc_server.worker_controller = WorkerController()
    print(
        f"WorkerController initialized with {len(wc_server.worker_controller.executor.workers)} workers"
    )

    # Create uvicorn config
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="warning")
    _server = uvicorn.Server(config)

    # Run in background thread
    _server_thread = threading.Thread(target=_server.run, daemon=True)
    _server_thread.start()

    print("Worker Controller server started on port 8000")


def stop_worker_controller():
    """Stop the worker controller server."""
    global _server, _server_thread

    if _server is not None:
        print("\nStopping Worker Controller...")
        _server.should_exit = True
        if _server_thread is not None:
            _server_thread.join(timeout=5)
        print("Worker Controller stopped.")


def wait_for_controller():
    """Wait for the worker controller to be ready."""
    print("Waiting for Worker Controller to be ready...")
    for i in range(30):
        try:
            resp = requests.get(f"{BASE_URL}/health", timeout=5)
            if resp.status_code == 200:
                print(f"Worker Controller is ready: {resp.json()}")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    raise RuntimeError("Worker Controller did not become ready in time")


def measure_worker_controller_cold_start(model_name: str, engine_uuid: str):
    """
    Measure cold start time using Worker Controller.

    Cold start = time from API call to first token generation.
    With Worker Controller, CUDA context and distributed setup are already done.
    """
    print(f"\n{'=' * 60}")
    print(f"Worker Controller Cold Start: {model_name}")
    print(f"{'=' * 60}")

    total_start = time.time()

    # Create engine
    print("Creating engine...")
    create_start = time.time()

    resp = requests.post(
        f"{BASE_URL}/engines",
        json={
            "engine_uuid": engine_uuid,
            "model": model_name,
            "gpu_memory_utilization": 0.3,
            "enforce_eager": True,
        },
        timeout=300,
    )

    create_time = time.time() - create_start

    if resp.status_code != 200:
        print(f"ERROR: Failed to create engine: {resp.text}")
        return None

    result = resp.json()
    port = result["port"]
    print(f"  Engine creation: {create_time:.2f}s")

    # Wait for API server ready
    print("Waiting for API server...")
    api_ready_start = time.time()
    engine_url = f"http://localhost:{port}"

    for i in range(60):
        try:
            resp = requests.get(f"{engine_url}/health", timeout=5)
            if resp.status_code == 200:
                break
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(0.5)

    api_ready_time = time.time() - api_ready_start
    print(f"  API server ready: {api_ready_time:.2f}s")

    # First inference (includes any remaining warmup)
    print("Running first inference...")
    inference_start = time.time()

    resp = requests.post(
        f"{engine_url}/v1/completions",
        json={
            "prompt": TEST_PROMPT,
        },
        timeout=60,
    )

    first_inference_time = time.time() - inference_start

    if resp.status_code == 200:
        generated = resp.json()["choices"][0]["text"]
        print(f"  First inference: {first_inference_time:.2f}s")
        print(f"  Generated: {generated!r}")
    else:
        print(f"  ERROR: {resp.text}")
        first_inference_time = float("inf")

    total_time = time.time() - total_start

    # Cleanup
    print("Cleaning up...")
    requests.delete(f"{BASE_URL}/engines/{engine_uuid}", timeout=60)
    time.sleep(2)  # Wait for cleanup

    return {
        "model": model_name,
        "create_time": create_time,
        "api_ready_time": api_ready_time,
        "first_inference_time": first_inference_time,
        "total_cold_start": total_time,
    }


def measure_standard_vllm_cold_start(model_name: str, port: int = 8001):
    """
    Measure cold start time using standard vLLM API server (vllm serve).

    Cold start = time from process start to first token generation via API.
    This includes CUDA context creation, distributed setup, model loading,
    and API server startup - matching what Worker Controller does.
    """
    print(f"\n{'=' * 60}")
    print(f"Standard vLLM API Server Cold Start: {model_name}")
    print(f"{'=' * 60}")

    total_start = time.time()

    # Start vLLM serve in subprocess
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_name,
        "--port",
        str(port),
        "--gpu-memory-utilization",
        "0.3",
        "--enforce-eager",
    ]

    print(f"Starting vLLM API server on port {port}...")
    server_start = time.time()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
    )

    # Wait for API server to be ready
    print("Waiting for API server to be ready...")
    api_url = f"http://localhost:{port}"
    server_ready = False
    server_ready_time = None

    for i in range(120):  # Wait up to 120 seconds
        try:
            resp = requests.get(f"{api_url}/health", timeout=2)
            if resp.status_code == 200:
                server_ready = True
                server_ready_time = time.time() - server_start
                print(f"  Server ready: {server_ready_time:.2f}s")
                break
        except requests.exceptions.ConnectionError:
            pass
        except requests.exceptions.ReadTimeout:
            pass
        time.sleep(0.5)

    if not server_ready:
        print("ERROR: Server did not become ready in time")
        proc.terminate()
        proc.wait()
        return None

    # First inference via API
    print("Running first inference...")
    inference_start = time.time()

    try:
        resp = requests.post(
            f"{api_url}/v1/completions",
            json={
                "model": model_name,
                "prompt": TEST_PROMPT,
                "max_tokens": 10,
                "temperature": 0.0,
            },
            timeout=60,
        )

        first_inference_time = time.time() - inference_start

        if resp.status_code == 200:
            generated = resp.json()["choices"][0]["text"]
            print(f"  First inference: {first_inference_time:.2f}s")
            print(f"  Generated: {generated!r}")
        else:
            print(f"  ERROR: {resp.status_code} - {resp.text}")
            first_inference_time = float("inf")
    except Exception as e:
        print(f"  ERROR: {e}")
        first_inference_time = float("inf")

    total_time = time.time() - total_start

    # Cleanup - terminate server
    print("Cleaning up server...")
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    time.sleep(2)  # Wait for port to be released

    return {
        "model": model_name,
        "server_ready_time": server_ready_time,
        "first_inference_time": first_inference_time,
        "total_cold_start": total_time,
    }


def main():
    print("=" * 80)
    print("Cold Start Latency: Sequential Model Loading Comparison")
    print("=" * 80)
    print()
    print(
        "This test measures the cumulative time to load multiple models SEQUENTIALLY."
    )
    print("Each model is loaded, tested, then unloaded before loading the next.")
    print()
    print("This demonstrates the Worker Controller's key advantage:")
    print("  - Standard vLLM: Must reinitialize CUDA context for EACH model load")
    print("  - Worker Controller: Reuses CUDA context across all model loads")
    print()
    print(f"Models to load sequentially: {[m['name'] for m in MODELS]}")
    print()

    results = {
        "worker_controller": [],
        "standard_vllm": [],
    }

    # =========================================================================
    # PHASE 1: Standard vLLM - Sequential model loads (each is a fresh process)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 1: Standard vLLM - Sequential Model Loads")
    print("=" * 80)
    print("Each model requires a fresh process with full CUDA initialization.")
    print()

    std_total_start = time.time()
    for model_info in MODELS:
        model_name = model_info["name"]
        result = measure_standard_vllm_cold_start(model_name)
        if result:
            results["standard_vllm"].append(result)
        time.sleep(3)  # Wait for GPU memory to clear
    std_total_time = time.time() - std_total_start

    # =========================================================================
    # PHASE 2: Worker Controller - Sequential model loads (reuses CUDA context)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PHASE 2: Worker Controller - Sequential Model Loads")
    print("=" * 80)
    print("Workers are pre-initialized. CUDA context is reused across all loads.")
    print()

    # Start and wait for controller
    start_worker_controller()
    atexit.register(stop_worker_controller)
    wait_for_controller()

    wc_total_start = time.time()
    for model_info in MODELS:
        model_name = model_info["name"]
        engine_uuid = model_info["uuid"]
        result = measure_worker_controller_cold_start(model_name, engine_uuid)
        if result:
            results["worker_controller"].append(result)
    wc_total_time = time.time() - wc_total_start

    # =========================================================================
    # RESULTS SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY - SEQUENTIAL MODEL LOADING")
    print("=" * 80)

    # Per-model breakdown
    print("\nPer-Model Cold Start Times:")
    print(
        "\n{:<25} {:>15} {:>15} {:>12} {:>10}".format(
            "Model", "Std vLLM (s)", "Worker Ctrl (s)", "Diff (s)", "Speedup"
        )
    )
    print("-" * 80)

    for i, model_info in enumerate(MODELS):
        model_name = model_info["name"]

        std_result = next(
            (r for r in results["standard_vllm"] if r["model"] == model_name), None
        )
        wc_result = next(
            (r for r in results["worker_controller"] if r["model"] == model_name), None
        )

        if std_result and wc_result:
            std_time = std_result["total_cold_start"]
            wc_time = wc_result["total_cold_start"]
            diff = std_time - wc_time
            speedup = std_time / wc_time if wc_time > 0 else float("inf")

            print(
                "{:<25} {:>15.2f} {:>15.2f} {:>12.2f} {:>9.1f}x".format(
                    model_name[:25], std_time, wc_time, diff, speedup
                )
            )

    # Cumulative totals (the key metric for sequential loading)
    print("\n" + "=" * 80)
    print("CUMULATIVE TIME FOR ALL {} MODELS:".format(len(MODELS)))
    print("=" * 80)

    std_cumulative = sum(r["total_cold_start"] for r in results["standard_vllm"])
    wc_cumulative = sum(r["total_cold_start"] for r in results["worker_controller"])
    cumulative_diff = std_cumulative - wc_cumulative
    cumulative_speedup = (
        std_cumulative / wc_cumulative if wc_cumulative > 0 else float("inf")
    )

    print(f"\n  Standard vLLM (fresh process each time):  {std_cumulative:.2f}s")
    print(f"  Worker Controller (reuses CUDA context):  {wc_cumulative:.2f}s")
    print(f"  Time saved:                               {cumulative_diff:.2f}s")
    print(f"  Speedup:                                  {cumulative_speedup:.2f}x")

    # Detailed breakdown
    print("\n" + "-" * 80)
    print("\nDetailed Breakdown:")

    for model_info in MODELS:
        model_name = model_info["name"]
        print(f"\n{model_name}:")

        std_result = next(
            (r for r in results["standard_vllm"] if r["model"] == model_name), None
        )
        wc_result = next(
            (r for r in results["worker_controller"] if r["model"] == model_name), None
        )

        if std_result:
            print("  Standard vLLM API Server:")
            print(
                f"    Server ready:    {std_result.get('server_ready_time', 'N/A'):.2f}s"
            )
            print(
                f"    First inference: {std_result.get('first_inference_time', 'N/A'):.2f}s"
            )
            print(f"    Total:           {std_result['total_cold_start']:.2f}s")

        if wc_result:
            print("  Worker Controller:")
            print(f"    Engine creation: {wc_result.get('create_time', 'N/A'):.2f}s")
            print(f"    API ready:       {wc_result.get('api_ready_time', 'N/A'):.2f}s")
            print(
                f"    First inference: {wc_result.get('first_inference_time', 'N/A'):.2f}s"
            )
            print(f"    Total:           {wc_result['total_cold_start']:.2f}s")

    print("\n" + "=" * 80)
    print("KEY TAKEAWAY:")
    print("=" * 80)
    print(
        f"For {len(MODELS)} sequential model loads, Worker Controller saved {cumulative_diff:.1f}s"
    )
    print("The more models you load/unload, the more time the Worker Controller saves")
    print("because it doesn't reinitialize CUDA context each time.")
    print("=" * 80)

    # Cleanup
    stop_worker_controller()


if __name__ == "__main__":
    main()
