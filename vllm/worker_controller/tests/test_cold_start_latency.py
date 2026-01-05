# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test cold start latency comparison: Worker Controller vs Standard vLLM.

This test demonstrates the cold start latency reduction achieved by
the Worker Controller's pre-initialized worker pool.

The test compares:
1. Worker Controller: Workers are already initialized, only model loading needed
2. Standard vLLM: Full initialization including CUDA context, distributed setup, etc.

Models tested:
- facebook/opt-125m
- Qwen/Qwen2.5-0.5B
"""

import requests
import time
import subprocess
import sys
import os
import signal
import json
import threading
import uvicorn
import atexit

# Import WorkerController
from vllm.worker_controller.worker_controller import WorkerController
from vllm.worker_controller.worker_controller_server import app, worker_controller
import vllm.worker_controller.worker_controller_server as wc_server

BASE_URL = "http://localhost:8000"

# Global reference to server thread
_server_thread = None
_server = None

# Test configuration
MODELS = [
    {"name": "facebook/opt-125m", "uuid": "opt-125m"},
    {"name": "Qwen/Qwen3-0.6B", "uuid": "qwen-0.6b"},
]

TEST_PROMPT = "Hello, my name is"


def start_worker_controller():
    """Start the worker controller server in a background thread."""
    global _server_thread, _server

    print("Starting Worker Controller...")

    # Initialize the WorkerController
    wc_server.worker_controller = WorkerController()
    print(
        f"WorkerController initialized with {len(wc_server.worker_controller.executor.workers)} workers")

    # Create uvicorn config
    config = uvicorn.Config(app, host="0.0.0.0",
                            port=8000, log_level="warning")
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
    print(f"\n{'='*60}")
    print(f"Worker Controller Cold Start: {model_name}")
    print(f"{'='*60}")

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
        first_inference_time = float('inf')

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


def measure_standard_vllm_cold_start(model_name: str):
    """
    Measure cold start time using standard vLLM (subprocess).

    Cold start = time from process start to first token generation.
    This includes CUDA context creation, distributed setup, model loading, etc.
    """
    print(f"\n{'='*60}")
    print(f"Standard vLLM Cold Start: {model_name}")
    print(f"{'='*60}")

    # Create a simple script to measure cold start
    script = f'''
import time
import sys

total_start = time.time()

# Import vLLM (triggers CUDA init)
print("Importing vLLM...", flush=True)
import_start = time.time()
from vllm import LLM, SamplingParams
import_time = time.time() - import_start
print(f"  Import time: {{import_time:.2f}}s", flush=True)

# Create LLM (model loading, KV cache setup, etc.)
print("Creating LLM...", flush=True)
create_start = time.time()
llm = LLM(
    model="{model_name}",
    gpu_memory_utilization=0.3,
    enforce_eager=True,
)
create_time = time.time() - create_start
print(f"  LLM creation: {{create_time:.2f}}s", flush=True)

# First inference
print("Running first inference...", flush=True)
inference_start = time.time()
sampling_params = SamplingParams(temperature=0.0, max_tokens=10)
outputs = llm.generate(["{TEST_PROMPT}"], sampling_params)
inference_time = time.time() - inference_start
print(f"  First inference: {{inference_time:.2f}}s", flush=True)
print(f"  Generated: {{outputs[0].outputs[0].text!r}}", flush=True)

total_time = time.time() - total_start
print(f"TOTAL_TIME:{{total_time:.4f}}", flush=True)
print(f"IMPORT_TIME:{{import_time:.4f}}", flush=True)
print(f"CREATE_TIME:{{create_time:.4f}}", flush=True)
print(f"INFERENCE_TIME:{{inference_time:.4f}}", flush=True)
'''

    total_start = time.time()

    # Run in subprocess
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0

    print("Starting vLLM subprocess...")
    proc = subprocess.Popen(
        [sys.executable, '-c', script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
    )

    # Capture output
    import_time = None
    create_time = None
    inference_time = None
    vllm_total_time = None

    for line in proc.stdout:
        print(f"  {line.rstrip()}")
        if line.startswith("TOTAL_TIME:"):
            vllm_total_time = float(line.split(":")[1])
        elif line.startswith("IMPORT_TIME:"):
            import_time = float(line.split(":")[1])
        elif line.startswith("CREATE_TIME:"):
            create_time = float(line.split(":")[1])
        elif line.startswith("INFERENCE_TIME:"):
            inference_time = float(line.split(":")[1])

    proc.wait()

    subprocess_total = time.time() - total_start

    return {
        "model": model_name,
        "import_time": import_time,
        "create_time": create_time,
        "first_inference_time": inference_time,
        "total_cold_start": vllm_total_time or subprocess_total,
    }


def main():
    print("="*70)
    print("Cold Start Latency Comparison: Worker Controller vs Standard vLLM")
    print("="*70)
    print()
    print("This test measures the time from 'request to serve a model' to")
    print("'first token generated'.")
    print()
    print("Worker Controller benefits:")
    print("  - CUDA context already initialized")
    print("  - PyTorch distributed already set up")
    print("  - Workers ready and waiting")
    print("  - Only model weights need to be loaded")
    print()

    results = {
        "worker_controller": [],
        "standard_vllm": [],
    }

    # First, run standard vLLM tests (while worker controller GPUs are free)
    print("\n" + "="*70)
    print("PHASE 1: Standard vLLM Cold Start Measurements")
    print("="*70)

    for model_info in MODELS:
        model_name = model_info["name"]
        result = measure_standard_vllm_cold_start(model_name)
        if result:
            results["standard_vllm"].append(result)
        time.sleep(3)  # Wait for GPU memory to clear

    # Now run Worker Controller tests
    print("\n" + "="*70)
    print("PHASE 2: Worker Controller Cold Start Measurements")
    print("="*70)

    # Start and wait for controller
    start_worker_controller()
    atexit.register(stop_worker_controller)
    wait_for_controller()

    for model_info in MODELS:
        model_name = model_info["name"]
        engine_uuid = model_info["uuid"]
        result = measure_worker_controller_cold_start(model_name, engine_uuid)
        if result:
            results["worker_controller"].append(result)

    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    print("\n{:<25} {:>15} {:>15} {:>15}".format(
        "Model", "Std vLLM (s)", "Worker Ctrl (s)", "Speedup"))
    print("-"*70)

    for i, model_info in enumerate(MODELS):
        model_name = model_info["name"]

        std_result = next(
            (r for r in results["standard_vllm"] if r["model"] == model_name), None)
        wc_result = next(
            (r for r in results["worker_controller"] if r["model"] == model_name), None)

        if std_result and wc_result:
            std_time = std_result["total_cold_start"]
            wc_time = wc_result["total_cold_start"]
            speedup = std_time / wc_time if wc_time > 0 else float('inf')

            print("{:<25} {:>15.2f} {:>15.2f} {:>14.1f}x".format(
                model_name[:25], std_time, wc_time, speedup))

    print("\n" + "-"*70)
    print("\nDetailed Breakdown:")

    for model_info in MODELS:
        model_name = model_info["name"]
        print(f"\n{model_name}:")

        std_result = next(
            (r for r in results["standard_vllm"] if r["model"] == model_name), None)
        wc_result = next(
            (r for r in results["worker_controller"] if r["model"] == model_name), None)

        if std_result:
            print("  Standard vLLM:")
            print(
                f"    Import time:     {std_result.get('import_time', 'N/A'):.2f}s")
            print(
                f"    Create time:     {std_result.get('create_time', 'N/A'):.2f}s")
            print(
                f"    First inference: {std_result.get('first_inference_time', 'N/A'):.2f}s")
            print(
                f"    Total:           {std_result['total_cold_start']:.2f}s")

        if wc_result:
            print("  Worker Controller:")
            print(
                f"    Engine creation: {wc_result.get('create_time', 'N/A'):.2f}s")
            print(
                f"    API ready:       {wc_result.get('api_ready_time', 'N/A'):.2f}s")
            print(
                f"    First inference: {wc_result.get('first_inference_time', 'N/A'):.2f}s")
            print(f"    Total:           {wc_result['total_cold_start']:.2f}s")

    print("\n" + "="*70)
    print("Test completed!")
    print("="*70)

    # Cleanup
    stop_worker_controller()


if __name__ == "__main__":
    main()
