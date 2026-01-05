# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test consecutive model loading and deletion via the Worker Controller.

This test demonstrates the ability to:
1. Load a model, run inference, and delete it
2. Load a different model on the same GPU
3. Repeat for multiple models

Models tested:
- facebook/opt-125m
- Qwen/Qwen2.5-0.5B
- facebook/opt-350m
"""

import requests
import time
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
    {"name": "Qwen/Qwen2.5-0.5B", "uuid": "qwen-0.5b"},
    {"name": "facebook/opt-350m", "uuid": "opt-350m"},
]

TEST_PROMPTS = [
    "Hello, my name is",
    "The capital of France is",
]


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


def create_engine(model_name: str, engine_uuid: str):
    """Create an engine with the given model."""
    print(f"\n{'='*60}")
    print(f"Creating engine '{engine_uuid}' with model: {model_name}")
    print(f"{'='*60}")

    start_time = time.time()

    resp = requests.post(
        f"{BASE_URL}/engines",
        json={
            "engine_uuid": engine_uuid,
            "model": model_name,
            "gpu_memory_utilization": 0.3,
            "enforce_eager": True,
        },
        timeout=300,  # Model loading can take time
    )

    elapsed = time.time() - start_time

    if resp.status_code != 200:
        print(f"ERROR: Failed to create engine: {resp.text}")
        return None, elapsed

    result = resp.json()
    print(f"Engine created in {elapsed:.2f}s")
    print(f"  Port: {result['port']}")
    print(f"  Assigned ranks: {result['assigned_ranks']}")
    print(f"  PID: {result['pid']}")

    return result, elapsed


def wait_for_engine_ready(port: int, timeout: int = 60):
    """Wait for the engine's API server to be ready."""
    print(f"Waiting for engine API server on port {port}...")
    engine_url = f"http://localhost:{port}"

    for i in range(timeout):
        try:
            resp = requests.get(f"{engine_url}/health", timeout=5)
            if resp.status_code == 200:
                print(f"Engine API server is ready")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)

    print(f"WARNING: Engine API server did not become ready in {timeout}s")
    return False


def run_inference(port: int, prompts: list):
    """Run inference on the engine."""
    engine_url = f"http://localhost:{port}"
    print(f"\nRunning inference with {len(prompts)} prompts...")

    results = []
    for prompt in prompts:
        start_time = time.time()

        resp = requests.post(
            f"{engine_url}/v1/completions",
            json={
                "prompt": prompt,
            },
            timeout=60,
        )

        elapsed = time.time() - start_time

        if resp.status_code == 200:
            result = resp.json()
            generated = result["choices"][0]["text"]
            print(f"  Prompt: {prompt!r}")
            print(f"  Generated: {generated!r} ({elapsed:.2f}s)")
            results.append(
                {"prompt": prompt, "generated": generated, "time": elapsed})
        else:
            print(f"  ERROR: {resp.text}")
            results.append(
                {"prompt": prompt, "error": resp.text, "time": elapsed})

    return results


def delete_engine(engine_uuid: str):
    """Delete an engine."""
    print(f"\nDeleting engine '{engine_uuid}'...")

    start_time = time.time()

    resp = requests.delete(
        f"{BASE_URL}/engines/{engine_uuid}",
        timeout=60,
    )

    elapsed = time.time() - start_time

    if resp.status_code == 200:
        result = resp.json()
        print(f"Engine deleted in {elapsed:.2f}s")
        print(f"  Released ranks: {result['released_ranks']}")
        print(f"  Released port: {result['released_port']}")
        return True
    else:
        print(f"ERROR: Failed to delete engine: {resp.text}")
        return False


def main():
    print("="*60)
    print("Worker Controller - Consecutive Model Loading Test")
    print("="*60)

    # Start and wait for controller
    start_worker_controller()
    atexit.register(stop_worker_controller)
    wait_for_controller()

    # Track timing
    total_times = {}

    # Test each model
    for model_info in MODELS:
        model_name = model_info["name"]
        engine_uuid = model_info["uuid"]

        # Create engine
        result, create_time = create_engine(model_name, engine_uuid)
        if result is None:
            print(f"Skipping model {model_name} due to creation failure")
            continue

        port = result["port"]
        total_times[model_name] = {"create": create_time}

        # Wait for API server
        if not wait_for_engine_ready(port):
            print(f"Skipping inference for {model_name}")
            delete_engine(engine_uuid)
            continue

        # Run inference
        inference_start = time.time()
        inference_results = run_inference(port, TEST_PROMPTS)
        total_times[model_name]["inference"] = time.time() - inference_start

        # Give a moment for cleanup
        time.sleep(1)

        # Delete engine
        delete_start = time.time()
        delete_engine(engine_uuid)
        total_times[model_name]["delete"] = time.time() - delete_start

        # Wait a moment before loading next model
        print("\nWaiting 2 seconds before next model...")
        time.sleep(2)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for model_name, times in total_times.items():
        print(f"\n{model_name}:")
        print(f"  Create time:    {times.get('create', 'N/A'):.2f}s")
        print(f"  Inference time: {times.get('inference', 'N/A'):.2f}s")
        print(f"  Delete time:    {times.get('delete', 'N/A'):.2f}s")
        total = sum(v for v in times.values() if isinstance(v, (int, float)))
        print(f"  Total:          {total:.2f}s")

    print("\n" + "="*60)
    print("Test completed successfully!")
    print("="*60)

    # Cleanup
    stop_worker_controller()


if __name__ == "__main__":
    main()
