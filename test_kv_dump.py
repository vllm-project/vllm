#!/usr/bin/env python3
"""Test script for KV cache dumping feature."""

import glob
import os
import time
from pathlib import Path

import requests

# Configuration
SERVER_URL = "http://localhost:8000"
OUTPUT_DIR = os.path.expanduser("~/temp/vllm_kv_dump_test")
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def wait_for_server(timeout: int = 120) -> bool:
    """Wait for the vLLM server to be ready."""
    print("Waiting for server to be ready...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{SERVER_URL}/health", timeout=5)
            if resp.status_code == 200:
                print("Server is ready!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(2)
    print("Server failed to start within timeout")
    return False


def send_completion_request(prompt: str, max_tokens: int = 50) -> dict:
    """Send a completion request to the server."""
    resp = requests.post(
        f"{SERVER_URL}/v1/completions",
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def analyze_dump_file(filepath: str) -> dict:
    """Analyze a KV dump safetensors file."""
    try:
        from safetensors import safe_open
        from safetensors.torch import load_file
    except ImportError:
        print("safetensors not installed, skipping file analysis")
        return {}

    # Load tensors
    tensors = load_file(filepath)

    # Load metadata
    with safe_open(filepath, framework="pt") as f:
        metadata = f.metadata()

    result = {
        "filepath": filepath,
        "metadata": metadata,
        "tensors": {},
    }

    for name, tensor in sorted(tensors.items()):
        result["tensors"][name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }

    return result


def run_tests():
    """Run the KV dump tests."""
    print("=" * 60)
    print("KV Cache Dump Test Suite")
    print("=" * 60)

    # Clear any existing dumps
    for f in glob.glob(f"{OUTPUT_DIR}/*.safetensors"):
        Path(f).unlink()
    print(f"\nCleared existing dumps in {OUTPUT_DIR}")

    # Wait for server
    if not wait_for_server():
        return False

    # Test 1: Simple code completion
    print("\n" + "-" * 60)
    print("Test 1: Simple code completion")
    print("-" * 60)

    prompt = "def fibonacci(n):\n    "
    print(f"Prompt: {repr(prompt)}")

    result = send_completion_request(prompt, max_tokens=30)
    completion = result["choices"][0]["text"]
    print(f"Completion: {repr(completion)}")
    print(f"Usage: {result['usage']}")

    # Wait a moment for the dump to be written
    time.sleep(1)

    # Check for dump files
    dump_files = glob.glob(f"{OUTPUT_DIR}/*.safetensors")
    print(f"\nDump files found: {len(dump_files)}")

    if not dump_files:
        print("ERROR: No dump files created!")
        return False

    # Analyze the dump
    for dump_file in dump_files:
        print(f"\n--- Analyzing: {Path(dump_file).name} ---")
        analysis = analyze_dump_file(dump_file)

        if analysis.get("metadata"):
            print("\nMetadata:")
            for k, v in sorted(analysis["metadata"].items()):
                print(f"  {k}: {v}")

        if analysis.get("tensors"):
            print("\nTensors:")
            for name, info in sorted(analysis["tensors"].items()):
                print(f"  {name}: shape={info['shape']}, dtype={info['dtype']}")

            # Validate S consistency
            s_values = []
            for name, info in analysis["tensors"].items():
                if name.startswith("K_layer") or name.startswith("V_layer"):
                    s_values.append(info["shape"][0])
                elif name == "token_ids":
                    s_values.append(info["shape"][0])

            if len(set(s_values)) == 1:
                print(f"\n✓ S dimension consistent across all tensors: S={s_values[0]}")
            else:
                print(f"\n✗ S dimension INCONSISTENT: {s_values}")
                return False

    # Test 2: Another prompt
    print("\n" + "-" * 60)
    print("Test 2: Natural language prompt")
    print("-" * 60)

    prompt2 = "The quick brown fox"
    print(f"Prompt: {repr(prompt2)}")

    result2 = send_completion_request(prompt2, max_tokens=20)
    completion2 = result2["choices"][0]["text"]
    print(f"Completion: {repr(completion2)}")

    time.sleep(1)

    new_dumps = glob.glob(f"{OUTPUT_DIR}/*.safetensors")
    print(f"\nTotal dump files now: {len(new_dumps)}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    import sys

    success = run_tests()
    sys.exit(0 if success else 1)
