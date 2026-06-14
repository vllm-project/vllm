#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Manual integration test: LoRA config loading with retry.

Run on a devGPU with the vLLM OSS venv activated:

    source ~/uv_env/vllm/bin/activate
    cd /data/users/$USER/fbsource/fbcode/vllm/trunk
    with-proxy python tests/lora/test_lora_config_retry_manual.py

This script:
  1. Launches a vLLM server with --enable-lora
  2. Dynamically loads a LoRA adapter (exercises _load_lora_config)
  3. Runs chat completion with the adapter
  4. Verifies the response
  5. Tears down the server
"""

import json
import os
import subprocess
import sys
import time
import urllib.request

MODEL = "Qwen/Qwen3-0.6B"
LORA_REPO = "charent/self_cognition_Alice"
LORA_NAME = "qwen3-lora-test"
PORT = 8192
BASE_URL = f"http://localhost:{PORT}"
TIMEOUT_SERVER_START = 300  # seconds


def wait_for_server():
    """Poll /health until the server is ready."""
    deadline = time.monotonic() + TIMEOUT_SERVER_START
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(f"{BASE_URL}/health")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    print("[OK] Server is healthy.")
                    return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError("Server did not become healthy in time.")


def api_post(path: str, body: dict) -> dict:
    """POST JSON to the vLLM OpenAI-compatible API."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/v1/{path}",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer token-abc123",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def main():
    # 1. Download LoRA adapter files
    print(f"[1/5] Downloading LoRA adapter: {LORA_REPO}")
    from huggingface_hub import snapshot_download

    lora_path = snapshot_download(repo_id=LORA_REPO)
    print(f"       Downloaded to: {lora_path}")

    # 2. Launch vLLM server
    print(f"[2/5] Starting vLLM server with {MODEL}...")
    env = os.environ.copy()
    env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    server_proc = subprocess.Popen(
        [
            "vllm",
            "serve",
            MODEL,
            "--dtype",
            "bfloat16",
            "--max-model-len",
            "8192",
            "--enforce-eager",
            "--enable-lora",
            "--max-lora-rank",
            "64",
            "--max-cpu-loras",
            "2",
            "--port",
            str(PORT),
        ],
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    try:
        wait_for_server()

        # 3. Dynamically load LoRA adapter (this exercises _load_lora_config)
        print(f"[3/5] Loading LoRA adapter '{LORA_NAME}' from {lora_path}")
        resp = api_post(
            "load_lora_adapter",
            {
                "lora_name": LORA_NAME,
                "lora_path": lora_path,
            },
        )
        print(f"       Response: {resp}")

        # 4. Run inference with LoRA
        print("[4/5] Running chat completion with LoRA adapter...")
        resp = api_post(
            "chat/completions",
            {
                "model": LORA_NAME,
                "messages": [
                    {"role": "user", "content": "What is your name?"},
                ],
                "max_tokens": 64,
            },
        )
        content = resp["choices"][0]["message"]["content"]
        print(f"       Model response: {content}")

        # 5. Verify response
        print("[5/5] Verifying response...")
        assert len(content) > 0, "Empty response from model"
        print("[PASS] Integration test passed.")

    finally:
        print("Shutting down server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_proc.wait()
        print("Server stopped.")


if __name__ == "__main__":
    main()
