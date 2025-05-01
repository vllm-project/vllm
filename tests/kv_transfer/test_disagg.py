# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import time
from subprocess import Popen

import pytest
import requests
import torch


# Fixture to set up environment variables and teardown servers after tests
@pytest.fixture(scope="module", autouse=True)
def setup_servers():
    if torch.cuda.device_count() < 2:
        pytest.skip("Skipping test: fewer than 2 GPUs available")

    # Set up environment variables
    VLLM_HOST_IP = subprocess.check_output("hostname -I | awk '{print $1}'",
                                           shell=True).decode().strip()
    os.environ["VLLM_HOST_IP"] = VLLM_HOST_IP

    # Start prefill instance
    prefill_cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        "meta-llama/Llama-3.2-1B-Instruct",
        "--port",
        "8100",
        "--gpu-memory-utilization",
        "0.5",
        "--max-model-len",
        "1000",
        "--kv-transfer-config",
        '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer",'\
        '"kv_rank":0,"kv_parallel_size":2}',
    ]
    prefill_env = os.environ.copy()
    prefill_env["CUDA_VISIBLE_DEVICES"] = "0"
    prefill_proc = Popen(prefill_cmd, env=prefill_env)

    # Start decode instance
    decode_cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        "meta-llama/Llama-3.2-1B-Instruct",
        "--port",
        "8200",
        "--gpu-memory-utilization",
        "0.5",
        "--max-model-len",
        "1000",
        "--kv-transfer-config",
        '{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer",'\
        '"kv_rank":1,"kv_parallel_size":2}',
    ]
    decode_env = os.environ.copy()
    decode_env["CUDA_VISIBLE_DEVICES"] = "1"
    decode_proc = Popen(decode_cmd, env=decode_env)

    # Wait for servers to be ready
    assert wait_for_server(8100), "Prefill server did not start in time"
    assert wait_for_server(8200), "Decode server did not start in time"

    # Yield to the test function and handle teardown after tests
    yield

    # Cleanup: kill the processes
    prefill_proc.terminate()
    decode_proc.terminate()

    # Additional cleanup if needed
    prefill_proc.wait()
    decode_proc.wait()


# Helper function to wait for server
def wait_for_server(port, timeout=240):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://localhost:{port}/v1/completions")
            if response.status_code in [200, 405]:
                return True
        except requests.ConnectionError:
            time.sleep(1)
    return False


# Test function to send curl requests and validate responses
@pytest.mark.parametrize("prompt", ["San Francisco is a", "Santa Clara is a"])
def test_disaggregated_prefilling(prompt):
    # Send to prefill
    response = requests.post("http://localhost:8100/v1/completions",
                             headers={"Content-Type": "application/json"},
                             json={
                                 "model": "meta-llama/Llama-3.2-1B-Instruct",
                                 "prompt": prompt,
                                 "max_tokens": 1,
                                 "temperature": 0
                             })
    assert response.status_code == 200

    # Send to decode
    response = requests.post("http://localhost:8200/v1/completions",
                             headers={"Content-Type": "application/json"},
                             json={
                                 "model": "meta-llama/Llama-3.2-1B-Instruct",
                                 "prompt": prompt,
                                 "max_tokens": 10,
                                 "temperature": 0
                             })
    assert response.status_code == 200
