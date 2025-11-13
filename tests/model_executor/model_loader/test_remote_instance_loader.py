# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Tests for the remote instance model loader.

To run these tests:
1. Install test dependencies:
   uv pip install -r requirements/common.txt -r requirements/dev.txt
   --torch-backend=auto
   uv pip install pytest pytest-asyncio

2. Run the tests:
   pytest -s -v tests/model_executor/model_loader/test_remote_instance_loader.py

Note: This test is marked as skip because it requires:
- Multiple GPUs (at least 8 GPUs for 2x2 TP/PP configuration for both seed
  and client instances)
- Coordinated seed and client servers
- Proper setup of environment variables
- Network communication between servers
"""

from http import HTTPStatus

import pytest
import requests
from huggingface_hub import snapshot_download

from tests.utils import RemoteOpenAIServer

# Test prompts
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


@pytest.fixture(scope="module")
def llama_3p2_1b_files():
    """Download the Llama-3.2-1B-Instruct model files for testing."""
    input_dir = snapshot_download(
        "meta-llama/Llama-3.2-1B-Instruct", ignore_patterns=["*.bin*", "original/*"]
    )
    yield input_dir


def test_remote_instance_loader_end_to_end(llama_3p2_1b_files, num_gpus_available):
    """
    End-to-end test for the remote instance loader.

    This test simulates the manual testing procedure:
    1. Start a seed server (source of weights)
    2. Start a client server (loads weights from seed server)
    3. Compare outputs from both servers

    Note: This test is marked as skip because it requires:
    - Multiple GPUs (at least 8 GPUs for 2x2 TP/PP configuration for both
      seed and client instances)
    - Coordinated seed and client servers
    - Proper setup of environment variables
    - Network communication between servers
    """
    # Need at least 8 GPUs (4 for seed instance + 4 for client instance)
    if num_gpus_available < 8:
        pytest.skip(
            "Not enough GPUs for 2x2 TP/PP configuration for both seed and "
            "client instances (requires 8 GPUs)"
        )

    input_dir = llama_3p2_1b_files
    seed_port = 12346
    client_port = 12347
    gpu_memory_utilization = 0.8

    # Server arguments for both seed and client instances
    common_args = [
        "--tensor-parallel-size",
        "2",
        "--pipeline-parallel-size",
        "2",
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-model-len",
        "1024",
        "--enforce-eager",
    ]

    # Run seed server (source of weights)
    seed_args = [
        "--host",
        "127.0.0.1",
        "--port",
        str(seed_port),
        *common_args,
    ]

    with RemoteOpenAIServer(input_dir, seed_args, auto_port=False) as seed_server:
        # Check if seed server is running
        response = requests.get(seed_server.url_for("health"))
        assert response.status_code == HTTPStatus.OK

        # Run client server (loads weights from seed server)
        # Set environment variables for remote instance loading
        # Use different GPUs for client instance to avoid conflict with seed instance
        client_env_dict = {
            "REMOTE_INSTANCE_IP": "127.0.0.1",
            "REMOTE_INSTANCE_SERVER_PORT": str(seed_port),
            "REMOTE_INSTANCE_PORTS": "[50000,50001,50002,50003]",
            "CUDA_VISIBLE_DEVICES": "4,5,6,7",  # Use different GPUs for client
        }

        client_args = [
            "--host",
            "127.0.0.1",
            "--port",
            str(client_port),
            "--load-format",
            "remote_instance",
            *common_args,
        ]

        with RemoteOpenAIServer(
            input_dir, client_args, env_dict=client_env_dict, auto_port=False
        ) as client_server:
            # Check if client server is running
            response = requests.get(client_server.url_for("health"))
            assert response.status_code == HTTPStatus.OK

            # Get clients for both servers
            seed_client = seed_server.get_client()
            client_client = client_server.get_client()

            # Get the model name from the seed server
            seed_models = seed_client.models.list()
            seed_model_name = seed_models.data[0].id

            # Get the model name from the client server
            client_models = client_client.models.list()
            client_model_name = client_models.data[0].id

            # Generate outputs from both servers and compare
            for prompt in prompts:
                # Generate from seed server
                seed_response = seed_client.completions.create(
                    model=seed_model_name,
                    prompt=prompt,
                    max_tokens=256,
                    temperature=0.0,
                )
                seed_text = seed_response.choices[0].text

                # Generate from client server
                client_response = client_client.completions.create(
                    model=client_model_name,
                    prompt=prompt,
                    max_tokens=256,
                    temperature=0.0,
                )
                client_text = client_response.choices[0].text

                # Compare outputs
                assert seed_text == client_text, (
                    f"Outputs from seed and client servers should be identical.\n"
                    f"Prompt: {prompt}\n"
                    f"Seed output: {seed_text}\n"
                    f"Client output: {client_text}"
                )
