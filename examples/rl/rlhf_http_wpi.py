# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Demonstrates reinforcement learning from human feedback (RLHF) using vLLM
via HTTP API, with WPI (Weight Propagation Interface) weight syncing APIs.

Unlike rlhf.py which creates a vLLM instance programmatically, this script
assumes you have already started a vLLM server using `vllm serve`. It uses:
- OpenAI-compatible API for inference requests
- HTTP endpoints for weight transfer control plane
- WPI (Weight Propagation Interface) for actual weight data transfer

Prerequisites:
    1. Ensure the WPI driver is running on the node(s).
    2. Start a vLLM server with WPI weight transfer enabled:

    $ VLLM_SERVER_DEV_MODE=1 vllm serve facebook/opt-125m \
        --enforce-eager \
        --weight-transfer-config '{"backend": "wpi"}' \
        --load-format dummy

    Then run this script:

    $ python rlhf_http_wpi.py
"""

import requests
import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM

from vllm.distributed.weight_transfer.wpi_engine import (
    WPITrainerSendWeightsArgs,
    WPIWeightTransferEngine,
)

BASE_URL = "http://localhost:8000"
MODEL_NAME = "facebook/opt-125m"


def generate_completions(client: OpenAI, model: str, prompts: list[str]) -> list[str]:
    """Generate completions using the OpenAI-compatible API."""
    results = []
    for prompt in prompts:
        response = client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=32,
            temperature=0,
        )
        results.append(response.choices[0].text)
    return results


def init_weight_transfer_engine(
    base_url: str,
    buffer_id: str,
    buffer_size_bytes: int,
    socket_dir: str,
    driver_port: int,
) -> None:
    """Initialize weight transfer via HTTP endpoint."""
    url = f"{base_url}/init_weight_transfer_engine"
    payload = {
        "init_info": dict(
            buffer_id=buffer_id,
            buffer_size_bytes=buffer_size_bytes,
            socket_dir=socket_dir,
            driver_port=driver_port,
            shard_index=-1,
            total_shards=0,
        )
    }
    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()


def pause_generation(base_url: str) -> None:
    """Pause generation via HTTP endpoint."""
    url = f"{base_url}/pause"
    response = requests.post(url, timeout=60)
    response.raise_for_status()


def resume_generation(base_url: str) -> None:
    """Resume generation via HTTP endpoint."""
    url = f"{base_url}/resume"
    response = requests.post(url, timeout=60)
    response.raise_for_status()


def get_world_size(base_url: str) -> int:
    """Get world size from the vLLM server."""
    url = f"{base_url}/get_world_size"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()["world_size"]


def main():
    # Get the inference world size from the vLLM server
    inference_world_size = get_world_size(BASE_URL)
    device = f"cuda:{inference_world_size}"
    torch.accelerator.set_device_index(device)

    # Load the training model
    print(f"Loading training model: {MODEL_NAME}")
    train_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
    train_model.to(device)

    # Create OpenAI client pointing to the vLLM server
    client = OpenAI(
        base_url=f"{BASE_URL}/v1",
        api_key="EMPTY",  # vLLM doesn't require an API key by default
    )

    # Test prompts
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Generate text before weight update. The output is expected to be nonsense
    # because the server is initialized with dummy weights.
    print("-" * 50)
    print("Generating text BEFORE weight update (expect nonsense):")
    print("-" * 50)
    outputs = generate_completions(client, MODEL_NAME, prompts)
    for prompt, generated_text in zip(prompts, outputs):
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)

    # WPI specific setup
    buffer_id = "vllm-weights"
    socket_dir = "/run/wpi/sockets"
    driver_port = 50051

    # Calculate model size in bytes
    total_model_bytes = sum(p.numel() * p.element_size() for p in train_model.parameters())
    print(f"Calculated model size: {total_model_bytes} bytes")

    print(f"Initializing weight transfer on server...")

    # Initialize weight transfer on vLLM server
    init_weight_transfer_engine(
        BASE_URL, buffer_id, total_model_bytes, socket_dir, driver_port
    )

    # Initialize WPI client on trainer side
    print("Initializing trainer WPI context...")
    ctx = WPIWeightTransferEngine.trainer_init(
        dict(
            buffer_id=buffer_id,
            buffer_size_bytes=total_model_bytes,
            socket_dir=socket_dir,
            driver_port=driver_port,
        ),
        target_node_ids=["127.0.0.1"],  # Assuming local testing
    )

    # Pause generation before weight sync
    pause_generation(BASE_URL)

    # Broadcast all weights from trainer to vLLM workers
    print("Propagating weights via WPI...")
    param_iter = ((n, p) for n, p in train_model.named_parameters())
    args = WPITrainerSendWeightsArgs(
        mode="http",
        url=BASE_URL,
        trainer_ctx=ctx,
    )
    
    # This will pack weights, trigger WPI propagate, and call /update_weights on server
    WPIWeightTransferEngine.trainer_send_weights(param_iter, args)

    # Resume generation after weight sync
    resume_generation(BASE_URL)

    # Generate text after weight update. The output is expected to be normal
    # because the real weights are now loaded.
    print("-" * 50)
    print("Generating text AFTER weight update:")
    print("-" * 50)
    outputs_updated = generate_completions(client, MODEL_NAME, prompts)
    for prompt, generated_text in zip(prompts, outputs_updated):
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)


if __name__ == "__main__":
    main()
