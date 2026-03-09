# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Demonstrates reinforcement learning from human feedback (RLHF) using vLLM
via HTTP API, with IPC-based weight syncing APIs.

Unlike rlhf_nccl.py which uses NCCL and can use separate GPUs, this script
uses CUDA IPC which requires the training model and vLLM server to be on the
same GPU. Memory must be carefully managed to fit both models.

Unlike rlhf.py which creates a vLLM instance programmatically, this script
assumes you have already started a vLLM server using `vllm serve`. It uses:
- OpenAI-compatible API for inference requests
- HTTP endpoints for weight transfer control plane
- CUDA IPC for actual weight data transfer

Prerequisites:
    Start a vLLM server with weight transfer enabled and reduced GPU memory
    utilization to leave room for the training model:

    $ VLLM_SERVER_DEV_MODE=1 VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
        vllm serve facebook/opt-125m --enforce-eager \
        --weight-transfer-config '{"backend": "ipc"}' \
        --load-format dummy \
        --gpu-memory-utilization 0.5

    Then run this script:

    $ python rlhf_http_ipc.py

The example performs the following steps:

* Load the training model on GPU 0 (same GPU as the vLLM server).
* Generate text using the vLLM server via OpenAI-compatible API. The output
  is expected to be nonsense because the server is initialized with dummy weights.
* Initialize weight transfer via HTTP endpoint (no-op for IPC).
* Broadcast the real weights from the training model to the vLLM server
  using CUDA IPC handles.
* Generate text again to show normal output after the weight update.
"""

import os

import requests
import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM

from vllm.distributed.weight_transfer.ipc_engine import (
    IPCTrainerSendWeightsArgs,
    IPCWeightTransferEngine,
)

BASE_URL = "http://localhost:8000"
MODEL_NAME = "facebook/opt-125m"

# Enable insecure serialization for IPC handle serialization
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"


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


def init_weight_transfer_engine(base_url: str) -> None:
    """Initialize weight transfer via HTTP endpoint (no-op for IPC)."""
    url = f"{base_url}/init_weight_transfer_engine"
    payload = {"init_info": dict()}
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
    # IPC requires the training model to be on the same GPU as the vLLM server
    # The server should be started on GPU 0 with reduced memory utilization
    device = "cuda:0"
    torch.cuda.set_device(device)

    # Load the training model on the same GPU as the server
    # Use bfloat16 to reduce memory footprint
    print(f"Loading training model: {MODEL_NAME} on {device}")
    print(
        "Note: Ensure the vLLM server was started with --gpu-memory-utilization 0.5 "
        "or lower to leave room for the training model."
    )
    train_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
    train_model.to(device)
    train_model.eval()  # Set to eval mode to save memory

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

    print("Initializing weight transfer (IPC backend)...")

    # Initialize weight transfer on vLLM server (no-op for IPC, but still required)
    init_weight_transfer_engine(BASE_URL)

    # Pause generation before weight sync
    pause_generation(BASE_URL)

    # Broadcast weights via IPC handles using HTTP mode
    print("Broadcasting weights via CUDA IPC (HTTP)...")
    trainer_args = IPCTrainerSendWeightsArgs(mode="http", url=BASE_URL)
    IPCWeightTransferEngine.trainer_send_weights(
        iterator=train_model.named_parameters(),
        trainer_args=trainer_args,
    )

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

    # Note: The training model and IPC handles remain in memory.
    # In a real RLHF training loop, you would update the training model
    # and create new IPC handles for each weight update.


if __name__ == "__main__":
    main()
