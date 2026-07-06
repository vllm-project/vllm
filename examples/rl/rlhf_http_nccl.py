# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Demonstrates reinforcement learning from human feedback (RLHF) using vLLM
via HTTP API, with native weight syncing APIs.

Unlike rlhf.py which creates a vLLM instance programmatically, this script
assumes you have already started a vLLM server using `vllm serve`. It uses:
- OpenAI-compatible API for inference requests
- HTTP endpoints for weight transfer control plane
- NCCL for actual weight data transfer

Prerequisites:
    Start a vLLM server with weight transfer enabled:

    $ VLLM_SERVER_DEV_MODE=1 vllm serve facebook/opt-125m \
        --enforce-eager \
        --weight-transfer-config '{"backend": "nccl"}' \
        --load-format dummy

    Then run this script:

    $ python rlhf_http.py

The example performs the following steps:

* Load the training model on GPU 0.
* Generate text using the vLLM server via OpenAI-compatible API. The output
  is expected to be nonsense because the server is initialized with dummy weights.
* Initialize weight transfer via HTTP endpoint.
* Broadcast the real weights from the training model to the vLLM server
  using NCCL.
* Generate text again to show normal output after the weight update.
"""

import requests
import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM

from vllm.config import NCCLWeightTransferConfig
from vllm.distributed.weight_transfer import (
    HTTPVLLMWeightSyncClient,
    ModuleSource,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.nccl_common import NCCLTrainerInitInfo
from vllm.utils.network_utils import get_ip, get_open_port

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
    world_size = inference_world_size + 1  # +1 for the trainer
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

    # Set up the communication channel between the training process and the
    # vLLM server. The trainer is rank 0, vLLM worker(s) start after it.
    master_address = get_ip()
    master_port = get_open_port()

    print(f"Initializing weight transfer: master={master_address}:{master_port}")

    # The trainer engine owns the full handshake. `trainer_init` kicks off the
    # server's init_weight_transfer_engine (via the HTTP client) on a worker
    # thread while opening the trainer's NCCL endpoint, so both ends rendezvous
    # together — no manual threading needed here.
    engine = WeightTransferTrainerFactory.trainer_init(
        backend="nccl",
        config=NCCLWeightTransferConfig(packed=True),
        init_info=NCCLTrainerInitInfo(
            master_address=master_address,
            master_port=master_port,
            world_size=world_size,
            rank=0,  # single-GPU trainer is the sole (sender) rank
        ),
        client=HTTPVLLMWeightSyncClient(BASE_URL),
        source=ModuleSource(train_model),
    )

    # Pause generation before weight sync
    pause_generation(BASE_URL)

    # One call drives start_weight_update / update_weights / finish_weight_update
    # over HTTP, concurrent with the NCCL broadcast.
    print("Broadcasting weights via NCCL...")
    engine.send_weights()

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
