# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Demonstrates reinforcement learning from human feedback (RLHF) using vLLM
via HTTP API, with new weight syncing APIs.

Unlike rlhf.py which creates a vLLM instance programmatically, this script
assumes you have already started a vLLM server using `vllm serve`. It uses:
- OpenAI-compatible API for inference requests
- HTTP endpoints for weight transfer control plane
- NCCL for actual weight data transfer

Prerequisites:
    Start a vLLM server with weight transfer enabled:

    $ vllm serve facebook/opt-125m \
        --enforce-eager \
        --weight-transfer-backend nccl \
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

from dataclasses import asdict

import requests
import torch
from openai import OpenAI
from rlhf_utils import stateless_init_process_group
from transformers import AutoModelForCausalLM

from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLInitInfo,
    NCCLUpdateInfo,
    NCCLWeightTransferEngine,
)
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


def init_weight_transfer(
    base_url: str,
    master_address: str,
    master_port: int,
    rank_offset: int,
    world_size: int,
) -> None:
    """Initialize weight transfer via HTTP endpoint."""
    url = f"{base_url}/init_weight_transfer"
    payload = {
        "init_info": asdict(
            NCCLInitInfo(
                master_address=master_address,
                master_port=master_port,
                rank_offset=rank_offset,
                world_size=world_size,
            )
        )
    }
    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()


def update_weights(
    base_url: str,
    names: list[str],
    dtype_names: list[str],
    shapes: list[list[int]],
    packed: bool = False,
) -> None:
    """Update weights via HTTP endpoint."""
    url = f"{base_url}/update_weights"
    payload = {
        "update_info": asdict(
            NCCLUpdateInfo(
                names=names,
                dtype_names=dtype_names,
                shapes=shapes,
                packed=packed,
            )
        )
    }
    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()


def finalize_weight_update(base_url: str) -> None:
    """Finalize weight update via HTTP endpoint."""
    url = f"{base_url}/finalize_weight_update"
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
    torch.cuda.set_device(device)

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
    # vLLM server. The trainer is rank 0, vLLM worker(s) start at rank_offset.
    master_address = get_ip()
    master_port = get_open_port()
    rank_offset = 1

    print(f"Initializing weight transfer: master={master_address}:{master_port}")

    # Initialize weight transfer on vLLM server (this is async, server will
    # wait for NCCL connection)
    import threading

    init_thread = threading.Thread(
        target=init_weight_transfer,
        args=(BASE_URL, master_address, master_port, rank_offset, world_size),
    )
    init_thread.start()

    # Initialize NCCL process group on trainer side
    model_update_group = stateless_init_process_group(
        master_address, master_port, 0, world_size, torch.device(device)
    )

    # Wait for init_weight_transfer to complete
    init_thread.join()

    # Collect weight metadata for the update request
    names = []
    dtype_names = []
    shapes = []
    for name, p in train_model.named_parameters():
        names.append(name)
        dtype_names.append(str(p.dtype).split(".")[-1])
        shapes.append(list(p.shape))

    # Start the update_weights call in a separate thread since it will block
    # waiting for NCCL broadcasts
    # packed=True enables efficient batched tensor broadcasting
    update_thread = threading.Thread(
        target=update_weights,
        args=(BASE_URL, names, dtype_names, shapes, True),  # packed=True
    )
    update_thread.start()

    # Broadcast all weights from trainer to vLLM workers
    print("Broadcasting weights via NCCL...")
    NCCLWeightTransferEngine.trainer_broadcast_weights(
        iterator=train_model.named_parameters(),
        group=model_update_group,
        packed=True,
    )

    # Wait for update_weights to complete
    update_thread.join()

    # Finalize the weight update (processes weights for quantization/kernel format)
    finalize_weight_update(BASE_URL)

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
