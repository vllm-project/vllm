# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
RLHF weight syncing against a `vllm serve` HTTP server, using NCCL for the
data plane.

This is the standard shape for weight transfer: the inference side is a plain
`vllm serve` process configured with nothing but a backend name, and the trainer
drives it through `HTTPVLLMWeightSyncClient`. The trainer never touches vLLM
internals — it only holds a URL.

  * OpenAI-compatible API for inference requests
  * HTTP endpoints for the weight-transfer control plane
  * NCCL for the weight data plane

3-GPU layout (single node):
  Inference — GPUs 0-1, `vllm serve` with TP=2 and fp8 quantization
  Training  — GPU 2, a plain Hugging Face model in this process

The trainer holds bf16 weights while the server runs fp8: weights are quantized
on the fly as the server loads them, so the two sides do not need matching
dtypes.

The script launches the server itself (and prints the exact command it runs, so
you can copy it), then:

  1. Generate over HTTP → gibberish (server started with dummy weights).
  2. Pause generation, sync real weights trainer → server over NCCL, resume.
  3. Generate again → sensible output.

Run:
    $ python examples/rl/rlhf_http_nccl.py
"""

import os
import subprocess
import sys
import time

import requests
import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM

from vllm.distributed.weight_transfer import (
    HTTPVLLMWeightSyncClient,
    ModuleSource,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerInitInfo
from vllm.utils.network_utils import get_ip, get_open_port

MODEL_NAME = "facebook/opt-125m"

SERVER_PORT = 8000
BASE_URL = f"http://localhost:{SERVER_PORT}"

INFERENCE_TP_SIZE = 2
# Physical GPUs for the server; the trainer takes the next one. `--device-ids`
# pins placement without CUDA_VISIBLE_DEVICES, so both sides keep full topology
# visibility and the trainer can address its GPU by its real index.
SERVER_DEVICE_IDS = "0,1"
TRAINER_DEVICE = "cuda:2"

PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


def start_vllm_server() -> subprocess.Popen:
    """Spawn `vllm serve` and block until it is healthy."""
    serve_args = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--tensor-parallel-size",
        str(INFERENCE_TP_SIZE),
        "--device-ids",
        SERVER_DEVICE_IDS,
        "--quantization",
        "fp8",
        "--enforce-eager",
        "--load-format",
        "dummy",
        "--port",
        str(SERVER_PORT),
        "--weight-transfer-config",
        '{"backend": "nccl"}',
    ]
    env = os.environ.copy()
    # Exposes the weight-transfer and pause/resume endpoints.
    env["VLLM_SERVER_DEV_MODE"] = "1"
    print(f"[server] Launching: {' '.join(serve_args)}")
    proc = subprocess.Popen(
        serve_args,
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
        start_new_session=True,
    )

    deadline = time.monotonic() + 900
    while True:
        if proc.poll() is not None:
            raise RuntimeError("vLLM server exited before becoming ready.")
        try:
            if requests.get(f"{BASE_URL}/health", timeout=5).status_code == 200:
                break
        except requests.RequestException:
            pass
        if time.monotonic() > deadline:
            raise RuntimeError("vLLM server failed to start in time.")
        time.sleep(2)
    print("[server] Ready.")
    return proc


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
    requests.post(f"{base_url}/pause", timeout=60).raise_for_status()


def resume_generation(base_url: str) -> None:
    """Resume generation via HTTP endpoint."""
    requests.post(f"{base_url}/resume", timeout=60).raise_for_status()


def get_world_size(base_url: str) -> int:
    """Get the number of inference workers from the vLLM server."""
    response = requests.get(f"{base_url}/get_world_size", timeout=10)
    response.raise_for_status()
    return response.json()["world_size"]


def print_generations(label: str, prompts: list[str], outputs: list[str]) -> None:
    print("-" * 50)
    print(label)
    print("-" * 50)
    for prompt, generated_text in zip(prompts, outputs):
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)


def main():
    server_proc = start_vllm_server()
    try:
        # The trainer sits on the GPU after the server's, and is NCCL rank 0.
        torch.accelerator.set_device_index(TRAINER_DEVICE)

        print(f"[trainer] Loading training model: {MODEL_NAME} on {TRAINER_DEVICE}")
        train_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.bfloat16
        )
        train_model.to(TRAINER_DEVICE)

        client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="EMPTY")

        # Generate with dummy weights — expect nonsense.
        outputs = generate_completions(client, MODEL_NAME, PROMPTS)
        print_generations("BEFORE weight sync (dummy weights):", PROMPTS, outputs)

        # The transfer NCCL group is the trainer (rank 0) plus every inference
        # worker, so ask the server how many workers it has rather than
        # hard-coding it.
        world_size = get_world_size(BASE_URL) + 1
        master_address = get_ip()
        master_port = get_open_port()
        print(
            f"[transfer] Rendezvous at {master_address}:{master_port}, "
            f"world_size={world_size} (1 trainer + {world_size - 1} vLLM workers)"
        )

        # `trainer_init` drives the full handshake: it calls the server's
        # init_weight_transfer_engine (via the HTTP client) on a worker thread
        # while opening the trainer's NCCL endpoint, so both ends rendezvous
        # together — no manual threading needed here.
        engine = WeightTransferTrainerFactory.trainer_init(
            init_info=NCCLTrainerInitInfo(
                master_address=master_address,
                master_port=master_port,
                world_size=world_size,
                rank=0,  # single-GPU trainer is the sole (sender) rank
                packed=True,
            ),
            client=HTTPVLLMWeightSyncClient(BASE_URL),
            source=ModuleSource(train_model),
        )

        pause_generation(BASE_URL)

        # One call drives start_weight_update / update_weights /
        # finish_weight_update over HTTP, concurrent with the NCCL broadcast.
        print("[sync] Broadcasting weights via NCCL...")
        engine.send_weights()
        print("[sync] Weight broadcast complete.")

        resume_generation(BASE_URL)

        # Generate with the synced weights — expect sensible output.
        outputs_updated = generate_completions(client, MODEL_NAME, PROMPTS)
        print_generations("AFTER weight sync (real weights):", PROMPTS, outputs_updated)
    finally:
        print("[server] Shutting down...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server_proc.kill()


if __name__ == "__main__":
    main()
