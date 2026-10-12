# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RLHF weight syncing against a `vllm serve` HTTP server, using CUDA IPC for the
data plane.

  * OpenAI-compatible API for inference requests
  * HTTP endpoints for the weight-transfer control plane
  * CUDA IPC handles for the weight data plane

1-GPU layout (single node): IPC shares GPU memory directly, so the server (TP=1)
and the training model both live on GPU 0. The server is started with
`--gpu-memory-utilization 0.5` to leave room for the training model.

The script starts the server itself, then:

  1. Generate over HTTP → gibberish (server started with dummy weights).
  2. Pause generation, sync real weights trainer → server over IPC, resume.
  3. Generate again → sensible output.

IPC handles are pickled for HTTP transport, so both sides need
`VLLM_ALLOW_INSECURE_SERIALIZATION=1`; this script sets it for itself and for
the server it spawns.

Run:
    $ python examples/rl/rlhf_http_ipc.py
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
from vllm.distributed.weight_transfer.ipc_engine import IPCTrainerInitInfo

MODEL_NAME = "facebook/opt-125m"

SERVER_PORT = 8000
BASE_URL = f"http://localhost:{SERVER_PORT}"

# IPC requires colocation: the server and the training model share this GPU.
SERVER_DEVICE_IDS = "0"
TRAINER_DEVICE = "cuda:0"
# Leave room on the shared GPU for the training model.
SERVER_GPU_MEMORY_UTILIZATION = 0.5

# Needed to (de)serialize IPC handles across the HTTP boundary.
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

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
        "1",
        "--device-ids",
        SERVER_DEVICE_IDS,
        "--enforce-eager",
        "--load-format",
        "dummy",
        "--gpu-memory-utilization",
        str(SERVER_GPU_MEMORY_UTILIZATION),
        "--port",
        str(SERVER_PORT),
        "--weight-transfer-config",
        '{"backend": "ipc"}',
    ]
    env = os.environ.copy()
    # Exposes the weight-transfer and pause/resume endpoints.
    env["VLLM_SERVER_DEV_MODE"] = "1"
    env["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
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
        # The training model must sit on the same physical GPU as the server.
        torch.accelerator.set_device_index(TRAINER_DEVICE)

        print(f"[trainer] Loading training model: {MODEL_NAME} on {TRAINER_DEVICE}")
        train_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.bfloat16
        )
        train_model.to(TRAINER_DEVICE)
        train_model.eval()  # eval mode to save memory on the shared GPU

        client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="EMPTY")

        # Generate with dummy weights — expect nonsense.
        outputs = generate_completions(client, MODEL_NAME, PROMPTS)
        print_generations("BEFORE weight sync (dummy weights):", PROMPTS, outputs)

        # IPC needs no data-plane rendezvous; `trainer_init` only ships the
        # `packed` flag, which the server must decode with.
        print("[transfer] Initializing IPC weight transfer...")
        engine = WeightTransferTrainerFactory.trainer_init(
            init_info=IPCTrainerInitInfo(rank=0, packed=False),  # rank 0 = sender
            client=HTTPVLLMWeightSyncClient(BASE_URL),
            source=ModuleSource(train_model),
        )

        pause_generation(BASE_URL)

        # Drives start_weight_update / update_weights / finish_weight_update.
        print("[sync] Sharing weights via CUDA IPC...")
        engine.send_weights()
        print("[sync] Weight transfer complete.")

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
