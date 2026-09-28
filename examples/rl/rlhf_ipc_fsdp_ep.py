# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
RLHF with FSDP2 training and vLLM expert-parallel inference using **CUDA IPC**
weight transfer and **packed** tensors.

Multi-rank version of `rlhf_http_ipc.py`: the trainer is 4 FSDP2 Ray actors
colocated with a data-parallel `vllm serve` on the same 4 physical GPUs.

4-GPU layout (single node), all colocated:
  Training  — 4 GPUs, PyTorch FSDP2 (fully_shard), as Ray actors
  Inference — the same 4 GPUs, `vllm serve --data-parallel-size 4 -tp 1
              --enable-expert-parallel` (EP_SIZE = TP x DP = 4)

IPC requires the trainer and the server to sit on the same GPUs, so the script
reserves the training GPUs through Ray first, asks Ray which ones it got, and
pins the server to exactly those with `--device-ids`.

Both sides share each GPU, so the server is capped with
`--gpu-memory-utilization` and its weights are moved aside for the transfer:

  1. `/sleep?level=1`             — offload server weights to CPU, drop KV cache
  2. `/wake_up?tags=weights`      — weights back on GPU, KV cache still free
  3. packed IPC transfer          — overwrite weights with room to spare
  4. `/wake_up?tags=kv_cache&tags=scheduling` — re-allocate KV cache, resume

Every FSDP rank builds an ``IPCTrainerWeightTransferEngine`` (via ``trainer_init``)
and calls ``send_weights()``; all ranks join the IPC handle all-gather, and only
rank 0 (the sender) ships the merged handles and drives the server.

This example was run on 4xH100.

Run:
    $ python examples/rl/rlhf_ipc_fsdp_ep.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import ray
import requests
import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download
from openai import OpenAI
from torch.distributed.fsdp import fully_shard
from transformers import AutoModelForCausalLM

from vllm.distributed.weight_transfer import (
    HTTPVLLMWeightSyncClient,
    ModuleSource,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.ipc_engine import IPCTrainerInitInfo
from vllm.utils.network_utils import get_ip, get_open_port

MODEL_NAME = "Qwen/Qwen3-30B-A3B"
SERVED_MODEL_NAME = "policy"

FSDP_WORLD_SIZE = 4
INFERENCE_TP_SIZE = 1
INFERENCE_DP_SIZE = 4

# Packed IPC transfer with a 1 GB chunk buffer.
PACKED = True
PACKED_BUFFER_SIZE_BYTES = 1024 * 1024 * 1024

# The server shares each GPU with a training rank, so cap what it reserves.
SERVER_GPU_MEMORY_UTILIZATION = 0.35

SERVER_PORT = 8000
BASE_URL = f"http://localhost:{SERVER_PORT}"

PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


@ray.remote(num_gpus=1)
class FSDPTrainWorker:
    """One FSDP2 worker per GPU; colocated with one vLLM DP rank."""

    def __init__(
        self,
        model_name: str,
        rank: int,
        fsdp_world_size: int,
        fsdp_master_addr: str,
        fsdp_master_port: int,
    ):
        self.rank = rank

        os.environ["MASTER_ADDR"] = fsdp_master_addr
        os.environ["MASTER_PORT"] = str(fsdp_master_port)

        dist.init_process_group(backend="nccl", rank=rank, world_size=fsdp_world_size)
        torch.accelerator.set_device_index(0)

        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        )

        for layer in model.model.layers:
            fully_shard(layer)
        fully_shard(model)

        self.model = model

    def get_rank(self):
        return self.rank

    def get_gpu_ids(self):
        """Physical GPU id(s) Ray assigned to this worker."""
        return ray.get_gpu_ids()

    def setup_engine(self, base_url: str):
        """Build the trainer IPC engine. Called on every FSDP rank."""
        self.engine = WeightTransferTrainerFactory.trainer_init(
            init_info=IPCTrainerInitInfo(
                rank=self.rank,  # FSDP rank; sender is 0
                packed=PACKED,
                packed_buffer_size_bytes=PACKED_BUFFER_SIZE_BYTES,
            ),
            client=HTTPVLLMWeightSyncClient(base_url),
            source=ModuleSource(self.model),
        )

    def gather_and_broadcast_weights_ipc(self):
        """Send the current weights to vLLM. Called on every FSDP rank."""
        self.engine.send_weights()


def start_vllm_server(model_path: str, device_ids: str) -> subprocess.Popen:
    """Spawn a `vllm serve` HTTP server (DP+EP) pinned to `device_ids`."""
    serve_args = [
        "vllm",
        "serve",
        model_path,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--tensor-parallel-size",
        str(INFERENCE_TP_SIZE),
        "--data-parallel-size",
        str(INFERENCE_DP_SIZE),
        "--enable-expert-parallel",
        # Pins the server to the same physical GPUs as the training ranks.
        "--device-ids",
        device_ids,
        "--enable-sleep-mode",
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
    env["VLLM_SERVER_DEV_MODE"] = "1"  # exposes weight-transfer + sleep endpoints
    env["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"  # IPC handles over HTTP
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    print(f"[server] Launching: {' '.join(serve_args)} (GPUs {device_ids})")
    proc = subprocess.Popen(
        serve_args,
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
        start_new_session=True,
    )

    # Wait for the server to come up (model load can take a while).
    deadline = time.monotonic() + 1800
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


def generate_completions(client: OpenAI, prompts: list[str]) -> list[str]:
    """Generate completions via the OpenAI HTTP API."""
    results = []
    for prompt in prompts:
        response = client.completions.create(
            model=SERVED_MODEL_NAME,
            prompt=prompt,
            max_tokens=32,
            temperature=0,
        )
        results.append(response.choices[0].text)
    return results


def sleep_engine(base_url: str, level: int) -> None:
    """Put the engine to sleep (level 1 offloads weights, drops KV cache)."""
    response = requests.post(f"{base_url}/sleep", params={"level": level}, timeout=600)
    response.raise_for_status()


def wake_up_engine(base_url: str, tags: list[str] | None = None) -> None:
    """Wake the engine, optionally only for specific memory tags."""
    params = [("tags", tag) for tag in tags] if tags else None
    response = requests.post(f"{base_url}/wake_up", params=params, timeout=600)
    response.raise_for_status()


def print_generations(label: str, prompts: list[str], outputs: list[str]) -> None:
    print("-" * 60)
    print(label)
    print("-" * 60)
    for prompt, text in zip(prompts, outputs):
        print(f"Prompt: {prompt!r}")
        print(f"Generated: {text!r}")
        print("-" * 60)


def main():
    ray.init(
        runtime_env={
            "env_vars": {
                # The trainer pickles IPC handles for the HTTP client.
                "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
            }
        }
    )

    local_model_path = snapshot_download(MODEL_NAME)
    print(f"[init] Model downloaded to {local_model_path}")

    fsdp_master_addr = get_ip()
    fsdp_master_port = get_open_port()

    # Launch the training workers first so Ray reserves their GPUs; the server
    # is then pinned to those same physical GPUs.
    fsdp_workers = [
        FSDPTrainWorker.remote(
            local_model_path,
            rank,
            FSDP_WORLD_SIZE,
            fsdp_master_addr,
            fsdp_master_port,
        )
        for rank in range(FSDP_WORLD_SIZE)
    ]
    ray.get([w.get_rank.remote() for w in fsdp_workers])
    print(f"[init] {FSDP_WORLD_SIZE} FSDP training workers ready.")

    training_gpus = sorted(
        int(g)
        for ids in ray.get([w.get_gpu_ids.remote() for w in fsdp_workers])
        for g in ids
    )
    if len(training_gpus) != INFERENCE_TP_SIZE * INFERENCE_DP_SIZE:
        raise RuntimeError(
            f"Need {INFERENCE_TP_SIZE * INFERENCE_DP_SIZE} colocated GPUs but "
            f"Ray assigned training to {training_gpus}."
        )
    device_ids = ",".join(str(g) for g in training_gpus)
    print(f"[init] Colocating training and inference on GPUs [{device_ids}].")

    server_proc = start_vllm_server(local_model_path, device_ids)
    try:
        client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="EMPTY")

        print("[generate] Generating with dummy weights...")
        outputs = generate_completions(client, PROMPTS)
        print_generations("BEFORE weight sync (dummy weights):", PROMPTS, outputs)

        # --- Weight transfer ---
        print("[transfer] Initializing IPC weight transfer (all FSDP ranks)...")
        ray.get([w.setup_engine.remote(BASE_URL) for w in fsdp_workers])

        print("[sync] Sleeping engine (offload weights + free KV cache)...")
        sleep_engine(BASE_URL, level=1)

        print("[sync] Waking weights (KV cache stays free)...")
        wake_up_engine(BASE_URL, tags=["weights"])

        print("[sync] Packed IPC transfer FSDP -> vLLM...")
        ray.get([w.gather_and_broadcast_weights_ipc.remote() for w in fsdp_workers])
        print("[sync] Weight transfer complete.")

        print("[sync] Waking KV cache + scheduling...")
        wake_up_engine(BASE_URL, tags=["kv_cache", "scheduling"])

        print("[generate] Generating with synced weights...")
        outputs_updated = generate_completions(client, PROMPTS)
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
