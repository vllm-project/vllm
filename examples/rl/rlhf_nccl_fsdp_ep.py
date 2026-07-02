# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
RLHF with FSDP2 training (4 GPUs) and vLLM expert-parallel inference (4 GPUs).

8-GPU layout:
  Training  — 4 GPUs, PyTorch FSDP2 (fully_shard), as Ray actors
  Inference — 4 GPUs, a `vllm serve` HTTP server with expert parallelism +
              data parallelism (TP=1, DP=4, enable_expert_parallel
              → EP_SIZE = TP×DP = 4)

The inference side is a standalone HTTP server (spawned by this script with
`vllm serve`), so both the weight-sync control plane (HTTP) and the NCCL data
plane run inside the rank-0 FSDP Ray actor. That lets the trainer use the
unified `TrainerWeightTransferEngine.send_weights()` with an
`HTTPVLLMWeightSyncClient` — one call drives start/update/finish on the server
concurrently with the NCCL broadcast. The 4 FSDP ranks all participate in the
incremental `full_tensor()` all-gather; only rank 0 holds the engine and
broadcasts (rank 0 is the only trainer rank in the NCCL group).

GPU split (single node): the server takes GPUs 0-3 (CUDA_VISIBLE_DEVICES), and
Ray (training) is restricted to GPUs 4-7.

Steps:
  1. Launch the vLLM HTTP server (EP+DP, dummy weights) on GPUs 0-3.
  2. Launch 4 FSDP training workers (Ray) on GPUs 4-7.
  3. Generate from prompts over HTTP → gibberish (random weights).
  4. Pause generation, transfer weights FSDP → server over NCCL, resume.
  5. Generate from prompts → sensible output (synced weights).

Assumes a single-node cluster with 8 GPUs.
"""

import json
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

from vllm.config import NCCLWeightTransferConfig
from vllm.distributed.weight_transfer import (
    HTTPVLLMWeightSyncClient,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.nccl_common import NCCLTrainerInitInfo
from vllm.utils.network_utils import get_ip, get_open_port

MODEL_NAME = "Qwen/Qwen3-30B-A3B"
SERVED_MODEL_NAME = "policy"

FSDP_WORLD_SIZE = 4
INFERENCE_TP_SIZE = 1
INFERENCE_DP_SIZE = 4

SERVER_GPUS = "0,1,2,3"  # inference (DP=4)
TRAINING_GPUS = "4,5,6,7"  # FSDP training (Ray)
SERVER_PORT = 8000
BASE_URL = f"http://localhost:{SERVER_PORT}"


@ray.remote(num_gpus=1)
class FSDPTrainWorker:
    """
    One FSDP2 training worker per GPU.  Four of these form the FSDP group.
    Rank 0 additionally drives weight transfer to the vLLM server.
    """

    def __init__(
        self,
        model_name: str,
        rank: int,
        fsdp_world_size: int,
        fsdp_master_addr: str,
        fsdp_master_port: int,
    ):
        self.rank = rank
        self.engine = None

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

        self.transfer_port = None
        self.transfer_master_address = None

    def get_rank(self):
        return self.rank

    # ---- weight-transfer setup (rank 0 only) ----

    def setup_transfer_endpoint(self):
        """Create the NCCL rendezvous endpoint for weight transfer."""
        assert self.rank == 0
        self.transfer_port = get_open_port()
        self.transfer_master_address = get_ip()
        return self.transfer_master_address, self.transfer_port

    def setup_engine(self, base_url: str, transfer_world_size: int):
        """Build the trainer engine on rank 0.

        `trainer_init` opens the trainer's rank-0 NCCL endpoint and, on a worker
        thread, calls the server's `init_weight_transfer_engine` over HTTP so
        both ends rendezvous together.
        """
        assert self.rank == 0
        self.engine = WeightTransferTrainerFactory.trainer_init(
            backend="nccl",
            config=NCCLWeightTransferConfig(packed=True),
            init_info=NCCLTrainerInitInfo(
                master_address=self.transfer_master_address,
                master_port=self.transfer_port,
                world_size=transfer_world_size,
            ),
            client=HTTPVLLMWeightSyncClient(base_url),
            # Yields sharded DTensors; the engine reads global shape/dtype for
            # metadata (no gather) and calls full_tensor() at broadcast time.
            weight_iterator=self.model.named_parameters,
        )

    # ---- collective ops (ALL FSDP ranks must call concurrently) ----

    def gather_and_broadcast_weights(self):
        """All-gather full parameters and broadcast them to the vLLM server.

        `full_tensor()` is a collective — all FSDP ranks must call it for each
        parameter in the same order. Rank 0 drives the full update (the engine
        runs the server-side update_weights concurrently with the NCCL
        broadcast); the other ranks just participate in the all-gather.
        """
        if self.rank == 0:
            self.engine.send_weights()
        else:
            for _, param in self.model.named_parameters():
                param.full_tensor()


def start_vllm_server() -> subprocess.Popen:
    """Spawn a `vllm serve` HTTP server (EP+DP) on SERVER_GPUS and wait for it."""
    serve_args = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--tensor-parallel-size",
        str(INFERENCE_TP_SIZE),
        "--data-parallel-size",
        str(INFERENCE_DP_SIZE),
        "--enable-expert-parallel",
        "--enforce-eager",
        "--load-format",
        "dummy",
        "--gpu-memory-utilization",
        "0.7",
        "--port",
        str(SERVER_PORT),
        "--weight-transfer-config",
        json.dumps({"backend": "nccl"}),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = SERVER_GPUS
    env["VLLM_SERVER_DEV_MODE"] = "1"  # exposes the weight-transfer endpoints
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    print(f"[server] Launching: {' '.join(serve_args)} (GPUs {SERVER_GPUS})")
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
    """Generate completions for a batch of prompts via the OpenAI HTTP API."""
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


def main():
    # Download model weights to local/shared disk once.
    local_model_path = snapshot_download(MODEL_NAME)
    print(f"[init] Model downloaded to {local_model_path}")

    # Start the inference server (GPUs 0-3). Do this before restricting Ray's
    # GPUs so the server keeps its own CUDA_VISIBLE_DEVICES.
    server_proc = start_vllm_server()
    try:
        # Restrict Ray (training) to GPUs 4-7 so it never collides with the
        # server. Must be set before ray.init() and before any CUDA use here.
        os.environ["CUDA_VISIBLE_DEVICES"] = TRAINING_GPUS
        ray.init()

        # FSDP rendezvous address (single-node).
        fsdp_master_addr = get_ip()
        fsdp_master_port = get_open_port()

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

        client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="EMPTY")

        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]

        # Generate with dummy weights — expect gibberish.
        print("[generate] Generating with dummy weights...")
        outputs = generate_completions(client, prompts)
        print("-" * 60)
        print("BEFORE weight sync (dummy weights):")
        print("-" * 60)
        for prompt, text in zip(prompts, outputs):
            print(f"Prompt: {prompt!r}")
            print(f"Generated: {text!r}")
            print("-" * 60)

        # --- Weight-transfer setup ---
        print("[transfer] Setting up weight-transfer endpoint...")
        transfer_addr, transfer_port = ray.get(
            fsdp_workers[0].setup_transfer_endpoint.remote()
        )
        print(f"[transfer] Endpoint ready at {transfer_addr}:{transfer_port}")

        transfer_world_size = INFERENCE_TP_SIZE * INFERENCE_DP_SIZE + 1
        print(
            f"[transfer] World size: {transfer_world_size} "
            f"(1 trainer + {INFERENCE_TP_SIZE * INFERENCE_DP_SIZE} vLLM workers)"
        )

        # Build the trainer engine on rank 0. This drives the server's
        # init_weight_transfer_engine (HTTP) while opening the trainer NCCL
        # endpoint, so both ends rendezvous together.
        print("[transfer] Initializing NCCL groups...")
        ray.get(fsdp_workers[0].setup_engine.remote(BASE_URL, transfer_world_size))
        print("[transfer] NCCL groups initialized.")

        # --- Pause, transfer weights, resume ---
        print("[sync] Pausing generation...")
        requests.post(f"{BASE_URL}/pause", timeout=60).raise_for_status()

        # All ranks participate in the FSDP all-gather; rank 0 additionally
        # drives start/update/finish on the server and the NCCL broadcast.
        print("[sync] Broadcasting weights from FSDP → vLLM...")
        ray.get([w.gather_and_broadcast_weights.remote() for w in fsdp_workers])
        print("[sync] Weight broadcast complete.")

        print("[sync] Resuming generation...")
        requests.post(f"{BASE_URL}/resume", timeout=60).raise_for_status()

        # Generate with synced weights — expect sensible output.
        print("[generate] Generating with synced weights...")
        outputs_updated = generate_completions(client, prompts)
        print("-" * 60)
        print("AFTER weight sync (real weights):")
        print("-" * 60)
        for prompt, text in zip(prompts, outputs_updated):
            print(f"Prompt: {prompt!r}")
            print(f"Generated: {text!r}")
            print("-" * 60)
    finally:
        print("[server] Shutting down...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server_proc.kill()


if __name__ == "__main__":
    main()
