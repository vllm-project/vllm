# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
RLHF with Megatron-Core training (4 GPUs) and vLLM inference (2 GPUs).

6-GPU layout:
  Training  — 4 GPUs, Megatron-Core (TP=2 × PP=2) via Megatron-Bridge,
              as Ray actors
  Inference — 2 GPUs, a `vllm serve` HTTP server with TP=2

Requires `megatron-core` and `megatron-bridge` (Python 3.12+) in addition to
vLLM's usual RLHF example dependencies.

This is the Megatron counterpart of `rlhf_nccl_fsdp_ep.py`: the weight-sync
control plane (HTTP) and the NCCL data plane both run inside the rank-0
Megatron Ray actor, driven by a single
`TrainerWeightTransferEngine.send_weights()` call. What changes is the
`WeightSource`: instead of `ModuleSource` over `named_parameters()`, a
`MegatronSource` (defined below) streams `bridge.export_hf_weights(...)`.

Megatron-Bridge's export is a collective with FSDP-`full_tensor()`-like
semantics: for each parameter it broadcasts across PP ranks, all-gathers
across TP ranks, and all-gathers experts across EP ranks, so **every**
trainer rank materializes the full HF-format tensor ("All ranks get full
tensors" — `AutoBridge.export_hf_weights`). All ranks must therefore iterate
the source in lockstep; only rank 0 (the sender) holds the transfer engine's
NCCL communicator and broadcasts to the inference workers.

GPU split (single node): Ray reserves 4 training GPUs first, and the server
is placed on complementary GPUs.

Steps:
  1. Launch 4 Megatron training workers (Ray); load real HF weights.
  2. Launch the vLLM HTTP server (TP=2, dummy weights) on the free GPUs.
  3. Generate from prompts over HTTP → gibberish (random weights).
  4. Pause generation, transfer weights Megatron → server over NCCL, resume.
  5. Generate from prompts → sensible output (synced weights).

Assumes a single-node cluster with 6+ GPUs.
"""

import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator

import ray
import requests
import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download
from openai import OpenAI

from vllm.config import NCCLWeightTransferConfig
from vllm.distributed.weight_transfer import (
    HTTPVLLMWeightSyncClient,
    ParamMeta,
    WeightSource,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.nccl_common import NCCLTrainerInitInfo
from vllm.utils.network_utils import get_ip, get_open_port

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SERVED_MODEL_NAME = "policy"

MEGATRON_TP_SIZE = 2
MEGATRON_PP_SIZE = 2
MEGATRON_WORLD_SIZE = MEGATRON_TP_SIZE * MEGATRON_PP_SIZE
INFERENCE_TP_SIZE = 2

SERVER_PORT = 8000
BASE_URL = f"http://localhost:{SERVER_PORT}"


class MegatronSource(WeightSource):
    """`WeightSource` over a Megatron-Bridge HF export.

    Iteration streams `bridge.export_hf_weights(model)` — a collective (PP
    broadcast + TP/EP all-gather inside each param mapping) that materializes
    every full HF-format tensor on **every** trainer rank, so all ranks must
    iterate in lockstep (the same contract `ModuleSource` has for FSDP
    `full_tensor()`).

    `metadata()` has no shape-only channel in Megatron-Bridge (fused-QKV
    splitting and MoE re-fusing mean HF names/shapes only exist after
    conversion), so the first call runs one full export, drops the tensors,
    and caches the result — itself a collective every rank must enter.

    Example-local for now; candidate for promotion next to `ModuleSource`.
    """

    def __init__(
        self,
        bridge,
        model: list[torch.nn.Module],
        dtype: torch.dtype | None = None,
    ) -> None:
        self._bridge = bridge
        self._model = model
        self._dtype = dtype
        self._meta: list[ParamMeta] | None = None

    def _export(self) -> Iterator[tuple[str, torch.Tensor]]:
        # cpu=False keeps tensors on GPU for the NCCL broadcast. Conversion
        # tasks are rebuilt on every pass so the mappings' PP-collective
        # caches start clean each round.
        yield from self._bridge.export_hf_weights(
            self._model, cpu=False, show_progress=False
        )

    def metadata(self) -> list[ParamMeta]:
        if self._meta is None:
            self._meta = [
                ParamMeta(name, self._dtype or tensor.dtype, tuple(tensor.shape))
                for name, tensor in self._export()
            ]
        return self._meta

    def __iter__(self) -> Iterator[tuple[str, torch.Tensor]]:
        for name, tensor in self._export():
            if self._dtype is not None:
                tensor = tensor.to(dtype=self._dtype)
            yield name, tensor


@ray.remote(num_gpus=1)
class MegatronTrainWorker:
    """
    One Megatron training worker per GPU. Four of these form the TP=2 × PP=2
    model-parallel group. Rank 0 additionally drives weight transfer to the
    vLLM server.
    """

    def __init__(
        self,
        model_path: str,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
    ):
        from megatron.bridge import AutoBridge
        from megatron.core import parallel_state, tensor_parallel

        self.rank = rank
        self.engine = None

        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)

        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.accelerator.set_device_index(0)

        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=MEGATRON_TP_SIZE,
            pipeline_model_parallel_size=MEGATRON_PP_SIZE,
        )
        tensor_parallel.model_parallel_cuda_manual_seed(42)

        # Bridge: HF config -> Megatron provider -> distributed model with the
        # real HF weights loaded (sharded per this rank's TP/PP coordinates).
        self.bridge = AutoBridge.from_hf_pretrained(model_path)
        provider = self.bridge.to_megatron_provider(load_weights=True)
        provider.tensor_model_parallel_size = MEGATRON_TP_SIZE
        provider.pipeline_model_parallel_size = MEGATRON_PP_SIZE
        provider.pipeline_dtype = torch.bfloat16
        # The fused weight-grad kernel comes from Apex, which this example
        # does not require.
        provider.gradient_accumulation_fusion = False
        provider.finalize()
        # List of module chunks (one per virtual-PP stage; a single entry here).
        self.model = provider.provide_distributed_model(wrap_with_ddp=False, bf16=True)

        self.transfer_port = None
        self.transfer_master_address = None

    def get_rank(self):
        return self.rank

    def get_gpu_ids(self):
        """Physical GPU id(s) Ray assigned to this worker (for server/train split)."""
        return ray.get_gpu_ids()

    # ---- weight-transfer setup (rank 0 only) ----

    def setup_transfer_endpoint(self):
        """Create the NCCL rendezvous endpoint for weight transfer."""
        assert self.rank == 0
        self.transfer_port = get_open_port()
        self.transfer_master_address = get_ip()
        return self.transfer_master_address, self.transfer_port

    def setup_engine(
        self,
        base_url: str,
        transfer_master_address: str,
        transfer_port: int,
        transfer_world_size: int,
    ):
        """Build the trainer engine on every Megatron rank.

        Called on all ranks with the shared rendezvous endpoint. Rank 0 is the
        sender: `trainer_init` opens its rank-0 NCCL endpoint and, on a worker
        thread, calls the server's `init_weight_transfer_engine` over HTTP so
        both ends rendezvous together. The other ranks skip the rendezvous and
        only join the export collectives during send_weights.
        """
        self.engine = WeightTransferTrainerFactory.trainer_init(
            backend="nccl",
            config=NCCLWeightTransferConfig(packed=True),
            init_info=NCCLTrainerInitInfo(
                master_address=transfer_master_address,
                master_port=transfer_port,
                world_size=transfer_world_size,
                rank=self.rank,  # Megatron global rank; sender is rank 0
            ),
            client=HTTPVLLMWeightSyncClient(base_url),
            # Streams full HF-format tensors, materialized on every rank by
            # the bridge's PP-broadcast + TP/EP-all-gather collectives.
            source=MegatronSource(self.bridge, self.model),
        )

    # ---- collective ops (ALL Megatron ranks must call concurrently) ----

    def gather_and_broadcast_weights(self):
        """Export full parameters and broadcast them to the vLLM server.

        Called on all Megatron ranks. `send_weights` iterates the bridge
        export (a collective every rank must enter in the same order); only
        rank 0 (the sender) drives the server-side update_weights concurrently
        with the NCCL broadcast — the other ranks only export.
        """
        self.engine.send_weights()


def start_vllm_server(server_gpus: str) -> subprocess.Popen:
    """Spawn a `vllm serve` HTTP server (TP=2) on `server_gpus` and wait for it."""
    serve_args = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--tensor-parallel-size",
        str(INFERENCE_TP_SIZE),
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
    env["CUDA_VISIBLE_DEVICES"] = server_gpus
    env["VLLM_SERVER_DEV_MODE"] = "1"  # exposes the weight-transfer endpoints
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    print(f"[server] Launching: {' '.join(serve_args)} (GPUs {server_gpus})")
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

    ray.init()

    # Megatron rendezvous address (single-node).
    megatron_master_addr = get_ip()
    megatron_master_port = get_open_port()

    # Launch the Megatron training workers first so Ray reserves their GPUs,
    # then place the inference server on the GPUs Ray did NOT use. This keeps
    # the two on disjoint physical GPUs whether ray.init() started a fresh
    # cluster or connected to an existing one.
    train_workers = [
        MegatronTrainWorker.remote(
            local_model_path,
            rank,
            MEGATRON_WORLD_SIZE,
            megatron_master_addr,
            megatron_master_port,
        )
        for rank in range(MEGATRON_WORLD_SIZE)
    ]
    ray.get([w.get_rank.remote() for w in train_workers])
    print(f"[init] {MEGATRON_WORLD_SIZE} Megatron training workers ready.")

    # Discover the physical GPUs Ray assigned to training; run the server on
    # the complementary GPUs.
    training_gpus = {
        int(g)
        for ids in ray.get([w.get_gpu_ids.remote() for w in train_workers])
        for g in ids
    }
    num_gpus = int(ray.cluster_resources().get("GPU", 0))
    server_gpu_ids = [g for g in range(num_gpus) if g not in training_gpus][
        :INFERENCE_TP_SIZE
    ]
    if len(server_gpu_ids) < INFERENCE_TP_SIZE:
        raise RuntimeError(
            f"Need {INFERENCE_TP_SIZE} free GPUs for the inference server but only "
            f"found {server_gpu_ids} (training uses {sorted(training_gpus)} of "
            f"{num_gpus} cluster GPUs)."
        )
    server_gpus = ",".join(str(g) for g in server_gpu_ids)
    print(f"[init] Training GPUs {sorted(training_gpus)}; server GPUs [{server_gpus}].")

    # Start the inference server on the complementary GPUs.
    server_proc = start_vllm_server(server_gpus)
    try:
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
            train_workers[0].setup_transfer_endpoint.remote()
        )
        print(f"[transfer] Endpoint ready at {transfer_addr}:{transfer_port}")

        transfer_world_size = INFERENCE_TP_SIZE + 1
        print(
            f"[transfer] World size: {transfer_world_size} "
            f"(1 trainer + {INFERENCE_TP_SIZE} vLLM workers)"
        )

        # Build the trainer engine on all Megatron ranks (rank 0 is the
        # sender). The sender drives the server's init_weight_transfer_engine
        # (HTTP) while opening the trainer NCCL endpoint, so both ends
        # rendezvous together; the other ranks only join the export
        # collectives.
        print("[transfer] Initializing NCCL groups (all Megatron ranks)...")
        ray.get(
            [
                w.setup_engine.remote(
                    BASE_URL, transfer_addr, transfer_port, transfer_world_size
                )
                for w in train_workers
            ]
        )
        print("[transfer] NCCL groups initialized.")

        # --- Pause, transfer weights, resume ---
        print("[sync] Pausing generation...")
        requests.post(f"{BASE_URL}/pause", timeout=60).raise_for_status()

        # All ranks participate in the bridge export collectives; rank 0
        # additionally drives start/update/finish on the server and the NCCL
        # broadcast.
        print("[sync] Broadcasting weights from Megatron → vLLM...")
        ray.get([w.gather_and_broadcast_weights.remote() for w in train_workers])
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
