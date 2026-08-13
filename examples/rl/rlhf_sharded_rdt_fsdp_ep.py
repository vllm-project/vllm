# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RLHF weight sync: 8-rank FSDP2 trainer -> vLLM DP8+EP inference via the
sharded-RDT weight-transfer backend, driven over the HTTP control plane.

The reference example for the backend: a fixed 8->8 (1:1) fleet with STRICT_PACK
placement-group scheduling (all FSDP ranks packed onto one node, inference DP
ranks on the other) and always-on expert parallelism. Producers and consumers
need not match 1:1 — ``RdtRouter`` (sharded_rdt_common.py) handles fan-in and
split fleets — but this example fixes the sizes to keep the wiring readable.

Architecture:
  - Each FSDP trainer rank builds a ``ShardedRDTTrainerWeightTransferEngine``
    over a ``ModuleSource`` of its FSDP-sharded model. All the RDT complexity —
    the per-rank NIXL serve actor, packed serve ring, gather cache, free
    ref-counting — is hidden inside the engine. ``send_weights`` gathers each
    layer group (``full_tensor()`` collectives rendezvous because every rank
    iterates the ModuleSource in the same order), shares it into the serve
    actor over CUDA IPC, and (rank 0) drives the inference start/update/finish.
  - Inference runs as a standard ``vllm serve`` process (no bespoke Ray-actor
    wrapper): the trainer reaches its weight-sync control plane through the
    RLHF HTTP routes via ``HTTPVLLMWeightSyncClient``, and generation uses the
    OpenAI ``/v1/completions`` API.

Requirements on the vLLM server (this script launches it for you):
  - ``VLLM_SERVER_DEV_MODE=1`` exposes the ``/init_weight_transfer_engine`` etc.
    RLHF routes.
  - ``--distributed-executor-backend ray`` + ``VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1``
    so the workers are RayExecutorV2 actors with tensor transport enabled — RDT's
    NIXL data plane and the ``ray.get_actor`` of the trainer's serve actors need
    this (the control plane being HTTP does not change that).
  - The server must share the trainer's Ray cluster (``address=auto``) so its
    workers can resolve the trainer serve actors by name+namespace.

Needs two 8-GPU nodes (8 trainer + 8 inference).
"""

import os
import sys
import time

import ray
import torch
import torch.distributed as dist
from ray.util.placement_group import placement_group, placement_group_table
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from rdt_fsdp_loader import load_sharded_from_disk
from rdt_vllm_serve import (
    http_generate,
    launch_vllm_serve,
    pause_generation,
    resume_generation,
    shutdown_server,
    wait_for_server,
)
from torch.distributed.fsdp import fully_shard
from transformers import AutoConfig, AutoModelForCausalLM

from vllm.distributed.weight_transfer import (
    HTTPVLLMWeightSyncClient,
    ModuleSource,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.sharded_rdt_trainer import (
    ShardedRDTTrainerInitInfo,
)
from vllm.utils.network_utils import get_open_port

MODEL_NAME = os.environ.get("RDT_MODEL", "Qwen/Qwen3-30B-A3B")
RAY_NAMESPACE = "sharded_rdt_fsdp_ep_example"
VLLM_PORT = int(os.environ.get("RDT_VLLM_PORT", "8100"))
VLLM_ENDPOINT = f"http://127.0.0.1:{VLLM_PORT}"

SYNC_ITERS = int(os.environ.get("RDT_SYNC_ITERS", "3"))

FSDP_WORLD_SIZE = 8
INFERENCE_TP_SIZE = 1
INFERENCE_DP_SIZE = 8
NUM_INFERENCE_CONSUMERS = INFERENCE_TP_SIZE * INFERENCE_DP_SIZE


@ray.remote(num_gpus=1)
class FSDPTrainWorker:
    """One FSDP2 training worker per GPU; 8 of them form the FSDP group. Each
    rank builds a sharded-RDT trainer engine over a ModuleSource of its model;
    the engine (not this actor) owns the NIXL serve surface, so this is a plain
    Ray actor with no producer mixin and no tensor-transport / concurrency
    options."""

    def __init__(
        self,
        model_name: str,
        rank: int,
        fsdp_world_size: int,
        fsdp_master_addr: str,
        fsdp_master_port: int,
    ):
        self.rank = rank
        self.world_size = fsdp_world_size
        self.engine = None

        os.environ["MASTER_ADDR"] = fsdp_master_addr
        os.environ["MASTER_PORT"] = str(fsdp_master_port)

        dist.init_process_group(backend="nccl", rank=rank, world_size=fsdp_world_size)
        torch.accelerator.set_device_index(0)

        config = AutoConfig.from_pretrained(model_name)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)

        for layer in model.model.layers:
            fully_shard(layer)
        fully_shard(model)

        model.to_empty(device="cuda")
        load_sharded_from_disk(model, model_name, config)

        self.model = model

    def get_rank(self):
        return self.rank

    def setup_engine(self, vllm_endpoint: str):
        # The trainer reaches the vLLM server's weight-sync control plane over
        # HTTP; the NIXL data plane still binds the trainer serve actors by name
        # in RAY_NAMESPACE (resolved on the vLLM workers via ray.get_actor).
        self.engine = WeightTransferTrainerFactory.trainer_init(
            ShardedRDTTrainerInitInfo(
                rank=self.rank,
                num_consumers=NUM_INFERENCE_CONSUMERS,
                trainer_actor_namespace=RAY_NAMESPACE,
                num_rdt_buffers=int(os.environ.get("NUM_RDT_BUFFERS", "2")),
                arena_presize_gb=float(os.environ.get("RDT_ARENA_PRESIZE_GB", "0")),
            ),
            client=HTTPVLLMWeightSyncClient(vllm_endpoint),
            source=ModuleSource(self.model),
        )

    def sync_weights(self):
        self.engine.send_weights()


def main():
    # Ship a minimal working_dir (this example dir) so Ray actors do NOT
    # inherit a workspace snapshot that shadows the editable vLLM install
    # (the snapshot lacks the compiled extensions). vLLM is imported from
    # the venv via py_executable.
    runtime_env: dict[str, object] = {
        "py_executable": sys.executable,
        "working_dir": os.path.dirname(os.path.abspath(__file__)),
    }
    forwarded = {
        k: os.environ[k]
        for k in (
            "NCCL_CUMEM_ENABLE",
            "VLLM_NCCL_SO_PATH",
            "LD_PRELOAD",
            "LD_LIBRARY_PATH",
        )
        if k in os.environ
    }
    if forwarded:
        runtime_env["env_vars"] = forwarded
    if not ray.is_initialized():
        ray.init(address="auto", runtime_env=runtime_env, namespace=RAY_NAMESPACE)

    local_model_path = MODEL_NAME
    print(f"[init] Using model id {local_model_path} (pre-cached on each node)")

    # Pin all FSDP ranks to a SINGLE node (STRICT_PACK) so the trainer NCCL
    # all-gather stays intra-node (NVLink). Create the trainer PG BEFORE the
    # server so the server's DP workers land on the other node's GPUs.
    trainer_pg = placement_group(
        [{"GPU": 1, "CPU": 1}] * FSDP_WORLD_SIZE, strategy="STRICT_PACK"
    )
    ray.get(trainer_pg.ready())

    pg_node_id = next(
        iter(placement_group_table(trainer_pg)["bundles_to_node_id"].values())
    )
    fsdp_master_addr = next(
        n["NodeManagerAddress"] for n in ray.nodes() if n["NodeID"] == pg_node_id
    )

    @ray.remote(
        num_cpus=0,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=trainer_pg
        ),
    )
    def _free_port_on_trainer_node():
        return get_open_port()

    fsdp_master_port = ray.get(_free_port_on_trainer_node.remote())
    print(f"[init] FSDP group on node {fsdp_master_addr}:{fsdp_master_port}")

    fsdp_workers = []
    for rank in range(FSDP_WORLD_SIZE):
        handle = FSDPTrainWorker.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=trainer_pg,
                placement_group_bundle_index=rank,
            ),
        ).remote(
            local_model_path,
            rank,
            FSDP_WORLD_SIZE,
            fsdp_master_addr,
            fsdp_master_port,
        )
        fsdp_workers.append(handle)
    ray.get([w.get_rank.remote() for w in fsdp_workers])
    print(f"[init] {FSDP_WORLD_SIZE} FSDP training workers ready.")

    server = launch_vllm_serve(
        local_model_path,
        tensor_parallel_size=INFERENCE_TP_SIZE,
        data_parallel_size=INFERENCE_DP_SIZE,
        enable_expert_parallel=True,
        port=VLLM_PORT,
    )
    try:
        wait_for_server(VLLM_ENDPOINT, server)

        print("[generate] Generating with dummy weights...")
        print("-" * 60)
        print("BEFORE weight sync (dummy weights):")
        print("-" * 60)
        for prompt, text in zip(
            [
                "Hello, my name is",
                "The president of the United States is",
                "The capital of France is",
                "The future of AI is",
            ],
            http_generate(
                VLLM_ENDPOINT,
                local_model_path,
                [
                    "Hello, my name is",
                    "The president of the United States is",
                    "The capital of France is",
                    "The future of AI is",
                ],
            ),
        ):
            print(f"Prompt: {prompt!r}\nGenerated: {text!r}\n" + "-" * 60)

        print("[sync] Building trainer engines (dry-run bake on the sender)...")
        _init_t0 = time.perf_counter()
        ray.get([w.setup_engine.remote(VLLM_ENDPOINT) for w in fsdp_workers])
        print(
            f"[sync] engine setup (incl. bake) took "
            f"{time.perf_counter() - _init_t0:.3f} s"
        )

        pause_generation(VLLM_ENDPOINT)

        for sync_iter in range(SYNC_ITERS):
            print(f"[sync] iter {sync_iter} [REPLAY]: gather + serve + update...")
            _sync_t0 = time.perf_counter()
            ray.get([w.sync_weights.remote() for w in fsdp_workers])
            print(
                f"[sync] iter {sync_iter} took {time.perf_counter() - _sync_t0:.3f} s"
            )

        resume_generation(VLLM_ENDPOINT)

        print("[generate] Generating with synced weights...")
        print("-" * 60)
        print("AFTER weight sync (real weights):")
        print("-" * 60)
        for prompt, text in zip(
            [
                "Hello, my name is",
                "The president of the United States is",
                "The capital of France is",
                "The future of AI is",
            ],
            http_generate(
                VLLM_ENDPOINT,
                local_model_path,
                [
                    "Hello, my name is",
                    "The president of the United States is",
                    "The capital of France is",
                    "The future of AI is",
                ],
            ),
        ):
            print(f"Prompt: {prompt!r}\nGenerated: {text!r}\n" + "-" * 60)
    finally:
        shutdown_server(server)


if __name__ == "__main__":
    main()
