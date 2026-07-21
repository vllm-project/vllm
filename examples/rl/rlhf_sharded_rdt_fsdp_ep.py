# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RLHF weight sync: 8-rank FSDP2 trainer -> vLLM DP8+EP inference via the
sharded-RDT weight-transfer backend, driven over the HTTP control plane.

A fixed 8->8 (1:1) specialization of rlhf_sharded_rdt_mn.py, kept as a distinct
example because it exercises STRICT_PACK placement-group scheduling (all FSDP
ranks packed onto one node, inference DP ranks on the other) and always-on
expert parallelism. For the arbitrary M:N regimes (fan-in / split) and the
env-driven fleet sizing, see rlhf_sharded_rdt_mn.py.

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

Needs two 8-GPU nodes (8 trainer + 8 inference). For a single-node,
size-configurable run, use rlhf_sharded_rdt_mn.py.
"""

import json
import os
import sys
import time

import ray
import torch
import torch.distributed as dist
from ray.util.placement_group import placement_group, placement_group_table
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
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


def _load_sharded_from_disk(model, model_name: str, config) -> None:
    """Stream each FSDP rank's local shard directly from the on-disk safetensors.

    The whole model is NEVER materialized on any single GPU. Call after
    ``fully_shard`` + ``model.to_empty('cuda')``.

    Three cases:
      * Normal params: FSDP2 shards them ``Shard(dim=0)``, so each rank reads only
        its rows ``disk[name][offset : offset + local_rows]``.
      * MoE experts: FUSED in the model (``experts.gate_up_proj`` [E, 2*I, H] and
        ``experts.down_proj`` [E, H, I]) but stored PER-EXPERT on disk. Each rank
        loads only its local experts' gate/up/down and fuses them
        (``gate_up = cat([gate, up], 0)``; down copied directly).
      * Buffers (rotary ``inv_freq``): recomputed from config after ``to_empty``.
    """
    import glob

    import regex as re
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

    snap = snapshot_download(model_name)
    index = os.path.join(snap, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            weight_map = json.load(f)["weight_map"]
    else:
        weight_map = {}
        for f in glob.glob(os.path.join(snap, "*.safetensors")):
            with safe_open(f, framework="pt") as sf:
                for k in sf.keys():  # noqa: SIM118 (safe_open is not iterable)
                    weight_map[k] = os.path.basename(f)

    handles: dict = {}

    def handle(key: str):
        fn = weight_map[key]
        h = handles.get(fn)
        if h is None:
            h = safe_open(os.path.join(snap, fn), framework="pt", device="cuda:0")
            handles[fn] = h
        return h

    expert_re = re.compile(r"^(.*\.experts)\.(gate_up_proj|down_proj)$")

    with torch.no_grad():
        for name, param in model.named_parameters():
            local = param.to_local().detach()
            lshape, goff = compute_local_shape_and_global_offset(
                param.shape, param.device_mesh, param.placements
            )
            local.zero_()
            n0 = lshape[0]
            if n0 == 0:
                continue
            m = expert_re.match(name)
            if m:
                prefix, kind = m.group(1), m.group(2)
                e0 = goff[0]
                for i in range(n0):
                    e = e0 + i
                    if kind == "gate_up_proj":
                        gk = f"{prefix}.{e}.gate_proj.weight"
                        uk = f"{prefix}.{e}.up_proj.weight"
                        g = handle(gk).get_tensor(gk)
                        u = handle(uk).get_tensor(uk)
                        local[i].copy_(torch.cat([g, u], dim=0))
                    else:
                        dk = f"{prefix}.{e}.down_proj.weight"
                        local[i].copy_(handle(dk).get_tensor(dk))
            else:
                if name not in weight_map:
                    raise RuntimeError(
                        f"param {name!r} is not in the checkpoint and is not a "
                        f"fused expert param (tied weights not handled here)."
                    )
                sliced = handle(name).get_slice(name)
                local[:n0].copy_(sliced[goff[0] : goff[0] + n0])

    rot = model.model.rotary_emb
    fresh = type(rot)(config=config, device=torch.device("cuda"))
    rot.inv_freq = fresh.inv_freq.to("cuda")
    if hasattr(rot, "original_inv_freq"):
        rot.original_inv_freq = rot.inv_freq
    if hasattr(fresh, "attention_scaling"):
        rot.attention_scaling = fresh.attention_scaling


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
        _load_sharded_from_disk(model, model_name, config)

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
                layerwise_split=int(os.environ.get("LAYERWISE_SPLIT", "1")),
                arena_presize_gb=float(os.environ.get("RDT_ARENA_PRESIZE_GB", "0")),
                nosync=os.environ.get("RDT_NOSYNC", "0") == "1",
                pack_check=os.environ.get("RDT_PACK_CHECK", "0") == "1",
            ),
            client=HTTPVLLMWeightSyncClient(vllm_endpoint),
            source=ModuleSource(self.model),
        )

    def sync_weights(self):
        self.engine.send_weights()

    def get_sync_timing(self):
        return self.engine.get_sync_timing()

    def reset_produce_timing(self):
        self.engine.reset_produce_timing()

    def get_produce_timing(self):
        return self.engine.get_produce_timing()


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
            ray.get([w.reset_produce_timing.remote() for w in fsdp_workers])

            print(f"[sync] iter {sync_iter} [REPLAY]: gather + serve + update...")
            _sync_t0 = time.perf_counter()
            ray.get([w.sync_weights.remote() for w in fsdp_workers])
            _sync_seconds = time.perf_counter() - _sync_t0

            ptimings = ray.get([w.get_produce_timing.remote() for w in fsdp_workers])
            gib = sum(p["bytes"] for p in ptimings) / (1024**3)
            slice_s_max = max(p["slice_seconds"] for p in ptimings)
            sender_timing = ray.get(fsdp_workers[0].get_sync_timing.remote())
            print("=" * 60)
            print(f"[profile] iter {sync_iter} [REPLAY]")
            print(f"[profile] total weight-sync wall time : {_sync_seconds:.3f} s")
            print(
                "[profile] sender breakdown (s)         : "
                + ", ".join(f"{k}={v:.3f}" for k, v in sender_timing.items())
            )
            print(f"[profile] trainer slice+clone (slowest): {slice_s_max:.3f} s")
            print(f"[profile] bytes produced (all ranks)   : {gib:.3f} GiB")
            print("=" * 60)

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
