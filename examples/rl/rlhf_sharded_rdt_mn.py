# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RLHF weight sync: arbitrary M:N FSDP2 trainer -> vLLM inference via the
sharded-RDT weight-transfer backend, driven over the HTTP control plane.
The canonical M:N example.

Fleet sizes and model come from env (see below), so the SAME file runs both
M:N regimes end to end:
  - more trainers than inference (P>C, e.g. 8->4): each consumer binds a
    contiguous block of producers and SPLITS every chunk-pull evenly across
    them (load balance);
  - more inference than trainers (C>P, e.g. 4->8): several consumers share one
    producer, which keeps a per-consumer serve ring and ref-counts frees.
The 1:1 case reduces to the pre-M:N behavior. See assign_producer_indices in
sharded_rdt_common.py for the block rule.

Architecture:
  - Each FSDP trainer rank builds a ``ShardedRDTTrainerWeightTransferEngine``
    over a ``ModuleSource`` of its model; the engine owns the per-rank NIXL
    serve actor / packed serve ring / gather cache. ``send_weights`` gathers
    each layer group (``full_tensor()`` collectives rendezvous because every
    rank iterates the ModuleSource in the same order) and (rank 0) drives the
    inference start/update/finish.
  - Inference runs as a standard ``vllm serve`` process; the trainer reaches
    its weight-sync control plane over the RLHF HTTP routes
    (``HTTPVLLMWeightSyncClient``) and generation uses ``/v1/completions``.

Env knobs:
  - fleet/model: MN_TRAINERS, MN_INFERENCE (or MN_INFERENCE_TP x MN_INFERENCE_DP),
    MN_MODEL (default Qwen/Qwen3-0.6B), MN_EP (1 for an MoE model). Dense models
    are served via TP (vLLM rejects DP over dense); MoE via DP(+EP).
  - transport: NUM_RDT_BUFFERS x LAYERWISE_SPLIT, RDT_ARENA_PRESIZE_GB,
    RDT_NOSYNC, RDT_PACK_CHECK, RDT_SYNC_ITERS.

The server is launched by this script (see rdt_vllm_serve.launch_vllm_serve) with
dev mode + the Ray v2 executor; it joins this process's Ray cluster so its
workers can resolve the trainer's serve actors. Single node fits e.g.
MN_TRAINERS=4 MN_INFERENCE=4; larger splits want a second node.
"""

import os
import sys
import time

import ray
import torch
import torch.distributed as dist
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
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

MODEL_NAME = os.environ.get("MN_MODEL", "Qwen/Qwen3-0.6B")
RAY_NAMESPACE = "sharded_rdt_mn_example"
VLLM_PORT = int(os.environ.get("RDT_VLLM_PORT", "8100"))
VLLM_ENDPOINT = f"http://127.0.0.1:{VLLM_PORT}"

SYNC_ITERS = int(os.environ.get("RDT_SYNC_ITERS", "3"))

FSDP_WORLD_SIZE = int(os.environ.get("MN_TRAINERS", "4"))
INFERENCE_TP_SIZE = int(os.environ.get("MN_INFERENCE_TP", "1"))
INFERENCE_DP_SIZE = int(
    os.environ.get("MN_INFERENCE_DP", os.environ.get("MN_INFERENCE", "8"))
)
NUM_INFERENCE_CONSUMERS = INFERENCE_TP_SIZE * INFERENCE_DP_SIZE
ENABLE_EP = os.environ.get("MN_EP", "0") == "1"


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
    import json

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
    """One FSDP2 training worker per GPU; MN_TRAINERS of them form the FSDP
    group. Each rank builds a sharded-RDT trainer engine over a ModuleSource of
    its model; the engine (not this actor) owns the NIXL serve surface, so this
    is a plain Ray actor with no producer mixin and no tensor-transport /
    concurrency options. ``full_tensor()`` all-gathers each layer to every rank,
    so any rank's serve actor can serve any NIXL pull."""

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

    # Pin the trainer fleet to one node so the FSDP NCCL all-gather stays
    # intra-node. NODE-AFFINITY, not a placement group: a PG on a partially-
    # filled node trips vLLM's DP-placement ``len(node_ip_keys)==1`` assertion.
    _trainer_ip = os.environ.get("RDT_TRAINER_NODE_IP")
    if _trainer_ip:
        trainer_node_id = next(
            n["NodeID"]
            for n in ray.nodes()
            if n["Alive"] and n["NodeManagerAddress"] == _trainer_ip
        )
    else:
        trainer_node_id = next(
            n["NodeID"]
            for n in ray.nodes()
            if n["Alive"] and n["Resources"].get("GPU", 0) > 0
        )
        _trainer_ip = next(
            n["NodeManagerAddress"]
            for n in ray.nodes()
            if n["NodeID"] == trainer_node_id
        )
    fsdp_master_addr = _trainer_ip
    trainer_sched = NodeAffinitySchedulingStrategy(node_id=trainer_node_id, soft=False)

    @ray.remote(num_cpus=0, scheduling_strategy=trainer_sched)
    def _free_port_on_trainer_node():
        return get_open_port()

    fsdp_master_port = ray.get(_free_port_on_trainer_node.remote())
    print(f"[init] FSDP group on node {fsdp_master_addr}:{fsdp_master_port}")

    fsdp_workers = []
    for rank in range(FSDP_WORLD_SIZE):
        handle = FSDPTrainWorker.options(
            num_gpus=1,
            scheduling_strategy=trainer_sched,
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
        enable_expert_parallel=ENABLE_EP,
        port=VLLM_PORT,
    )
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    try:
        wait_for_server(VLLM_ENDPOINT, server)

        print("-" * 60)
        print("BEFORE weight sync (dummy weights):")
        print("-" * 60)
        for prompt, text in zip(
            prompts, http_generate(VLLM_ENDPOINT, local_model_path, prompts)
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
            per_rank_gib = [p["bytes"] / (1024**3) for p in ptimings]
            slice_s_max = max(p["slice_seconds"] for p in ptimings)
            calls = sum(p["calls"] for p in ptimings)
            specs = sum(p["specs"] for p in ptimings)
            sender_timing = ray.get(fsdp_workers[0].get_sync_timing.remote())
            print("=" * 60)
            print(f"[profile] iter {sync_iter} [REPLAY]")
            print(f"[profile] total weight-sync wall time : {_sync_seconds:.3f} s")
            print(
                "[profile] sender breakdown (s)         : "
                + ", ".join(f"{k}={v:.3f}" for k, v in sender_timing.items())
            )
            print(f"[profile] trainer produce calls (all) : {calls}")
            print(f"[profile] trainer specs (slices) total: {specs}")
            print(f"[profile] trainer slice+clone (slowest): {slice_s_max:.3f} s")
            print(f"[profile] bytes produced (all ranks)   : {gib:.3f} GiB")
            print(
                "[profile] per-rank GiB                 : "
                + ", ".join(f"r{r}={g:.2f}" for r, g in enumerate(per_rank_gib))
            )
            print("=" * 60)

        resume_generation(VLLM_ENDPOINT)

        print("-" * 60)
        print("AFTER weight sync (real weights):")
        print("-" * 60)
        for prompt, text in zip(
            prompts, http_generate(VLLM_ENDPOINT, local_model_path, prompts)
        ):
            print(f"Prompt: {prompt!r}\nGenerated: {text!r}\n" + "-" * 60)
    finally:
        shutdown_server(server)


if __name__ == "__main__":
    main()
