# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RLHF weight sync: 8-GPU FSDP2 trainer -> 8-GPU vLLM (DP+EP) inference via
the sharded-RDT weight-transfer backend (Qwen3-30B-A3B, bf16).

Architecture (single-call design, shared with rlhf_sharded_rdt_kimi.py):
  - trainer ranks mix in RDTShardedProducer (rdt_producer.py): a self-paced
    per-group all-gather plan (``full_tensor()`` collectives rendezvous safely
    because every rank runs the IDENTICAL ordered plan) + the packed serve
    ring that mirrors the consumer's byte layout.
  - the driver makes ONE ``engine.update_weights`` per sync carrying all names
    + group_lens; the engine chunk-plans each group, pipelines packed pulls
    over its receive ring, and frees each group's gather on the producer as
    its chunks finish.
  - tuning knobs (env): NUM_RDT_BUFFERS x LAYERWISE_SPLIT (working set vs the
    fabric's address-translation reach), RDT_ARENA_PRESIZE_GB, RDT_NIC_RAILS /
    UCX_IB_NUM_PATHS (fabric), RDT_NOSYNC (paired Ray patch), RDT_PACK_CHECK.

Run on a 2-node 8+8 GPU Ray cluster (trainer fleet pinned to one node via a
STRICT_PACK placement group; vLLM DP fleet lands on the other). See
multi_node_rdt.md for the cluster runbook and optimization history.
"""

import asyncio
import os
import sys
import threading
import time
import uuid
from dataclasses import asdict

import ray
import torch
import torch.distributed as dist
from ray.util.placement_group import placement_group, placement_group_table
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torch.distributed.fsdp import fully_shard
from transformers import AutoConfig, AutoModelForCausalLM

import vllm
from vllm import SamplingParams
from vllm.config import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
)
from vllm.distributed.weight_transfer.sharded_rdt_engine import (
    ShardedRDTWeightTransferInitInfo,
    ShardedRDTWeightTransferUpdateInfo,
)
from vllm.utils.network_utils import get_open_port
from vllm.v1.executor import Executor

# Local module (ships with the example via runtime_env working_dir): shared
# sharded-RDT producer (packed serve ring + self-paced gather plan).
from rdt_producer import RDTShardedProducer, layerwise_groups

MODEL_NAME = "Qwen/Qwen3-30B-A3B"
TRAINER_ACTOR_NAME = "sharded_rdt_fsdp_trainer"
RAY_NAMESPACE = "sharded_rdt_fsdp_example"


def trainer_actor_name(rank: int) -> str:
    """Ray actor name for an FSDP rank's RDT producer.

    Rank 0 keeps the canonical name (back-compat with external tooling); ranks
    1+ get a ``_rank{N}`` suffix. All ranks are named so inference workers can
    resolve and pull from any of them for load balancing.
    """
    return TRAINER_ACTOR_NAME if rank == 0 else f"{TRAINER_ACTOR_NAME}_rank{rank}"

# RDT_SYNC_ITERS -> how many back-to-back weight syncs to run. The sharded RDT
# backend bakes a replay plan on the first sync for a given name set and
# replays it on subsequent syncs, so use >=2 to observe the replay speedup.
SYNC_ITERS = int(os.environ.get("RDT_SYNC_ITERS", "3"))

FSDP_WORLD_SIZE = 8
INFERENCE_TP_SIZE = 1
INFERENCE_DP_SIZE = 8
# vLLM workers in the inference EP group; each one calls
# rdt_produce_weights_batched once per layer. Used only to size the actor
# threadpool (one concurrent produce call per worker, plus gather).
NUM_INFERENCE_CONSUMERS = INFERENCE_TP_SIZE * INFERENCE_DP_SIZE


def _load_sharded_from_disk(model, model_name: str, config) -> None:
    """Stream each FSDP rank's local shard directly from the on-disk safetensors.

    The whole model is NEVER materialized on any single GPU. This replaces the
    ``from_pretrained`` path, which loaded the full model on EVERY rank before
    ``fully_shard`` -- fine for models that fit on one GPU, but OOMs for ones that
    don't (e.g. Kimi-K2). Call after ``fully_shard`` + ``model.to_empty('cuda')``.

    Three cases:
      * Normal params: FSDP2 shards them ``Shard(dim=0)``, so each rank reads only
        its rows ``disk[name][offset : offset + local_rows]`` (a partial
        safetensors read -- never the whole tensor).
      * MoE experts: FUSED in the model (``experts.gate_up_proj`` [E, 2*I, H] and
        ``experts.down_proj`` [E, H, I]) but stored PER-EXPERT on disk. The fused
        dim 0 is the expert dim, so each rank loads only its local experts'
        individual gate/up/down and fuses them (``gate_up = cat([gate, up], 0)``,
        verified against from_pretrained; down copied directly).
      * Buffers (rotary ``inv_freq``): not in the checkpoint and garbage after
        ``to_empty``; recomputed from config via ``ROPE_INIT_FUNCTIONS``.
    """
    import glob
    import json
    import os
    import re

    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

    snap = snapshot_download(model_name)  # already cached -> local dir, no download
    index = os.path.join(snap, "model.safetensors.index.json")
    if os.path.exists(index):
        weight_map = json.load(open(index))["weight_map"]
    else:
        weight_map = {}
        for f in glob.glob(os.path.join(snap, "*.safetensors")):
            with safe_open(f, framework="pt") as sf:
                for k in sf.keys():
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

    # no_grad + detach: params have requires_grad=True, and writing in-place into
    # the autograd view returned by ``to_local()`` is forbidden by autograd. We
    # only fill storage (never train here), so detach and disable grad tracking.
    with torch.no_grad():
        for name, param in model.named_parameters():
            local = param.to_local().detach()  # this rank's shard storage
            lshape, goff = compute_local_shape_and_global_offset(
                param.shape, param.device_mesh, param.placements
            )
            local.zero_()  # zero first so any FSDP padding rows stay zero
            n0 = lshape[0]  # real rows this rank owns along the sharded dim
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
                    else:  # down_proj: stored per-expert directly, no fusion
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

    # Recompute the rotary inv_freq buffers: non-persistent, not in the
    # checkpoint, and garbage after to_empty(). Re-instantiate the rotary module
    # (its __init__ computes inv_freq from config) -- version- and rope-type-
    # agnostic, so this also works for scaled rope (yarn/longrope) on other models.
    rot = model.model.rotary_emb
    fresh = type(rot)(config=config, device=torch.device("cuda"))
    rot.inv_freq = fresh.inv_freq.to("cuda")
    if hasattr(rot, "original_inv_freq"):
        rot.original_inv_freq = rot.inv_freq
    if hasattr(fresh, "attention_scaling"):
        rot.attention_scaling = fresh.attention_scaling


# max_concurrency=8 lets each rank service inbound ``gather_layer`` calls AND
# the concurrent ``rdt_produce_weights_batched`` calls from the vLLM workers it
# serves, on separate threads in the actor's threadpool. With static 1:1 load
# balancing each rank serves a single inference worker, but 8 gives ample
# headroom (and tolerates other routing policies).
# Concurrent produce calls are read-only against the cache, so they need no
# locking beyond the gather/free synchronization: the driver frees a layer
# group only after its update_weights has fully drained.
@ray.remote(num_gpus=1, max_concurrency=8, enable_tensor_transport=True)
class FSDPTrainWorker(RDTShardedProducer):
    """One FSDP2 training worker per GPU.

    Four of these form the FSDP group. Every rank serves RDT-tagged slice
    requests to the vLLM inference workers: ``full_tensor()`` all-gathers each
    layer to all ranks, so each rank can serve NIXL pulls. Inference workers are
    statically mapped 1:1 onto the ranks to spread the trainer-side clone + NIC
    egress instead of funneling everything through rank 0.
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
        self.world_size = fsdp_world_size

        os.environ["MASTER_ADDR"] = fsdp_master_addr
        os.environ["MASTER_PORT"] = str(fsdp_master_port)

        dist.init_process_group(backend="nccl", rank=rank, world_size=fsdp_world_size)
        torch.accelerator.set_device_index(0)

        # Memory-scalable load: build on META (zero allocation), shard, then stream
        # each rank's shard directly from the on-disk safetensors. The whole model
        # is NEVER materialized on any single GPU. (The old ``from_pretrained``
        # path put the full model on EVERY rank before sharding, which OOMs for
        # models that don't fit on one GPU, e.g. Kimi-K2.)
        config = AutoConfig.from_pretrained(model_name)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)

        # Capture metadata BEFORE fully_shard so we have stable names/dtypes
        # /shapes to hand to vLLM's update_info. Valid on the meta model. After
        # sharding, params become DTensors but keep the same names.
        self.weight_names = [n for n, _ in model.named_parameters()]
        self.weight_dtype_names = [
            str(p.dtype).split(".")[-1] for _, p in model.named_parameters()
        ]
        self.weight_shapes = [list(p.shape) for _, p in model.named_parameters()]

        for layer in model.model.layers:
            fully_shard(layer)
        fully_shard(model)

        # Allocate ONLY the local shards (empty) on GPU, then fill from disk.
        model.to_empty(device="cuda")
        _load_sharded_from_disk(model, model_name, config)

        self.model = model
        # Post-sharding lookup. Each entry is a DTensor with full_tensor()
        # available as a collective.
        self._param_lookup = dict(model.named_parameters())

        # Shared sharded-RDT producer: gather cache + packed serve ring +
        # timing (see rdt_producer.py). Gathered full tensors are FRESH buffers
        # published per group and freed by the engine's free_gather.
        self.init_rdt_producer()

        from vllm.distributed.weight_transfer._nixl_profile import install_nixl_timing

        install_nixl_timing()

    def get_rank(self):
        return self.rank

    def get_weight_metadata(self):
        return self.weight_names, self.weight_dtype_names, self.weight_shapes

    # ---------- gather hook (RDTShardedProducer contract) ----------
    def rdt_gather_group(self, names: list[str]) -> None:
        """Collectively all-gather one layer-aligned group and publish it.

        Every FSDP rank runs the IDENTICAL ordered plan (run_gather_plan), so
        the per-name ``full_tensor()`` collectives rendezvous safely. Every
        rank caches the gathered tensors so every rank can serve its 1:1
        inference worker's pulls (load-balancing, not just rank 0)."""
        entries: dict[str, torch.Tensor] = {}
        for name in names:
            entries[name] = self._param_lookup[name].full_tensor()
        self.rdt_publish_gathered(entries)


def create_async_engine(**kwargs):
    """Create an AsyncLLMEngine directly (no subclass needed)."""
    engine_args = vllm.AsyncEngineArgs(**kwargs)
    vllm_config = engine_args.create_engine_config()
    executor_class = Executor.get_class(vllm_config)
    return vllm.AsyncLLMEngine(
        vllm_config=vllm_config,
        executor_class=executor_class,
        log_requests=engine_args.enable_log_requests,
        log_stats=not engine_args.disable_log_stats,
    )


async def generate_batch(engine, prompts, sampling_params):
    """Generate completions for a batch of prompts."""

    async def gen_one(prompt):
        output = None
        async for request_output in engine.generate(
            {"prompt": prompt},
            sampling_params,
            request_id=str(uuid.uuid4()),
        ):
            output = request_output
        return output

    return await asyncio.gather(*[gen_one(p) for p in prompts])


async def main():
    # Pin Ray workers to the driver's Python so they pick up the venv
    # (mirrors the boilerplate from the other RDT examples).
    runtime_env: dict[str, object] = {"py_executable": sys.executable}
    forwarded = {
        k: os.environ[k]
        for k in ("NCCL_CUMEM_ENABLE", "VLLM_NCCL_SO_PATH", "LD_PRELOAD")
        if k in os.environ
    }
    if forwarded:
        runtime_env["env_vars"] = forwarded
    # On an Anyscale workspace a Ray head node is already running, and
    # ``RAY_OVERRIDE_RESOURCES`` pins object_store_memory to the full /dev/shm
    # size. Attach to that managed cluster rather than starting a fresh node
    # (a fresh start trips Ray's "object store exceeds /dev/shm" guard, and
    # object_store_memory must not be passed when connecting to an existing
    # cluster).
    # When launched as a Ray task on a GPU node (so the driver can detect the
    # CUDA platform — the head node has no GPU), Ray is already initialized; only
    # init here when run directly as a top-level driver.
    if not ray.is_initialized():
        ray.init(
            address="auto",
            runtime_env=runtime_env,
            namespace=RAY_NAMESPACE,
        )

    # Multi-node: no shared filesystem, so we don't snapshot_download on the
    # (GPU-less) driver. Each GPU node has the model pre-cached in its local HF
    # cache; passing the bare repo id lets the trainer (from_pretrained) and the
    # vLLM workers (config only, since load_format="dummy") resolve it from the
    # node-local cache without a driver-side 60GB download.
    local_model_path = MODEL_NAME
    print(f"[init] Using model id {local_model_path} (pre-cached on each node)")

    # Pin all FSDP ranks to a SINGLE node (STRICT_PACK) so the trainer NCCL
    # all-gather stays intra-node (NVLink) and the "all trainer ranks
    # co-located" requirement holds. Filling one 8-GPU node with the trainer
    # also forces vLLM's 8 inference workers onto the *other* 8-GPU node (the
    # only remaining GPU capacity), which co-locates inference as well.
    trainer_pg = placement_group(
        [{"GPU": 1, "CPU": 1}] * FSDP_WORLD_SIZE, strategy="STRICT_PACK"
    )
    ray.get(trainer_pg.ready())

    # The FSDP rank-0 TCP-store rendezvous runs on the trainer node, so
    # MASTER_ADDR must be that node's IP (not the driver's). Resolve the node
    # the placement group landed on, and pick a port that is free *there*.
    pg_node_id = next(iter(placement_group_table(trainer_pg)["bundles_to_node_id"].values()))
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

    # Every rank is a named RDT producer so inference workers can spread their
    # pulls across all ranks (static 1:1). Rank 0 keeps the canonical name for
    # back-compat; ranks 1+ get a ``_rank{N}`` suffix. ``producer_names`` is
    # ordered by rank and handed to the engine's ``trainer_actor_names``.
    fsdp_workers = []
    for rank in range(FSDP_WORLD_SIZE):
        common_args = (
            local_model_path,
            rank,
            FSDP_WORLD_SIZE,
            fsdp_master_addr,
            fsdp_master_port,
        )
        handle = FSDPTrainWorker.options(
            name=trainer_actor_name(rank),
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=trainer_pg,
                placement_group_bundle_index=rank,
            ),
        ).remote(*common_args)
        fsdp_workers.append(handle)
    producer_names = [trainer_actor_name(r) for r in range(FSDP_WORLD_SIZE)]
    ray.get([w.get_rank.remote() for w in fsdp_workers])
    print(f"[init] {FSDP_WORLD_SIZE} FSDP training workers ready.")

    print("[engine] Creating AsyncLLMEngine...")
    engine = create_async_engine(
        model=local_model_path,
        enforce_eager=True,
        tensor_parallel_size=INFERENCE_TP_SIZE,
        data_parallel_size=INFERENCE_DP_SIZE,
        enable_expert_parallel=True,
        distributed_executor_backend="ray",
        data_parallel_backend="ray",
        weight_transfer_config=WeightTransferConfig(backend="sharded_rdt"),
        load_format="dummy",
        gpu_memory_utilization=0.7,
    )
    print("[engine] AsyncLLMEngine created.")

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0)

    print("[generate] Generating with dummy weights...")
    outputs = await generate_batch(engine, prompts, sampling_params)
    print("-" * 60)
    print("BEFORE weight sync (dummy weights):")
    print("-" * 60)
    for output in outputs:
        print(f"Prompt: {output.prompt!r}")
        print(f"Generated: {output.outputs[0].text!r}")
        print("-" * 60)

    # ---- Weight transfer ----
    # Fetch the trainer's full parameter metadata *before* init: the sharded RDT
    # engine bakes its replay plan over all of these during
    # init_weight_transfer_engine. The driver also partitions the flat name list
    # into layer-aligned groups for its own per-layer gather/free schedule;
    # update_weights then passes each group's gathered names.
    names, dtype_names, shapes = ray.get(fsdp_workers[0].get_weight_metadata.remote())
    layer_groups = layerwise_groups(names)
    print(
        f"[sync] {len(names)} params -> {len(layer_groups)} gather groups "
        f"(max group size = {max(len(g) for g in layer_groups)} params)."
    )

    # Truncate the worker-side consumer timing file BEFORE init: the engine
    # writes its per-worker RPC-baseline record (bare Ray RTT + tiny-nixl RTT)
    # during init_weight_transfer_engine, and per-pull records during the sync
    # loop. Both must survive into the driver's end-of-run read.
    import json

    consumer_file = "/tmp/rdt_profile/consumer.jsonl"
    os.makedirs(os.path.dirname(consumer_file), exist_ok=True)
    open(consumer_file, "w").close()

    print("[sync] Initializing sharded RDT engine (dry-run bake)...")
    _init_t0 = time.perf_counter()
    await engine.init_weight_transfer_engine(
        WeightTransferInitRequest(
            init_info=asdict(
                ShardedRDTWeightTransferInitInfo(
                    trainer_actor_names=producer_names,
                    trainer_actor_namespace=RAY_NAMESPACE,
                    names=names,
                    dtype_names=dtype_names,
                    shapes=shapes,
                    # ring depth x chunks per group: keep K x (group/S) under
                    # the fabric's address-translation reach (~2-3 GB/flow)
                    num_rdt_buffers=int(os.environ.get("NUM_RDT_BUFFERS", "2")),
                    layerwise_split=int(os.environ.get("LAYERWISE_SPLIT", "1")),
                    arena_presize_gb=float(
                        os.environ.get("RDT_ARENA_PRESIZE_GB", "0")
                    ),
                    pack_check=os.environ.get("RDT_PACK_CHECK", "0") == "1",
                )
            )
        )
    )
    _init_seconds = time.perf_counter() - _init_t0
    print(f"[sync] init_weight_transfer_engine (incl. bake) took {_init_seconds:.3f} s")

    print("[sync] Pausing generation...")
    await engine.pause_generation(mode="abort")

    # Run SYNC_ITERS back-to-back syncs. The plans were baked at init, so every
    # sync is a replay. Each iter brackets the per-group loop in its own
    # start/finish_weight_update, since initialize/finalize_layerwise_reload run
    # per sync.
    for sync_iter in range(SYNC_ITERS):
        # Zero produce counters on every rank so each iter is timed
        # independently (every rank now produces a share of the slices).
        ray.get([w.reset_produce_timing.remote() for w in fsdp_workers])
        # Zero per-process NIXL counters too, so producer-side registration time
        # is attributed per iter (bake-iter vs replay-iters).
        ray.get([w.reset_nixl_timing.remote() for w in fsdp_workers])

        await engine.start_weight_update(is_checkpoint_format=True)

        # One update_weights for the whole sync: the trainers self-pace their
        # gathers from the (identical) plan — per-group full_tensor collectives
        # rendezvous safely — and the engine frees each group's gather via
        # free_gather as its chunks finish. The chunk pipeline never drains
        # until the sync ends (no per-group call boundaries or worker rejoins).
        print(f"[sync] iter {sync_iter} [REPLAY]: gather + update_weights...")
        _sync_t0 = time.perf_counter()
        run_refs = [w.run_gather_plan.remote(layer_groups) for w in fsdp_workers]
        gi = ShardedRDTWeightTransferUpdateInfo(
            names=[n for g in layer_groups for n in g],
            group_lens=[len(g) for g in layer_groups],
        )
        await engine.update_weights(
            WeightTransferUpdateRequest(update_info=asdict(gi))
        )
        ray.get(run_refs)  # surfaces gather errors; ~0s
        await engine.finish_weight_update()
        _sync_seconds = time.perf_counter() - _sync_t0

        # ---- Per-iter profiling summary ----
        # Trainer-side: produce calls (RPC count) and slice+clone time, summed
        # across ALL ranks (each rank now produces a share of the slices).
        # Because the ranks clone in PARALLEL, the wall-clock-relevant term is
        # the SLOWEST rank's slice+clone time (``slice_s_max``), not the sum;
        # aggregate throughput = total bytes / slowest-rank time. We also print
        # per-rank GiB to show how evenly the 1:1 routing balanced the load.
        ptimings = ray.get([w.get_produce_timing.remote() for w in fsdp_workers])
        gib = sum(p["bytes"] for p in ptimings) / (1024**3)
        per_rank_gib = [p["bytes"] / (1024**3) for p in ptimings]
        slice_s_max = max(p["slice_seconds"] for p in ptimings)
        calls = sum(p["calls"] for p in ptimings)
        specs = sum(p["specs"] for p in ptimings)
        wait_max = max(p["wait_seconds"] for p in ptimings)
        # Producer-side NIXL counters (this iter): registration is the cost that
        # fires for every fresh clone buffer. transfer_seconds should be ~0 here
        # (producers are passive RDMA responders, never call transfer()).
        ntimings = ray.get([w.get_nixl_timing.remote() for w in fsdp_workers])
        reg_s_max = max(n["register_seconds"] for n in ntimings)
        reg_calls = sum(n["register_calls"] for n in ntimings)
        prod_xfer = sum(n["transfer_seconds"] for n in ntimings)
        # Producer-side post-return extract = cuda.sync + register + descs.
        # Isolate the per-RPC cuda.synchronize() by subtracting register+descs.
        extract_s_max = max(n["extract_seconds"] for n in ntimings)
        sync_per_rank = [
            n["extract_seconds"] - n["register_seconds"] - n["descs_seconds"]
            for n in ntimings
        ]
        cuda_sync_max = max(sync_per_rank)
        method_s_max = max(p["method_seconds"] for p in ptimings)
        print("=" * 60)
        print(
            f"[profile] iter {sync_iter} [REPLAY]"
        )
        if sync_iter == 0:
            print(f"[profile] init_weight_transfer_engine : {_init_seconds:.3f} s")
        print(f"[profile] total weight-sync wall time : {_sync_seconds:.3f} s")
        print(f"[profile] trainer produce calls (all) : {calls}")
        print(f"[profile] trainer specs (slices) total: {specs}")
        print(f"[profile] trainer gather-cache wait    : {wait_max:.3f} s (max)")
        print(f"[profile] trainer slice+clone (slowest): {slice_s_max:.3f} s")
        print(
            f"[profile] producer NIXL register (slow): {reg_s_max:.3f} s "
            f"({reg_calls} regs; producer xfer={prod_xfer:.3f}s should be ~0)"
        )
        print(
            f"[profile] producer method total (slow) : {method_s_max:.3f} s "
            f"(time inside rdt_produce_weights_batched: wait+clone)"
        )
        print(
            f"[profile] producer extract (slow)      : {extract_s_max:.3f} s "
            f"of which cuda.sync ~= {cuda_sync_max:.3f} s  <-- scales w/ GPU work?"
        )
        print(f"[profile] bytes produced (all ranks)   : {gib:.3f} GiB")
        print(
            "[profile] per-rank GiB                 : "
            + ", ".join(f"r{r}={g:.2f}" for r, g in enumerate(per_rank_gib))
        )
        if slice_s_max > 0:
            print(
                f"[profile] agg clone throughput         : "
                f"{gib / slice_s_max:.1f} GiB/s"
            )
        print("=" * 60)

    # ---- Consumer-side summary: per-worker sums over the whole run ----
    from collections import defaultdict

    per_pid: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    try:
        with open(consumer_file) as f:
            for line in f:
                rec = json.loads(line)
                for k, v in rec.items():
                    if isinstance(v, (int, float)):
                        per_pid[rec["pid"]][k] += v
    except FileNotFoundError:
        pass  # driver not co-located with the inference node's timing file
    print("=" * 60)
    print("[profile] CONSUMER per-worker totals (all iters)")
    for pid, a in sorted(per_pid.items()):
        xfer = a.get("transfer_seconds", 0.0)
        gb = a.get("bytes", 0) / 1e9
        rate = gb / xfer if xfer else 0.0
        print(f"[profile]   pid={pid}  pull={a.get('pull', 0):.2f}s  "
              f"transfer={xfer:.2f}s ({rate:.1f} GB/s)  "
              f"process={a.get('process', 0):.2f}s")
    print("=" * 60)

    print("[sync] Resuming generation...")
    await engine.resume_generation()

    print("[generate] Generating with synced weights...")
    outputs_updated = await generate_batch(engine, prompts, sampling_params)
    print("-" * 60)
    print("AFTER weight sync (real weights):")
    print("-" * 60)
    for output in outputs_updated:
        print(f"Prompt: {output.prompt!r}")
        print(f"Generated: {output.outputs[0].text!r}")
        print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main())
