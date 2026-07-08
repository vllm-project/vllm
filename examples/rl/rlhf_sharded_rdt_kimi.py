# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sharded-RDT weight sync for Kimi K2 (DeepseekV3, FP8 block-quant) — 8 trainer
GPUs -> 8 vLLM inference GPUs, both fleets one node each.

Differs from rlhf_sharded_rdt_fsdp_ep.py (Qwen/bf16) in the TRAINER: Kimi is a
~1 TB FP8 checkpoint that (a) must not be materialized whole on any GPU and
(b) must be served in the SAME fp8 + block-scale form vLLM consumes (HF would
dequantize to bf16). So the trainer is NOT an HF model — it is a raw, sharded
FP8 checkpoint server: every checkpoint tensor (fp8 .weight + fp32
.weight_scale_inv, individual experts, separate q_a/kv_a) is loaded to GPU up
front, FSDP-sharded (Shard(0)); the sync gathers per layer and serves the
individual checkpoint names vLLM's DeepseekV3 load_weights fuses internally.

The vLLM weight-transfer engine (bake/replay) is unchanged: the bake drives
vLLM's own load_weights over the checkpoint names, so fp8 weight + weight_scale_inv
flow through as ordinary slices into the fp8/fp32 layerwise-reload destinations.

PRODUCER: the shared RDTShardedProducer (rdt_producer.py) — self-paced per-group
all-gathers (full_tensor() into fresh buffers, published as views) + the packed
serve ring. Intentionally sharding-agnostic: the serve copies each op-chain slice
into the packed arena, so it drops in on top of a stock fully_shard'd trainer
where we don't control the gather layout.

DRIVER: one update_weights per sync (names + group_lens); the engine pipelines
chunked packed pulls over the receive ring and frees each group's gather on the
producer as its chunks finish. See multi_node_rdt.md for the optimization history.
"""
import asyncio
import glob
import json
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

# Local module (ships with the example via runtime_env working_dir): the
# two-level critical-path profiler. See rdt_profile.py for the design.
from rdt_producer import RDTShardedProducer, layerwise_groups
from rdt_profile import CriticalPathProfiler, make_consumer_file_tasks

MODEL_NAME = "moonshotai/Kimi-K2-Instruct"
TRAINER_ACTOR_NAME = "sharded_rdt_kimi_trainer"
RAY_NAMESPACE = "sharded_rdt_kimi_example"

FSDP_WORLD_SIZE = 8
INFERENCE_TP_SIZE = 1
INFERENCE_DP_SIZE = 8
NUM_INFERENCE_CONSUMERS = INFERENCE_TP_SIZE * INFERENCE_DP_SIZE
SYNC_ITERS = int(os.environ.get("RDT_SYNC_ITERS", "2"))

_ST_DTYPE = {"F8_E4M3": torch.float8_e4m3fn, "BF16": torch.bfloat16,
             "F32": torch.float32, "F16": torch.float16}

def trainer_actor_name(rank: int) -> str:
    return TRAINER_ACTOR_NAME if rank == 0 else f"{TRAINER_ACTOR_NAME}_rank{rank}"


@ray.remote(num_gpus=1, max_concurrency=8, enable_tensor_transport=True)
class KimiTrainWorker(RDTShardedProducer):
    """Raw FP8 checkpoint server (one per GPU), sharded with STANDARD ``fully_shard``.
    Holds every checkpoint tensor as a fully_shard'd param (Shard(0) DTensor) on GPU;
    gathers per layer via ``full_tensor()`` and serves slices over NIXL. No HF model /
    no forward; gather + serve only."""

    def __init__(self, model_name, rank, world_size, master_addr, master_port):
        self.rank = rank
        self.world_size = world_size
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.accelerator.set_device_index(0)

        self._load_checkpoint(model_name)

        # Shared sharded-RDT producer state: gather cache + serve ring +
        # scoped-sync + gc.freeze + timing (see examples/rl/rdt_producer.py).
        self.init_rdt_producer()
        # physical (fused stack / individual) -> gathered full tensor; the
        # per-name cache entries are VIEWS into these (freed together).
        self._cache_phys: dict[str, torch.Tensor] = {}

        from vllm.distributed.weight_transfer._nixl_profile import install_nixl_timing
        install_nixl_timing()

    def _load_checkpoint(self, model_name):
        """Load the FP8 checkpoint as a STANDARD FSDP2 model.

        Build a plain ``nn.Module`` whose parameters are the checkpoint tensors
        (fp8 ``.weight`` + fp32 ``.weight_scale_inv``; routed experts FUSED per
        (layer, proj, wkind) into ``[E, *expert]`` params, like an HF MoE model's
        fused experts), then call ``fully_shard`` on it and stream each rank's shard
        from disk. There is NO custom DTensor construction: ``fully_shard`` shards
        every param ``Shard(0)`` and ``full_tensor()`` reconstructs it byte-exact --
        exactly the contract a real RL trainer's FSDP2 model exposes, so this drops
        in wherever we don't control the gather layout. We still SERVE INDIVIDUAL
        checkpoint names (views/slices of the fused params) via ``_name_to_src`` --
        the contract vLLM's DeepseekV3 load_weights expects. Experts are fused only
        so ``full_tensor()`` is ~20 large all-gathers/layer, not ~2300 tiny ones
        (same reason HF MoE models fuse them); that's a data-layout choice, not
        custom sharding.
        """
        import re
        from collections import OrderedDict

        import torch.nn as nn
        from safetensors import safe_open
        from torch.distributed.tensor._utils import (
            compute_local_shape_and_global_offset,
        )
        snap = glob.glob(
            f"/root/.cache/huggingface/hub/models--{model_name.replace('/','--')}/snapshots/*"
        )[0]
        wmap = json.load(open(os.path.join(snap, "model.safetensors.index.json")))[
            "weight_map"
        ]
        # Exclude rotary_emb.inv_freq (vLLM recomputes it) and layerwise SKIP_TENSORS
        # (e.g. e_score_correction_bias -- stays a real param -> would break the bake's
        # meta-copy; not part of the layerwise reload path so can't ride the sync).
        _SKIP = {"_expert_map", "expert_mask", "expert_global_to_physical",
                 "expert_physical_to_global", "expert_local_to_global",
                 "e_score_correction_bias"}
        names = [n for n in wmap
                 if "rotary_emb.inv_freq" not in n and n.rsplit(".", 1)[-1] not in _SKIP]

        self._handles: dict = {}

        def H(k):
            fn = wmap[k]
            if fn not in self._handles:
                self._handles[fn] = safe_open(
                    os.path.join(snap, fn), framework="pt", device="cuda:0"
                )
            return self._handles[fn]

        # metadata (per INDIVIDUAL name) for the engine init_info / name contract
        self.weight_names = names
        self.weight_dtype_names = []
        self.weight_shapes = []
        for n in names:
            sl = H(n).get_slice(n)
            self.weight_shapes.append(list(sl.get_shape()))
            self.weight_dtype_names.append(str(_ST_DTYPE[sl.get_dtype()]).split(".")[-1])

        # Parse names: routed experts -> per-(layer,proj,wkind) FUSED param; else
        # individual. _name_to_src maps each served (individual) name to its physical
        # param + expert row; unchanged from the serve's point of view.
        expert_re = re.compile(
            r"^(.*\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)$"
        )
        self._name_to_src: dict[str, tuple[str, int | None]] = {}
        stacks: dict[str, dict[int, str]] = {}  # phys_key -> {expert_idx: ckpt_name}
        individuals: list[str] = []
        for n in names:
            m = expert_re.match(n)
            if m:
                pk = f"{m.group(1)}.{m.group(3)}.{m.group(4)}"  # synthetic fused key
                self._name_to_src[n] = (pk, int(m.group(2)))
                stacks.setdefault(pk, {})[int(m.group(2))] = n
            else:
                self._name_to_src[n] = (n, None)
                individuals.append(n)

        # Physical param spec per phys key: (pk, full_shape, dtype, loader). loader is
        # ("indiv", ckpt_name) or ("stack", {expert_idx: ckpt_name}).
        specs: list[tuple] = []
        for n in individuals:
            sl = H(n).get_slice(n)
            specs.append((n, tuple(sl.get_shape()), _ST_DTYPE[sl.get_dtype()],
                          ("indiv", n)))
        for pk, idx_map in stacks.items():
            E = len(idx_map)
            assert set(idx_map) == set(range(E)), f"non-contiguous experts in {pk}"
            sl = H(idx_map[0]).get_slice(idx_map[0])
            specs.append((pk, (E,) + tuple(sl.get_shape()),
                          _ST_DTYPE[sl.get_dtype()], ("stack", idx_map)))

        # Group phys by decoder layer (pre/post -> -1) and build one ParameterDict
        # submodule per group; fully_shard each submodule + the root (the standard
        # FSDP2 pattern -- see rlhf_sharded_rdt_fsdp_ep.py). Params start on META so
        # nothing is allocated before sharding; keyed by index to avoid name mangling.
        def _lyr(pk):
            return (int(pk[len("model.layers."):].split(".", 1)[0])
                    if pk.startswith("model.layers.") else -1)
        by_layer: "OrderedDict[int, list]" = OrderedDict()
        for s in specs:
            by_layer.setdefault(_lyr(s[0]), []).append(s)

        root = nn.Module()
        root.groups = nn.ModuleList()
        submods: list[tuple] = []
        for _lyr_idx, group_specs in by_layer.items():
            sub = nn.Module()
            pd = nn.ParameterDict()
            for j, (pk, shape, dt, _loader) in enumerate(group_specs):
                pd[str(j)] = nn.Parameter(
                    torch.empty(shape, dtype=dt, device="meta"), requires_grad=False)
            sub.pd = pd
            root.groups.append(sub)
            submods.append((sub, group_specs))
        for sub, _ in submods:
            fully_shard(sub)
        fully_shard(root)

        # Allocate ONLY each rank's local shards, then stream them from disk. The full
        # checkpoint is never materialized on any GPU. Uses FSDP2's own local
        # shape/offset so any padding rows stay zero (full_tensor() strips them).
        root.to_empty(device="cuda")
        self.model = root
        self._phys: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for sub, group_specs in submods:
                for j, (pk, shape, dt, loader) in enumerate(group_specs):
                    param = sub.pd[str(j)]
                    self._phys[pk] = param
                    local = param.to_local().detach()
                    lshape, goff = compute_local_shape_and_global_offset(
                        param.shape, param.device_mesh, param.placements)
                    local.zero_()
                    n0 = lshape[0]  # real rows this rank owns along the sharded dim
                    if n0 == 0:
                        continue
                    kind, info = loader
                    if kind == "indiv":
                        local[:n0].copy_(H(info).get_slice(info)[goff[0]:goff[0] + n0])
                    else:  # fused expert stack: dim-0 rows ARE experts
                        for i in range(n0):
                            cn = info[goff[0] + i]
                            local[i].copy_(H(cn).get_tensor(cn))
        torch.cuda.synchronize()

    def get_rank(self):
        return self.rank

    def ping(self):
        return None

    def get_weight_metadata(self):
        return self.weight_names, self.weight_dtype_names, self.weight_shapes

    # ---- gather / free hooks (RDTShardedProducer contract) ----
    def rdt_gather_group(self, names):
        """All-gather one group's unique PHYSICAL tensors (fused expert stacks +
        individuals) via ``DTensor.full_tensor()`` into FRESH buffers, then
        publish each requested name as a view (stack[expert_idx] or the full
        tensor). ``full_tensor()`` is FSDP's own collective, so this works for
        any DTensor sharding a real trainer produces."""
        phys_keys: list[str] = []
        seen: set[str] = set()
        for n in names:
            pk = self._name_to_src[n][0]
            if pk not in seen:
                seen.add(pk)
                phys_keys.append(pk)
        gathered: dict[str, torch.Tensor] = {}
        for pk in phys_keys:
            gathered[pk] = self._phys[pk].full_tensor()  # all-gather -> FRESH buf
        with self._cache_cond:
            self._cache_phys.update(gathered)
        entries = {}
        for n in names:
            pk, idx = self._name_to_src[n]
            entries[n] = gathered[pk] if idx is None else gathered[pk][idx]
        self.rdt_publish_gathered(entries)

    def rdt_free_group(self, names):
        """Also drop the gathered physical stacks the name views alias."""
        super().rdt_free_group(names)
        pks = {self._name_to_src[n][0] for n in names}
        for pk in pks:
            self._cache_phys.pop(pk, None)


def create_async_engine(**kwargs):
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
    async def gen_one(prompt):
        out = None
        async for o in engine.generate({"prompt": prompt}, sampling_params,
                                       request_id=str(uuid.uuid4())):
            out = o
        return out
    return await asyncio.gather(*[gen_one(p) for p in prompts])


async def main():
    runtime_env: dict[str, object] = {"py_executable": sys.executable}
    if not ray.is_initialized():
        ray.init(address="auto", runtime_env=runtime_env, namespace=RAY_NAMESPACE)

    local_model_path = MODEL_NAME
    print(f"[init] Kimi trainer = raw fp8 sharded checkpoint server", flush=True)

    # Pin trainer to ONE node (STRICT_PACK) -> inference lands on the other node.
    trainer_pg = placement_group([{"GPU": 1, "CPU": 1}] * FSDP_WORLD_SIZE,
                                 strategy="STRICT_PACK")
    ray.get(trainer_pg.ready())
    pg_node_id = next(iter(placement_group_table(trainer_pg)["bundles_to_node_id"].values()))
    fsdp_master_addr = next(n["NodeManagerAddress"] for n in ray.nodes()
                            if n["NodeID"] == pg_node_id)

    # The 8 inference (consumer) workers land on the OTHER GPU node and write
    # their per-pull timing to that node's local /tmp. With no shared FS and the
    # driver possibly on the trainer node, collect that file via Ray tasks pinned
    # to the inference node (falls back to the trainer node in a single-node run).
    inference_ip = next((n["NodeManagerAddress"] for n in ray.nodes()
                         if n["Alive"] and n["Resources"].get("GPU", 0) > 0
                         and n["NodeManagerAddress"] != fsdp_master_addr),
                        fsdp_master_addr)

    @ray.remote(num_cpus=0, scheduling_strategy=PlacementGroupSchedulingStrategy(
        placement_group=trainer_pg))
    def _free_port():
        return get_open_port()
    fsdp_master_port = ray.get(_free_port.remote())
    print(f"[init] trainer on {fsdp_master_addr}:{fsdp_master_port}", flush=True)

    workers = []
    for rank in range(FSDP_WORLD_SIZE):
        h = KimiTrainWorker.options(
            name=trainer_actor_name(rank),
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=trainer_pg, placement_group_bundle_index=rank),
        ).remote(local_model_path, rank, FSDP_WORLD_SIZE,
                 fsdp_master_addr, fsdp_master_port)
        workers.append(h)
    producer_names = [trainer_actor_name(r) for r in range(FSDP_WORLD_SIZE)]
    ray.get([w.get_rank.remote() for w in workers])
    print(f"[init] {FSDP_WORLD_SIZE} Kimi trainer workers ready (weights resident).",
          flush=True)

    print("[engine] Creating AsyncLLMEngine (Kimi fp8, DP8/EP8)...", flush=True)
    engine = create_async_engine(
        model=local_model_path,
        trust_remote_code=True,
        enforce_eager=True,
        tensor_parallel_size=INFERENCE_TP_SIZE,
        data_parallel_size=INFERENCE_DP_SIZE,
        enable_expert_parallel=True,
        distributed_executor_backend="ray",
        data_parallel_backend="ray",
        weight_transfer_config=WeightTransferConfig(backend="sharded_rdt"),
        load_format="dummy",
        max_model_len=2048,
        max_num_seqs=4,
        gpu_memory_utilization=0.90,
        kv_cache_dtype="fp8",
    )
    print("[engine] AsyncLLMEngine created.", flush=True)

    prompts = ["The capital of France is", "The future of AI is"]
    sampling_params = SamplingParams(temperature=0, max_tokens=16)

    print("[generate] BEFORE sync (dummy weights):", flush=True)
    for o in await generate_batch(engine, prompts, sampling_params):
        print(f"  {o.prompt!r} -> {o.outputs[0].text!r}", flush=True)

    names, dtype_names, shapes = ray.get(workers[0].get_weight_metadata.remote())
    layer_groups = layerwise_groups(names)
    print(f"[sync] {len(names)} params -> {len(layer_groups)} gather groups "
          f"(max {max(len(g) for g in layer_groups)} params/group).", flush=True)

    consumer_file = "/tmp/rdt_profile/consumer.jsonl"
    read_consumer, truncate_consumer = make_consumer_file_tasks(
        inference_ip, consumer_file)

    print("[sync] init_weight_transfer_engine (bake)...", flush=True)
    _t0 = time.perf_counter()
    await engine.init_weight_transfer_engine(WeightTransferInitRequest(
        init_info=asdict(ShardedRDTWeightTransferInitInfo(
            trainer_actor_names=producer_names,
            trainer_actor_namespace=RAY_NAMESPACE,
            names=names, dtype_names=dtype_names, shapes=shapes,
            # [RDT-RING] receive/serve ring depth K x chunk split S: keep
            # K x (layer_bytes / S) under the fabric's address-translation
            # reach (~2-3 GB/flow here). The producer reads NUM_RDT_BUFFERS
            # and RDT_ARENA_PRESIZE_GB from env directly (init_info fields
            # exist because env vars don't reach vLLM worker procs).
            num_rdt_buffers=int(os.environ.get("NUM_RDT_BUFFERS", "2")),
            layerwise_split=int(os.environ.get("LAYERWISE_SPLIT", "1")),
            arena_presize_gb=float(os.environ.get("RDT_ARENA_PRESIZE_GB", "2.6")),
            pack_check=os.environ.get("RDT_PACK_CHECK", "0") == "1"))))
    print(f"[sync] bake took {time.perf_counter()-_t0:.1f}s", flush=True)

    await engine.pause_generation(mode="abort")

    for sync_iter in range(SYNC_ITERS):
        # Clean this iter's per-process counters: producer (on trainer actors),
        # consumer NIXL (on trainer actors), and the consumer jsonl (on the
        # inference node). After truncate, the file holds exactly this iter.
        truncate_consumer()
        ray.get([w.reset_produce_timing.remote() for w in workers])
        ray.get([w.reset_nixl_timing.remote() for w in workers])

        cp = CriticalPathProfiler(sync_iter)
        cp.begin()
        with cp.timed("start_weight_update"):
            await engine.start_weight_update(is_checkpoint_format=True)
        print(f"[sync] iter {sync_iter}: gather + update_weights...", flush=True)
        # One update_weights for the whole sync. The trainers self-pace
        # their gathers from the (identical) plan — the per-group collectives
        # rendezvous safely — and the ENGINE frees each group's gather via
        # free_gather after its last chunk, so the driver has no per-group
        # loop at all: no per-call first-chunk produce_wait, no per-group
        # 8-worker rejoin, and the chunk pipeline never drains until the sync
        # ends.
        run_refs = [w.run_gather_plan.remote(layer_groups) for w in workers]
        gi = ShardedRDTWeightTransferUpdateInfo(
            names=[n for g in layer_groups for n in g],
            group_lens=[len(g) for g in layer_groups])
        with cp.timed("update_weights"):
            await engine.update_weights(
                WeightTransferUpdateRequest(update_info=asdict(gi)))
        # Surfaces gather errors; ~0s (all gathers precede the last chunks).
        with cp.timed("gather_tail"):
            ray.get(run_refs)
        with cp.timed("finish_weight_update"):
            await engine.finish_weight_update()
        cp.finish()
        pt = ray.get([w.get_produce_timing.remote() for w in workers])
        nt = ray.get([w.get_nixl_timing.remote() for w in workers])
        gib = sum(p["bytes"] for p in pt) / (1024**3)
        print(cp.report(consumer_jsonl=read_consumer(), producer_timings=pt,
                        producer_nixl=nt, n_groups=len(layer_groups)),
              flush=True)
        print(f"[profile]   bytes moved (all producers)={gib:.2f}GiB  "
              f"produce calls={sum(p['calls'] for p in pt)}", flush=True)

    await engine.resume_generation()
    print("[generate] AFTER sync (real weights):", flush=True)
    for o in await generate_batch(engine, prompts, sampling_params):
        print(f"  {o.prompt!r} -> {o.outputs[0].text!r}", flush=True)
    print("main() returned", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
