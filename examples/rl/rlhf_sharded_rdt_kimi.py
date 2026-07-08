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

PRODUCER SERVE PATH — SIMPLE (copy into a staging buffer). Each sync, gather_layer
all-gathers each physical tensor via full_tensor() into a fresh buffer, and the
serve replays each name's view op-chain then COPIES the slice into a reused,
registered per-dtype serve arena (the staging buffer). This is intentionally
sharding-agnostic: it does NOT require the served slice to be a zero-copy view of a
persistent, pre-registered gather buffer, so it drops in on top of a stock
fully_shard'd trainer where we don't control the gather layout. It is the revert of
the "slice baking" optimization (persistent registered gather buffer + cached serve
plan + zero-copy views), which was faster (~0.05 s vs ~1.1 s producer serve/iter)
but coupled the serve to a bespoke gather. See multi_node_rdt.md for the profiled
critical-path delta.
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

_ALLOWED_OPS = frozenset({"narrow", "view", "reshape", "__getitem__", "unsqueeze",
                          "squeeze", "transpose", "t", "permute", "flatten",
                          "contiguous", "chunk"})


def trainer_actor_name(rank: int) -> str:
    return TRAINER_ACTOR_NAME if rank == 0 else f"{TRAINER_ACTOR_NAME}_rank{rank}"


def _layerwise_groups(names: list[str]) -> list[list[str]]:
    """Partition flat checkpoint names into pre / per-decoder-layer / post groups
    (same scheme as the Qwen example; keys on ``model.layers.<N>.``)."""
    pre: list[str] = []
    layers: dict[int, list[str]] = {}
    post: list[str] = []
    seen = False
    for n in names:
        if n.startswith("model.layers."):
            seen = True
            idx = int(n[len("model.layers."):].split(".", 1)[0])
            layers.setdefault(idx, []).append(n)
        elif not seen:
            pre.append(n)
        else:
            post.append(n)
    groups: list[list[str]] = []
    if pre:
        groups.append(pre)
    for i in sorted(layers):
        groups.append(layers[i])
    if post:
        groups.append(post)
    return groups


@ray.remote(num_gpus=1, max_concurrency=8, enable_tensor_transport=True)
class KimiTrainWorker:
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

        # [RDT-GC] Straggler FIX (measured): ~0.4% of produce calls stalled 0.24-0.6s
        # with the RDMA transfer dead-normal -> stop-the-world gen-2 GC scanning the
        # large resident tensor graph (139k params) during the 148k-slice produce.
        # gc.freeze() moves that static graph into a permanent gen never scanned, so
        # gen-2 collections stay cheap (real garbage still collected). Measured: p99
        # pull 0.240->0.160s, >0.3s stragglers 0.37%->0.20%, median unchanged. Default
        # on; RDT_GC=0 disables the fix (A/B), =2 fully disables GC (aggressive test).
        import gc as _gc
        _gcmode = os.environ.get("RDT_GC", "1")
        if _gcmode == "1":
            _gc.collect(); _gc.freeze()
        elif _gcmode == "2":
            _gc.disable()

        # gather cache (bounded: freed per layer group) + sync
        self._cache: dict[str, torch.Tensor] = {}       # individual name -> view
        self._cache_phys: dict[str, torch.Tensor] = {}  # physical (stack/indiv) -> full
        self._cache_cond = threading.Condition()
        self._gather_error: BaseException | None = None

        from ray.experimental import register_nixl_memory
        self._register_nixl_memory = register_nixl_memory

        # Simple, sharding-agnostic producer serve (reverted from the gbuf +
        # cached-serve-plan "slice baking"): gather_layer all-gathers each phys via
        # full_tensor() into a FRESH buffer; the serve path replays each name's
        # op-chain and copies the resulting slice into this reused, registered
        # per-dtype serve arena (a staging buffer), then serves the arena view. It
        # does NOT assume the served slice is a zero-copy view of a pre-registered
        # region, so it works regardless of how the trainer shards/gathers the model
        # -- the point of the revert: a real RL trainer wraps its model with
        # fully_shard and we don't control the gather layout.
        self._serve_arenas: dict[torch.dtype, torch.Tensor] = {}

        # [RDT-NOSYNC EXPERIMENT] When RDT_NOSYNC=1, replace Ray's whole-device
        # cuda.synchronize in extract_tensor_transport_metadata (which blocks on the
        # OVERLAPPING next-group all_gather, ~1s/iter) with a SCOPED guarantee: run
        # the serve copies on a dedicated stream that waits only on THIS group's
        # gather-completion event, then sync that stream -- so the served buffers are
        # materialized before produce returns, without waiting on unrelated GPU work.
        # (Ray's device sync is env-gated out under the same flag by
        # deploy_rdt_nosync.py.) Default RDT_NOSYNC=0 = stock: copies on the default
        # stream, Ray's device sync guarantees materialization.
        self._scoped_sync = os.environ.get("RDT_NOSYNC", "0") == "1"
        self._serve_stream = torch.cuda.Stream() if self._scoped_sync else None
        self._cache_event: dict[str, "torch.cuda.Event"] = {}

        # [RDT-COALESCE] When set, produce returns ONE contiguous tensor per dtype
        # (the whole packed serve arena) instead of one view per name, so the NIXL
        # transfer is ~2 big descriptors instead of ~309 small ones (~144 fp8 weights
        # + ~144 tiny fp32 scales). The consumer lays its receive arena out with the
        # IDENTICAL per-dtype/aligned/keys-order layout, so the bytes match exactly.
        # Driver mirrors this into init_info.coalesce_dtype_blobs for the consumer.
        self._coalesce = os.environ.get("RDT_COALESCE", "0") == "1"

        # profiling counters
        self._timing_lock = threading.Lock()
        self._produce_calls = self._produce_specs = self._produce_bytes = 0
        self._produce_wait_seconds = self._produce_slice_seconds = 0.0
        self._produce_method_seconds = 0.0
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

    # ---- profiling helpers ----
    def get_nixl_timing(self):
        from vllm.distributed.weight_transfer import _nixl_profile
        return _nixl_profile.snapshot()

    def reset_nixl_timing(self):
        from vllm.distributed.weight_transfer import _nixl_profile
        _nixl_profile.reset()

    def read_nic_counters(self):
        from vllm.distributed.weight_transfer import _nixl_profile
        return _nixl_profile.read_efa_counters()

    def get_produce_timing(self):
        with self._timing_lock:
            return dict(calls=self._produce_calls, specs=self._produce_specs,
                        wait_seconds=self._produce_wait_seconds,
                        slice_seconds=self._produce_slice_seconds,
                        bytes=self._produce_bytes,
                        method_seconds=self._produce_method_seconds)

    def reset_produce_timing(self):
        with self._timing_lock:
            self._produce_calls = self._produce_specs = self._produce_bytes = 0
            self._produce_wait_seconds = self._produce_slice_seconds = 0.0
            self._produce_method_seconds = 0.0

    # ---- gather / serve / free ----
    def gather_layer(self, names, slot):
        """Gather a layer group by all-gathering each unique PHYSICAL tensor (stacks
        + individuals) via ``DTensor.full_tensor()`` into a FRESH buffer, then expose
        each requested name as a view into that fresh tensor (stack[expert_idx] or
        the individual full tensor).

        Simple and sharding-agnostic: ``full_tensor()`` is FSDP's own all-gather, so
        this works for any DTensor sharding a real trainer produces (no assumption of
        a specific layout, no persistent registered gather buffer). ``slot`` is
        accepted for driver-signature compatibility but IGNORED (fresh alloc each
        call; the driver frees the group's refs synchronously after its update
        drains, keeping resident memory bounded to ~2 groups).
        """
        try:
            phys_keys: list[str] = []
            seen: set[str] = set()
            for n in names:
                pk = self._name_to_src[n][0]
                if pk not in seen:
                    seen.add(pk)
                    phys_keys.append(pk)
            gathered: dict[str, torch.Tensor] = {}
            for pk in phys_keys:
                gathered[pk] = self._phys[pk].full_tensor()   # all-gather -> FRESH buf
            # [RDT-NOSYNC] Record a completion event for THIS group's all_gathers so
            # the serve can scope its sync to exactly this group (not the overlapping
            # next-group gather). Only recorded/used when scoped_sync is on.
            gather_ev = None
            if self._scoped_sync:
                gather_ev = torch.cuda.Event()
                gather_ev.record()
            with self._cache_cond:
                self._cache_phys.update(gathered)
                for n in names:
                    pk, idx = self._name_to_src[n]
                    self._cache[n] = gathered[pk] if idx is None else gathered[pk][idx]
                    if gather_ev is not None:
                        self._cache_event[n] = gather_ev
                self._cache_cond.notify_all()
        except BaseException as e:
            with self._cache_cond:
                self._gather_error = e
                self._cache_cond.notify_all()
            raise

    @ray.method(tensor_transport="nixl")
    def rdt_warmup(self):
        return torch.zeros(1, device="cuda:0")

    @ray.method(tensor_transport="nixl")
    def rdt_produce_weights_batched(self, specs):
        _t_m0 = time.perf_counter()
        needed = sorted({n for n, _ in specs})
        _t_w0 = time.perf_counter()
        with self._cache_cond:
            while not all(n in self._cache for n in needed):
                if self._gather_error is not None:
                    raise RuntimeError(f"gather errored before {needed}: "
                                       f"{self._gather_error!r}")
                self._cache_cond.wait()
        wait_s = time.perf_counter() - _t_w0

        _t_s0 = time.perf_counter()
        # Simple serve: REPLAY every name's op-chain (pure Python) to produce the
        # slice, then COPY each slice into the reused, registered per-dtype serve
        # arena (staging buffer) and serve the arena view. No cached serve plan and
        # no zero-copy gbuf views -- the copy makes the serve independent of how the
        # slice was gathered/sharded, at the cost of the op-chain replay (~1.8 s) +
        # the copy (~0.6 s) the "slice baking" had removed.
        sliced: list = []                    # (dtype, off, numel, shape, tensor)
        totals: dict[torch.dtype, int] = {}
        nbytes = 0
        for name, chain in specs:
            t = self._cache[name]
            for op, args, kw in chain:
                if op not in _ALLOWED_OPS:
                    raise ValueError(f"{name!r}: disallowed op {op!r}")
                t = getattr(t, op)(*args, **dict(kw))
            dt, n = t.dtype, t.numel()
            off = totals.get(dt, 0)
            sliced.append((dt, off, n, tuple(t.shape), t))
            totals[dt] = off + ((n + 7) & ~7)
            nbytes += t.element_size() * n
        for dt, total in totals.items():
            a = self._serve_arenas.get(dt)
            if a is None or a.numel() < total:
                a = torch.empty(total, dtype=dt, device="cuda:0")
                self._register_nixl_memory(a)                   # registered once, reused
                self._serve_arenas[dt] = a
        served: list = [None] * len(specs)
        # [RDT-NOSYNC] Scoped-sync serve: enqueue the copies on a dedicated stream
        # that waits ONLY on this group's gather event(s), then synchronize that
        # stream -- the served buffers are materialized before we return (so Ray's
        # whole-device sync can be skipped) WITHOUT blocking on the overlapping
        # next-group all_gather. When scoped_sync is off, ss is None and
        # torch.cuda.stream(None) is a no-op (copies on the default stream, as before).
        ss = self._serve_stream
        if ss is not None:
            for ev in {id(e): e for e in
                       (self._cache_event.get(nm) for nm in needed)
                       if e is not None}.values():
                ss.wait_event(ev)
        with torch.cuda.stream(ss):
            for i, (dt, off, n, shape, t) in enumerate(sliced):
                view = self._serve_arenas[dt][off:off + n].reshape(shape)
                view.copy_(t)                                   # the per-slice copy
                served[i] = view
        if ss is not None:
            ss.synchronize()
        slice_s = time.perf_counter() - _t_s0
        with self._timing_lock:
            self._produce_calls += 1
            self._produce_specs += len(specs)
            self._produce_wait_seconds += wait_s
            self._produce_slice_seconds += slice_s
            self._produce_bytes += nbytes
            self._produce_method_seconds += time.perf_counter() - _t_m0
        if self._coalesce:
            # [RDT-COALESCE] Return one contiguous tensor per dtype (sorted-dtype
            # order, matching the consumer's set_target_for_ref). The data is already
            # packed into these arenas by the copy loop above; we just hand back the
            # whole regions instead of per-name views -> ~2 descriptors, not ~309.
            return [self._serve_arenas[dt][:totals[dt]] for dt in sorted(totals, key=str)]
        return served

    def free_group(self, names):
        with self._cache_cond:
            pks: set[str] = set()
            for name in names:
                self._cache.pop(name, None)
                self._cache_event.pop(name, None)
                pks.add(self._name_to_src[name][0])
            # Drop the gathered physical tensors (stacks) so the layer's memory is
            # released; the per-name views above held refs to them.
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
    layer_groups = _layerwise_groups(names)
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
            warmup_method_name=None,
            # [RDT-COALESCE] carry the flag to the consumer via init_info (env vars
            # don't reach vLLM worker procs); producer reads RDT_COALESCE directly.
            coalesce_dtype_blobs=os.environ.get("RDT_COALESCE", "0") == "1",
            # [RDT-INLINE diagnostic] serialize process off the transfer (test HBM contention)
            inline_process=os.environ.get("RDT_INLINE_PROCESS", "0") == "1",
            # [RDT-ONE-SLOT diagnostic] single receive slot: shrink the RDMA-write
            # working set under the ~2-3GB translation-cache reach (42 vs 34 GB/s)
            one_slot=os.environ.get("RDT_ONE_SLOT", "0") == "1"))))
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
        # [RDT-NO-OVERLAP] Diagnostic: drain the previous group's update_weights
        # (the inter-node NIXL read) BEFORE dispatching this group's intra-node
        # all-gather, so the gather never runs concurrently with the transfer.
        # Tests whether the gather (NVLink + HBM writes) contends with the RDMA read
        # (HBM reads) and depresses transfer BW. Costs the gather's hiding.
        no_overlap = os.environ.get("RDT_NO_OVERLAP", "0") == "1"
        prev_task = None
        prev_names = None
        pending_frees: list = []  # fire-and-forget free_group refs; drained once/iter
        for gidx, group_names in enumerate(layer_groups):
            gi = ShardedRDTWeightTransferUpdateInfo(names=group_names)
            slot = gidx % 2  # accepted by gather_layer for signature compat (ignored)
            if no_overlap and prev_task is not None:
                with cp.timed("update_weights"):
                    await prev_task
                pending_frees += [w.free_group.remote(prev_names) for w in workers]
                prev_task = None
            # Dispatch this group's gather NOW (non-blocking) so it overlaps the
            # previous group's update_weights, which we await next.
            gfuts = [w.gather_layer.remote(group_names, slot) for w in workers]
            if prev_task is not None:
                with cp.timed("update_weights"):
                    await prev_task
                # Fire-and-forget free(prev), drained once at end of iter. Safe even
                # though the gather allocates a FRESH full_tensor per group (no
                # persistent gbuf): the WORKER runs each free_group as its RPC arrives
                # (~one update_weights, ~7s, apart) -- NOT deferred to the driver's
                # drain -- so it reclaims the prev group's memory long before the next
                # group's gather, keeping ~2 groups resident. Moving the ~0.6s Ray
                # round-trip off the critical path (was synchronous here).
                pending_frees += [w.free_group.remote(prev_names) for w in workers]
            # For gidx==0 this is the pipeline fill (nothing overlapped it); for
            # gidx>0 it is only the residual gather tail not hidden by the await.
            with cp.timed("gather_fill" if gidx == 0 else "gather_tail"):
                ray.get(gfuts)
            prev_task = asyncio.create_task(engine.update_weights(
                WeightTransferUpdateRequest(update_info=asdict(gi))))
            prev_names = group_names
        if prev_task is not None:
            with cp.timed("update_weights"):
                await prev_task
            pending_frees += [w.free_group.remote(prev_names) for w in workers]
        # Single drain barrier: ensure this iter's cache is fully evicted before the
        # next iter's produce (else a stale view could satisfy the produce wait). ~0
        # on the critical path since the workers ran each free as its RPC arrived.
        with cp.timed("free_group"):
            ray.get(pending_frees)
        with cp.timed("finish_weight_update"):
            await engine.finish_weight_update()
        cp.finish()

        # Level-2 attribution inputs (per-process durations, gathered after the
        # iter): producer execution timing + consumer NIXL counters from the
        # trainer actors, and the consumer jsonl from the inference node.
        pt = ray.get([w.get_produce_timing.remote() for w in workers])
        nt = ray.get([w.get_nixl_timing.remote() for w in workers])
        gib = sum(p["bytes"] for p in pt) / (1024**3)
        print(cp.report(consumer_jsonl=read_consumer(), producer_timings=pt,
                        producer_nixl=nt, n_groups=len(layer_groups)), flush=True)
        print(f"[profile]   bytes moved (all producers)={gib:.2f}GiB  "
              f"produce calls={sum(p['calls'] for p in pt)}", flush=True)

    # [RPC-PROBE] Measure the bare per-group control-plane directly: an empty
    # update_weights (names=[]) does NO pull/gather/serve, so its driver round-trip
    # is pure driver->AsyncLLMEngine->DP fan-out to 8 workers->await. Compares to a
    # bare 8-way Ray actor fan-out (trainer ping) to separate Ray fan-out from the
    # engine/DP path. Answers whether the ~9ms/group residual is real RPC cost.
    import statistics as _st
    empty = WeightTransferUpdateRequest(update_info=asdict(
        ShardedRDTWeightTransferUpdateInfo(names=[])))
    await engine.start_weight_update(is_checkpoint_format=True)
    for _ in range(3):                      # warmup
        await engine.update_weights(empty)
    ms = []
    for _ in range(30):
        _t = time.perf_counter()
        await engine.update_weights(empty)
        ms.append((time.perf_counter() - _t) * 1000)
    await engine.finish_weight_update()
    ms.sort()
    print(f"[rpc-probe] bare update_weights([]) round-trip ms (n={len(ms)}): "
          f"min={ms[0]:.2f} med={ms[len(ms)//2]:.2f} mean={_st.mean(ms):.2f} "
          f"p90={ms[int(len(ms)*0.9)]:.2f} max={ms[-1]:.2f}", flush=True)
    pms = []
    for _ in range(30):
        _t = time.perf_counter()
        ray.get([w.ping.remote() for w in workers])
        pms.append((time.perf_counter() - _t) * 1000)
    pms.sort()
    print(f"[rpc-probe] bare 8-way trainer ping fan-out ms (n={len(pms)}): "
          f"min={pms[0]:.2f} med={pms[len(pms)//2]:.2f} mean={_st.mean(pms):.2f} "
          f"max={pms[-1]:.2f}", flush=True)

    await engine.resume_generation()
    print("[generate] AFTER sync (real weights):", flush=True)
    for o in await generate_batch(engine, prompts, sampling_params):
        print(f"  {o.prompt!r} -> {o.outputs[0].text!r}", flush=True)
    print("main() returned", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
