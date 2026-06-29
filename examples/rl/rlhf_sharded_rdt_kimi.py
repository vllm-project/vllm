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


def _cstride(shape):
    """Contiguous strides for a shape (for DTensor.from_local explicit global stride)."""
    s = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        s[i] = s[i + 1] * shape[i + 1]
    return tuple(s)


@ray.remote(num_gpus=1, max_concurrency=8, enable_tensor_transport=True)
class KimiTrainWorker:
    """Raw FP8 sharded checkpoint server (one per GPU). Holds every checkpoint
    tensor sharded Shard(0) on GPU; gathers per layer and serves slices over NIXL.
    No HF model / no forward; gather + serve only."""

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

        # gather cache (bounded: freed per layer group) + sync
        self._cache: dict[str, torch.Tensor] = {}       # individual name -> view
        self._cache_phys: dict[str, torch.Tensor] = {}  # physical (stack/indiv) -> full
        self._cache_cond = threading.Condition()
        self._gather_error: BaseException | None = None

        # Producer-side serve arenas, ONE PER DTYPE: clone each produced slice into
        # a persistent, NIXL-pre-registered arena (registered once) and serve VIEWS,
        # instead of cloning into fresh buffers (which re-registers ~148k buffers per
        # iter). Reused across produce calls -- safe because the driver serializes
        # update_weights (transfer K drains before produce K+1 overwrites). Holds
        # only one group's SERVED slices (~a couple GB), so memory is bounded.
        from ray.experimental import register_nixl_memory
        self._register_nixl_memory = register_nixl_memory
        self._serve_arenas: dict[torch.dtype, torch.Tensor] = {}

        # profiling counters
        self._timing_lock = threading.Lock()
        self._produce_calls = self._produce_specs = self._produce_bytes = 0
        self._produce_wait_seconds = self._produce_slice_seconds = 0.0
        self._produce_method_seconds = 0.0
        from vllm.distributed.weight_transfer._nixl_profile import install_nixl_timing
        install_nixl_timing()

    def _load_checkpoint(self, model_name):
        """Load the FP8 checkpoint to GPU sharded Shard(0), STACKING routed experts.

        Routed experts are stacked per (layer, proj, weight/scale) into fused
        tensors [E, *expert] (like the Qwen HF model's fused experts), so the gather
        is a handful of large all-gathers per layer instead of ~2300 tiny ones.
        Non-expert params stay individual. We still serve INDIVIDUAL checkpoint names
        (views/slices of the stacks) -- the contract vLLM's DeepseekV3 load_weights
        expects. Sharding is manual uniform Shard(0) (verified byte-exact), not
        fully_shard (which flat-shards a multi-param holder non-uniformly).
        """
        import re
        from safetensors import safe_open
        from torch.distributed.tensor import DTensor, Shard, init_device_mesh
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

        # Parse names: routed experts -> per-(layer,proj,wkind) stack; else individual.
        expert_re = re.compile(
            r"^(.*\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)$"
        )
        self._name_to_src: dict[str, tuple[str, int | None]] = {}
        stacks: dict[str, dict[int, str]] = {}  # phys_key -> {expert_idx: ckpt_name}
        individuals: list[str] = []
        for n in names:
            m = expert_re.match(n)
            if m:
                pk = f"{m.group(1)}.{m.group(3)}.{m.group(4)}"  # synthetic stack key
                self._name_to_src[n] = (pk, int(m.group(2)))
                stacks.setdefault(pk, {})[int(m.group(2))] = n
            else:
                self._name_to_src[n] = (n, None)
                individuals.append(n)

        mesh = init_device_mesh("cuda", (self.world_size,))
        self._mesh = mesh
        self._phys: dict[str, torch.Tensor] = {}

        def make_shard0(full_shape, dtype, load_rows):
            """Uniform Shard(0) DTensor; load_rows(start,end) returns rows [start:end)
            on cuda. Last rank's shard is zero-padded."""
            D = full_shape[0]
            rest = tuple(full_shape[1:])
            world = self.world_size
            sp = (D + world - 1) // world
            start, end = self.rank * sp, min((self.rank + 1) * sp, D)
            local = torch.zeros((sp,) + rest, dtype=dtype, device="cuda:0")
            if end > start:
                local[: end - start].copy_(load_rows(start, end))
            return DTensor.from_local(local, mesh, [Shard(0)], run_check=False,
                                      shape=torch.Size(full_shape),
                                      stride=_cstride(full_shape))

        for n in individuals:
            sl = H(n).get_slice(n)
            shape, dt = tuple(sl.get_shape()), _ST_DTYPE[sl.get_dtype()]
            self._phys[n] = make_shard0(
                shape, dt, lambda a, b, _n=n: H(_n).get_slice(_n)[a:b])

        for pk, idx_map in stacks.items():
            E = len(idx_map)
            assert set(idx_map) == set(range(E)), f"non-contiguous experts in {pk}"
            sl = H(idx_map[0]).get_slice(idx_map[0])
            eshape, dt = tuple(sl.get_shape()), _ST_DTYPE[sl.get_dtype()]

            def load_rows(a, b, _im=idx_map, _es=eshape, _dt=dt):
                out = torch.empty((b - a,) + _es, dtype=_dt, device="cuda:0")
                for i, e in enumerate(range(a, b)):
                    cn = _im[e]
                    out[i].copy_(H(cn).get_tensor(cn))
                return out

            self._phys[pk] = make_shard0((E,) + eshape, dt, load_rows)
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
    def gather_layer(self, names):
        """Gather a layer group: full_tensor() each unique PHYSICAL tensor (stacks +
        individuals -- ~20/layer instead of ~2300 individual experts), then expose
        each requested individual name as a view (stack_full[expert_idx], or the
        individual full tensor). Bounded: the driver frees each group after its
        update_weights drains.
        """
        try:
            phys_keys: list[str] = []
            seen: set[str] = set()
            for n in names:
                pk = self._name_to_src[n][0]
                if pk not in seen:
                    seen.add(pk)
                    phys_keys.append(pk)
            full_phys = {pk: self._phys[pk].full_tensor() for pk in phys_keys}
            with self._cache_cond:
                self._cache_phys.update(full_phys)
                for n in names:
                    pk, idx = self._name_to_src[n]
                    self._cache[n] = full_phys[pk] if idx is None else full_phys[pk][idx]
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
        # Replay op-chains -> the produced slice views (into the gather cache).
        sliced: list[torch.Tensor] = []
        for name, chain in specs:
            t = self._cache[name]
            for op, args, kw in chain:
                if op not in _ALLOWED_OPS:
                    raise ValueError(f"{name!r}: disallowed op {op!r}")
                t = getattr(t, op)(*args, **dict(kw))
            sliced.append(t)
        # Lay each slice into the persistent per-dtype serve arena and clone in;
        # serve the arena VIEW (registration reused -> no per-pull register).
        layout: list[tuple[torch.dtype, int]] = []
        totals: dict[torch.dtype, int] = {}
        for t in sliced:
            dt = t.dtype
            off = totals.get(dt, 0)
            layout.append((dt, off))
            n = t.numel()
            totals[dt] = off + ((n + 7) & ~7)
        for dt, total in totals.items():
            a = self._serve_arenas.get(dt)
            if a is None or a.numel() < total:
                a = torch.empty(total, dtype=dt, device="cuda:0")
                self._register_nixl_memory(a)  # one-time; pinned for process life
                self._serve_arenas[dt] = a
        out: list[torch.Tensor] = []
        nbytes = 0
        for t, (dt, off) in zip(sliced, layout):
            n = t.numel()
            view = self._serve_arenas[dt][off:off + n].reshape(t.shape)
            view.copy_(t)
            out.append(view)
            nbytes += view.element_size() * n
        torch.accelerator.synchronize()
        slice_s = time.perf_counter() - _t_s0
        with self._timing_lock:
            self._produce_calls += 1
            self._produce_specs += len(specs)
            self._produce_wait_seconds += wait_s
            self._produce_slice_seconds += slice_s
            self._produce_bytes += nbytes
            self._produce_method_seconds += time.perf_counter() - _t_m0
        return out

    def free_group(self, names):
        with self._cache_cond:
            pks: set[str] = set()
            for name in names:
                self._cache.pop(name, None)
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
    os.makedirs(os.path.dirname(consumer_file), exist_ok=True)
    open(consumer_file, "w").close()

    print("[sync] init_weight_transfer_engine (bake)...", flush=True)
    _t0 = time.perf_counter()
    await engine.init_weight_transfer_engine(WeightTransferInitRequest(
        init_info=asdict(ShardedRDTWeightTransferInitInfo(
            trainer_actor_names=producer_names,
            trainer_actor_namespace=RAY_NAMESPACE,
            names=names, dtype_names=dtype_names, shapes=shapes,
            warmup_method_name=None))))
    print(f"[sync] bake took {time.perf_counter()-_t0:.1f}s", flush=True)

    await engine.pause_generation(mode="abort")

    for sync_iter in range(SYNC_ITERS):
        ray.get([w.reset_produce_timing.remote() for w in workers])
        ray.get([w.reset_nixl_timing.remote() for w in workers])
        await engine.start_weight_update(is_checkpoint_format=True)
        print(f"[sync] iter {sync_iter}: gather + update_weights...", flush=True)
        _s0 = time.perf_counter()
        allgather_s = 0.0
        prev_task = None
        prev_names = None
        for group_names in layer_groups:
            gi = ShardedRDTWeightTransferUpdateInfo(names=group_names)
            gfuts = [w.gather_layer.remote(group_names) for w in workers]
            if prev_task is not None:
                await prev_task
                ray.get([w.free_group.remote(prev_names) for w in workers])
            _tg = time.perf_counter()
            ray.get(gfuts)
            allgather_s += time.perf_counter() - _tg
            prev_task = asyncio.create_task(engine.update_weights(
                WeightTransferUpdateRequest(update_info=asdict(gi))))
            prev_names = group_names
        if prev_task is not None:
            await prev_task
            ray.get([w.free_group.remote(prev_names) for w in workers])
        await engine.finish_weight_update()
        secs = time.perf_counter() - _s0
        pt = ray.get([w.get_produce_timing.remote() for w in workers])
        nt = ray.get([w.get_nixl_timing.remote() for w in workers])
        gib = sum(p["bytes"] for p in pt) / (1024**3)
        print(f"[profile] iter {sync_iter}: total={secs:.2f}s  bytes={gib:.2f}GiB  "
              f"calls={sum(p['calls'] for p in pt)}", flush=True)
        print(f"[profile]   allgather(overlap'd tail)={allgather_s:.2f}s  "
              f"producer slice+clone(max)={max(p['slice_seconds'] for p in pt):.2f}s  "
              f"producer method(max)={max(p['method_seconds'] for p in pt):.2f}s", flush=True)
        print(f"[profile]   producer NIXL register(max)={max(n['register_seconds'] for n in nt):.2f}s "
              f"({sum(n['register_calls'] for n in nt)} regs)  "
              f"extract(max)={max(n['extract_seconds'] for n in nt):.2f}s", flush=True)

    # consumer-side breakdown (warm iters), read from the engine's jsonl
    try:
        from collections import defaultdict
        recs = defaultdict(list)
        base = []
        with open(consumer_file) as f:
            for line in f:
                r = json.loads(line)
                (base if r.get("mode") == "baseline" else recs[r["pid"]]).append(r)
        n_groups = len(layer_groups)
        agg = dict.fromkeys(("pull", "transfer_seconds", "register_seconds",
                             "deregister_seconds", "descs_seconds", "process"), 0.0)
        nreg = cnt = 0
        for pid, rows in recs.items():
            warm = rows[n_groups:] if len(rows) > n_groups else rows
            for r in warm:
                for k in agg:
                    agg[k] += r.get(k, 0.0)
                nreg += r.get("register_calls", 0); cnt += 1
        if cnt:
            print(f"[profile] CONSUMER warm (sum over {cnt} pulls, all workers): "
                  f"pull={agg['pull']:.1f}s transfer={agg['transfer_seconds']:.1f}s "
                  f"register={agg['register_seconds']:.1f}s ({nreg} regs) "
                  f"descs={agg['descs_seconds']:.1f}s process={agg['process']:.1f}s", flush=True)
    except Exception as e:
        print(f"[profile] consumer read failed: {e!r}", flush=True)

    await engine.resume_generation()
    print("[generate] AFTER sync (real weights):", flush=True)
    for o in await generate_batch(engine, prompts, sampling_params):
        print(f"  {o.prompt!r} -> {o.outputs[0].text!r}", flush=True)
    print("main() returned", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
