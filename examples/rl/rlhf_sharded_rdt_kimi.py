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

The custom gather layout — experts fused per (layer, proj, wkind) into
``[E, *expert]`` stacks so ``full_tensor()`` is ~20 large all-gathers/layer
rather than ~2300 tiny ones, then served as per-expert views — is expressed as
a ``KimiCheckpointSource(WeightSource)``. The sharded-RDT trainer engine
consumes that source exactly like ``ModuleSource`` in the other examples: all
the NIXL serve / gather-cache / packed-ring complexity stays inside the engine.
"""

import glob
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

from vllm.distributed.weight_transfer import (
    HTTPVLLMWeightSyncClient,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.base import ParamMeta, WeightSource
from vllm.distributed.weight_transfer.sharded_rdt_common import layerwise_groups
from vllm.distributed.weight_transfer.sharded_rdt_trainer import (
    ShardedRDTTrainerInitInfo,
)
from vllm.utils.network_utils import get_open_port

MODEL_NAME = "moonshotai/Kimi-K2-Instruct"
RAY_NAMESPACE = "sharded_rdt_kimi_example"
VLLM_PORT = int(os.environ.get("RDT_VLLM_PORT", "8100"))
VLLM_ENDPOINT = f"http://127.0.0.1:{VLLM_PORT}"

FSDP_WORLD_SIZE = 8
INFERENCE_TP_SIZE = 1
INFERENCE_DP_SIZE = 8
NUM_INFERENCE_CONSUMERS = INFERENCE_TP_SIZE * INFERENCE_DP_SIZE
SYNC_ITERS = int(os.environ.get("RDT_SYNC_ITERS", "2"))

_ST_DTYPE = {
    "F8_E4M3": torch.float8_e4m3fn,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F16": torch.float16,
}


class KimiCheckpointSource(WeightSource):
    """WeightSource over Kimi's fused fp8 checkpoint stacks.

    ``metadata()`` reports the INDIVIDUAL checkpoint names (fp8 ``.weight`` +
    fp32 ``.weight_scale_inv``, per-expert), group-major so the engine's
    ``layerwise_groups`` partition matches. Iteration gathers each physical
    param (fused expert stack or individual) once via ``full_tensor()`` — the
    FSDP collective every rank runs in lockstep — and yields each requested
    name as a view (``stack[expert_idx]`` or the whole tensor). A gathered
    stack is dropped after its last view is yielded; the engine's in-flight
    refs (which alias the stack storage) keep it live until the group is served.
    """

    def __init__(self, phys, name_to_src, names, dtype_names, shapes):
        self._phys = phys
        self._name_to_src = name_to_src
        # Group-major order so flatten(layerwise_groups(names)) == names, which
        # the engine asserts.
        ordered = [n for g in layerwise_groups(names) for n in g]
        dt = dict(zip(names, dtype_names))
        sh = dict(zip(names, shapes))
        self._ordered = ordered
        self._meta = [
            ParamMeta(n, getattr(torch, dt[n]), tuple(sh[n])) for n in ordered
        ]
        # Last position each physical key is needed, so iteration can free it.
        self._last = {}
        for i, n in enumerate(ordered):
            self._last[self._name_to_src[n][0]] = i

    def metadata(self):
        return list(self._meta)

    def __iter__(self):
        gathered: dict = {}
        for i, n in enumerate(self._ordered):
            pk, idx = self._name_to_src[n]
            if pk not in gathered:
                gathered[pk] = self._phys[pk].full_tensor()  # collective
            t = gathered[pk] if idx is None else gathered[pk][idx]
            yield n, t
            if self._last[pk] == i:
                gathered.pop(pk, None)


@ray.remote(num_gpus=1)
class KimiTrainWorker:
    """Raw FP8 checkpoint server (one per GPU), sharded with standard
    ``fully_shard``. Holds every checkpoint tensor as a Shard(0) DTensor on GPU;
    the trainer engine gathers per layer and serves slices over NIXL. No HF
    model / no forward. A plain Ray actor — the NIXL serve surface lives in the
    engine's serve actor, not here."""

    def __init__(self, model_name, rank, world_size, master_addr, master_port):
        self.rank = rank
        self.world_size = world_size
        self.engine = None
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.accelerator.set_device_index(0)

        self._load_checkpoint(model_name)

    def _load_checkpoint(self, model_name):
        """Load the FP8 checkpoint as a STANDARD FSDP2 model.

        Build a plain ``nn.Module`` whose parameters are the checkpoint tensors
        (fp8 ``.weight`` + fp32 ``.weight_scale_inv``; routed experts FUSED per
        (layer, proj, wkind) into ``[E, *expert]`` params), then ``fully_shard``
        it and stream each rank's shard from disk. ``full_tensor()`` reconstructs
        each param byte-exact. Sets ``self._phys`` (phys key -> DTensor),
        ``self._name_to_src`` (served name -> (phys key, expert idx | None)), and
        ``self.weight_{names,dtype_names,shapes}`` (per individual served name).
        """
        from collections import OrderedDict

        import regex as re
        import torch.nn as nn
        from safetensors import safe_open
        from torch.distributed.tensor._utils import (
            compute_local_shape_and_global_offset,
        )

        repo = model_name.replace("/", "--")
        snap = glob.glob(f"/root/.cache/huggingface/hub/models--{repo}/snapshots/*")[0]
        with open(os.path.join(snap, "model.safetensors.index.json")) as f:
            wmap = json.load(f)["weight_map"]
        _SKIP = {
            "_expert_map",
            "expert_mask",
            "expert_global_to_physical",
            "expert_physical_to_global",
            "expert_local_to_global",
            "e_score_correction_bias",
        }
        names = [
            n
            for n in wmap
            if "rotary_emb.inv_freq" not in n and n.rsplit(".", 1)[-1] not in _SKIP
        ]

        self._handles: dict = {}

        def H(k):
            fn = wmap[k]
            if fn not in self._handles:
                self._handles[fn] = safe_open(
                    os.path.join(snap, fn), framework="pt", device="cuda:0"
                )
            return self._handles[fn]

        self.weight_names = names
        self.weight_dtype_names = []
        self.weight_shapes = []
        for n in names:
            sl = H(n).get_slice(n)
            self.weight_shapes.append(list(sl.get_shape()))
            self.weight_dtype_names.append(
                str(_ST_DTYPE[sl.get_dtype()]).split(".")[-1]
            )

        expert_re = re.compile(
            r"^(.*\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)$"
        )
        self._name_to_src: dict[str, tuple[str, int | None]] = {}
        stacks: dict[str, dict[int, str]] = {}
        individuals: list[str] = []
        for n in names:
            m = expert_re.match(n)
            if m:
                pk = f"{m.group(1)}.{m.group(3)}.{m.group(4)}"
                self._name_to_src[n] = (pk, int(m.group(2)))
                stacks.setdefault(pk, {})[int(m.group(2))] = n
            else:
                self._name_to_src[n] = (n, None)
                individuals.append(n)

        specs: list[tuple] = []
        for n in individuals:
            sl = H(n).get_slice(n)
            specs.append(
                (n, tuple(sl.get_shape()), _ST_DTYPE[sl.get_dtype()], ("indiv", n))
            )
        for pk, idx_map in stacks.items():
            E = len(idx_map)
            assert set(idx_map) == set(range(E)), f"non-contiguous experts in {pk}"
            sl = H(idx_map[0]).get_slice(idx_map[0])
            specs.append(
                (
                    pk,
                    (E,) + tuple(sl.get_shape()),
                    _ST_DTYPE[sl.get_dtype()],
                    ("stack", idx_map),
                )
            )

        def _lyr(pk):
            return (
                int(pk[len("model.layers.") :].split(".", 1)[0])
                if pk.startswith("model.layers.")
                else -1
            )

        by_layer: OrderedDict[int, list] = OrderedDict()
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
                    torch.empty(shape, dtype=dt, device="meta"), requires_grad=False
                )
            sub.pd = pd
            root.groups.append(sub)
            submods.append((sub, group_specs))
        for sub, _ in submods:
            fully_shard(sub)
        fully_shard(root)

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
                        param.shape, param.device_mesh, param.placements
                    )
                    local.zero_()
                    n0 = lshape[0]
                    if n0 == 0:
                        continue
                    kind, info = loader
                    if kind == "indiv":
                        local[:n0].copy_(
                            H(info).get_slice(info)[goff[0] : goff[0] + n0]
                        )
                    else:
                        for i in range(n0):
                            cn = info[goff[0] + i]
                            local[i].copy_(H(cn).get_tensor(cn))
        torch.accelerator.synchronize()

    def get_rank(self):
        return self.rank

    def setup_engine(self, vllm_endpoint: str):
        source = KimiCheckpointSource(
            self._phys,
            self._name_to_src,
            self.weight_names,
            self.weight_dtype_names,
            self.weight_shapes,
        )
        self.engine = WeightTransferTrainerFactory.trainer_init(
            ShardedRDTTrainerInitInfo(
                rank=self.rank,
                num_consumers=NUM_INFERENCE_CONSUMERS,
                trainer_actor_namespace=RAY_NAMESPACE,
                num_rdt_buffers=int(os.environ.get("NUM_RDT_BUFFERS", "2")),
                layerwise_split=int(os.environ.get("LAYERWISE_SPLIT", "1")),
                arena_presize_gb=float(os.environ.get("RDT_ARENA_PRESIZE_GB", "2.6")),
                nosync=os.environ.get("RDT_NOSYNC", "0") == "1",
                pack_check=os.environ.get("RDT_PACK_CHECK", "0") == "1",
            ),
            client=HTTPVLLMWeightSyncClient(vllm_endpoint),
            source=source,
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
    if not ray.is_initialized():
        ray.init(address="auto", runtime_env=runtime_env, namespace=RAY_NAMESPACE)

    local_model_path = MODEL_NAME
    print("[init] Kimi trainer = raw fp8 sharded checkpoint server", flush=True)

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
    def _free_port():
        return get_open_port()

    fsdp_master_port = ray.get(_free_port.remote())
    print(f"[init] trainer on {fsdp_master_addr}:{fsdp_master_port}", flush=True)

    workers = []
    for rank in range(FSDP_WORLD_SIZE):
        h = KimiTrainWorker.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=trainer_pg, placement_group_bundle_index=rank
            ),
        ).remote(
            local_model_path, rank, FSDP_WORLD_SIZE, fsdp_master_addr, fsdp_master_port
        )
        workers.append(h)
    ray.get([w.get_rank.remote() for w in workers])
    print(
        f"[init] {FSDP_WORLD_SIZE} Kimi trainer workers ready (weights resident).",
        flush=True,
    )

    print("[engine] Launching vllm serve (Kimi fp8, DP8/EP8)...", flush=True)
    server = launch_vllm_serve(
        local_model_path,
        tensor_parallel_size=INFERENCE_TP_SIZE,
        data_parallel_size=INFERENCE_DP_SIZE,
        enable_expert_parallel=True,
        port=VLLM_PORT,
        gpu_memory_utilization=0.90,
        extra_args=[
            "--trust-remote-code",
            "--max-model-len",
            "2048",
            "--max-num-seqs",
            "4",
            "--kv-cache-dtype",
            "fp8",
        ],
    )
    prompts = ["The capital of France is", "The future of AI is"]
    try:
        wait_for_server(VLLM_ENDPOINT, server)

        print("[generate] BEFORE sync (dummy weights):", flush=True)
        for prompt, text in zip(
            prompts, http_generate(VLLM_ENDPOINT, local_model_path, prompts)
        ):
            print(f"  {prompt!r} -> {text!r}", flush=True)

        print("[sync] Building trainer engines (bake on the sender)...", flush=True)
        _t0 = time.perf_counter()
        ray.get([w.setup_engine.remote(VLLM_ENDPOINT) for w in workers])
        print(
            f"[sync] engine setup (incl. bake) took {time.perf_counter() - _t0:.1f}s",
            flush=True,
        )

        pause_generation(VLLM_ENDPOINT)

        for sync_iter in range(SYNC_ITERS):
            ray.get([w.reset_produce_timing.remote() for w in workers])

            print(f"[sync] iter {sync_iter}: gather + serve + update...", flush=True)
            _sync_t0 = time.perf_counter()
            ray.get([w.sync_weights.remote() for w in workers])
            _sync_seconds = time.perf_counter() - _sync_t0

            pt = ray.get([w.get_produce_timing.remote() for w in workers])
            gib = sum(p["bytes"] for p in pt) / (1024**3)
            sender_timing = ray.get(workers[0].get_sync_timing.remote())
            print("=" * 60, flush=True)
            print(
                f"[profile] iter {sync_iter}: wall {_sync_seconds:.3f}s  "
                f"bytes={gib:.2f}GiB  produce_calls={sum(p['calls'] for p in pt)}",
                flush=True,
            )
            print(
                "[profile] sender breakdown (s): "
                + ", ".join(f"{k}={v:.3f}" for k, v in sender_timing.items()),
                flush=True,
            )
            print("=" * 60, flush=True)

        resume_generation(VLLM_ENDPOINT)
        print("[generate] AFTER sync (real weights):", flush=True)
        for prompt, text in zip(
            prompts, http_generate(VLLM_ENDPOINT, local_model_path, prompts)
        ):
            print(f"  {prompt!r} -> {text!r}", flush=True)
        print("main() returned", flush=True)
    finally:
        shutdown_server(server)


if __name__ == "__main__":
    main()
