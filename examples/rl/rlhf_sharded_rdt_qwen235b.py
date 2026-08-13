# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sharded-RDT weight sync for Qwen3-235B-A22B (bf16 MoE) — 16 trainer GPUs
across TWO nodes -> 8 vLLM inference GPUs (TP8) on a third node.

Ceiling probe for the SkyRL Megatron path: the trainer is a raw, FSDP-sharded
bf16 checkpoint server (no HF model, no Megatron, no bridge) using the fused
expert-stack gather layout, so sync time here is the engine+fabric floor this
cluster can reach. Derived from rlhf_sharded_rdt_kimi.py; differences: bf16
(no block scales), 16-rank FSDP spanning 2 nodes (PACK), TP8 consumers that
pull per-worker SHARDS (~1/8 of the model each) rather than DP full copies.

Experts are fused per (layer, proj) into ``[E, *expert]`` stacks so
``full_tensor()`` is ~3 large all-gathers/layer rather than ~390 tiny ones,
then served as per-expert views — the same ``WeightSource`` pattern as the
Kimi example; all NIXL serve / gather-cache / packed-ring complexity stays
inside the engine.
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
from ray.util.scheduling_strategies import (
    NodeAffinitySchedulingStrategy,
    PlacementGroupSchedulingStrategy,
)
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
from vllm.distributed.weight_transfer.base import (
    ParamMeta,
    WeightSource,
    layerwise_groups,
)
from vllm.distributed.weight_transfer.sharded_rdt_trainer import (
    ShardedRDTTrainerInitInfo,
)
from vllm.utils.network_utils import get_open_port

MODEL_NAME = "Qwen/Qwen3-235B-A22B"
RAY_NAMESPACE = "sharded_rdt_qwen235b_example"
VLLM_PORT = int(os.environ.get("RDT_VLLM_PORT", "8100"))
VLLM_ENDPOINT = f"http://127.0.0.1:{VLLM_PORT}"

# 16 trainer ranks (two 8-GPU nodes) is the intended shape. Override to run the
# same probe on a smaller cluster: FSDP_WORLD_SIZE=8 fits ONE trainer node, which
# together with the TP8 inference node needs 16 GPUs instead of 24. Per-rank
# weight memory is the model / FSDP_WORLD_SIZE, so 8 ranks hold ~59GiB each of
# the ~470GiB bf16 checkpoint — it fits on 80GiB cards, with less headroom.
FSDP_WORLD_SIZE = int(os.environ.get("FSDP_WORLD_SIZE", "16"))
INFERENCE_TP_SIZE = 8
INFERENCE_DP_SIZE = 1
NUM_INFERENCE_CONSUMERS = INFERENCE_TP_SIZE * INFERENCE_DP_SIZE
SYNC_ITERS = int(os.environ.get("RDT_SYNC_ITERS", "3"))

# Pipeline-parallel producer layout: the ranks are split into NUM_PP_STAGES
# stages, and a stage holds only its slice of the layers, sharded across the
# stage's ranks. Nothing is gathered across stages — each layer's gather runs
# within the one stage that owns it, and consumers route each pull to an owner
# (see RdtRouter). NUM_PP_STAGES=1 is the historical gather-to-all layout: one
# stage of every rank, each holding every layer.
NUM_PP_STAGES = int(os.environ.get("NUM_PP_STAGES", "2"))
assert FSDP_WORLD_SIZE % NUM_PP_STAGES == 0, (
    f"NUM_PP_STAGES={NUM_PP_STAGES} must divide FSDP_WORLD_SIZE={FSDP_WORLD_SIZE}"
)
STAGE_SIZE = FSDP_WORLD_SIZE // NUM_PP_STAGES

_ST_DTYPE = {
    "F8_E4M3": torch.float8_e4m3fn,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F16": torch.float16,
}


def _group_stages(groups: list[list[str]], num_stages: int) -> list[int]:
    """Stage that owns each gather group.

    ``groups`` is ``layerwise_groups(names)``: the pre block, one group per
    decoder layer, then the post block. Decoder layers split contiguously across
    stages; the pre block goes to the first stage and the post block to the last,
    mirroring where a pipeline-parallel trainer would hold them.
    """
    layer_of = {}
    for gi, group in enumerate(groups):
        first = group[0]
        if first.startswith("model.layers."):
            layer_of[gi] = int(first[len("model.layers.") :].split(".", 1)[0])
    num_layers = max(layer_of.values()) + 1 if layer_of else 1
    stages = []
    for gi in range(len(groups)):
        if gi in layer_of:
            stages.append(min(layer_of[gi] * num_stages // num_layers, num_stages - 1))
        else:
            stages.append(0 if gi == 0 else num_stages - 1)
    return stages


class QwenCheckpointSource(WeightSource):
    """WeightSource over Qwen3-235B's fused bf16 checkpoint stacks.

    ``metadata()`` reports the INDIVIDUAL checkpoint names (fp8 ``.weight`` +
    fp32 ``.weight_scale_inv``, per-expert) for the WHOLE model, group-major so
    the engine's ``layerwise_groups`` partition matches — every rank describes
    the same model even when it holds only part of it, which keeps the group
    partition and the consumers' plan identical fleet-wide.

    Iteration covers only ``owned_group_idx``, the groups this rank's stage
    holds: it gathers each physical param (fused expert stack or individual)
    once via ``full_tensor()`` — the FSDP collective the stage's ranks run in
    lockstep — and yields each requested name as a view (``stack[expert_idx]``
    or the whole tensor). A gathered stack is dropped after its last view is
    yielded; the engine's in-flight refs (which alias the stack storage) keep it
    live until the group is served.
    """

    def __init__(self, phys, name_to_src, names, dtype_names, shapes, owned_group_idx):
        self._phys = phys
        self._name_to_src = name_to_src
        # Group-major order so flatten(layerwise_groups(names)) == names, which
        # the engine asserts.
        groups = layerwise_groups(names)
        ordered = [n for g in groups for n in g]
        dt = dict(zip(names, dtype_names))
        sh = dict(zip(names, shapes))
        self._meta = [
            ParamMeta(n, getattr(torch, dt[n]), tuple(sh[n])) for n in ordered
        ]
        self._owned_group_idx = sorted(owned_group_idx)
        self._owned = [n for gi in self._owned_group_idx for n in groups[gi]]
        # Claiming a group whose params this rank never built would fail as a
        # KeyError mid-gather; catch it while the mistake is still local.
        missing = sorted({self._name_to_src[n][0] for n in self._owned} - set(phys))
        if missing:
            raise ValueError(
                f"owned groups {self._owned_group_idx[:4]}... need {len(missing)} "
                f"params this rank does not hold, e.g. {missing[:3]}"
            )
        # Last position each physical key is needed, over the OWNED walk, so
        # iteration can free a gathered stack as soon as its views are out.
        self._last = {}
        for i, n in enumerate(self._owned):
            self._last[self._name_to_src[n][0]] = i

    def metadata(self):
        return list(self._meta)

    def owned_groups(self):
        return list(self._owned_group_idx)

    def __iter__(self):
        gathered: dict = {}
        for i, n in enumerate(self._owned):
            pk, idx = self._name_to_src[n]
            if pk not in gathered:
                gathered[pk] = self._phys[pk].full_tensor()  # collective
            t = gathered[pk] if idx is None else gathered[pk][idx]
            yield n, t
            if self._last[pk] == i:
                gathered.pop(pk, None)


@ray.remote(num_gpus=1)
class QwenTrainWorker:
    """Raw bf16 checkpoint server (one per GPU), sharded with standard
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
        """Load the bf16 checkpoint as a STANDARD FSDP2 model.

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
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        snap = glob.glob(f"{hf_home}/hub/models--{repo}/snapshots/*")[0]
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

        # Which gather group each spec belongs to, and which groups this rank's
        # stage owns. Only owned specs become parameters — a rank never holds,
        # and never gathers, another stage's layers.
        groups = layerwise_groups(names)
        group_of_name = {n: gi for gi, g in enumerate(groups) for n in g}
        stages = _group_stages(groups, NUM_PP_STAGES)
        my_stage = self.rank // STAGE_SIZE
        self.owned_group_idx = [
            gi for gi in range(len(groups)) if stages[gi] == my_stage
        ]
        owned = set(self.owned_group_idx)

        def _group_of_spec(pk, loader):
            # A stack's per-expert names all sit in one group; individuals are
            # their own name.
            probe = pk if loader[0] == "indiv" else loader[1][0]
            return group_of_name[probe]

        by_group: OrderedDict[int, list] = OrderedDict()
        for s in specs:
            gi = _group_of_spec(s[0], s[3])
            if gi in owned:
                by_group.setdefault(gi, []).append(s)

        # Shard within the stage only: mesh row ("shard") spans this rank's
        # stage, so full_tensor() is a stage-local all-gather and no collective
        # ever crosses stages. With NUM_PP_STAGES=1 the row is the whole world,
        # i.e. the historical layout.
        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh(
            "cuda", (NUM_PP_STAGES, STAGE_SIZE), mesh_dim_names=("stage", "shard")
        )
        stage_mesh = mesh["shard"]

        root = nn.Module()
        root.groups = nn.ModuleList()
        submods: list[tuple] = []
        for _gi, group_specs in by_group.items():
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
            fully_shard(sub, mesh=stage_mesh)
        fully_shard(root, mesh=stage_mesh)

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
        source = QwenCheckpointSource(
            self._phys,
            self._name_to_src,
            self.weight_names,
            self.weight_dtype_names,
            self.weight_shapes,
            self.owned_group_idx,
        )
        self.engine = WeightTransferTrainerFactory.trainer_init(
            ShardedRDTTrainerInitInfo(
                rank=self.rank,
                num_consumers=NUM_INFERENCE_CONSUMERS,
                trainer_actor_namespace=RAY_NAMESPACE,
                num_rdt_buffers=int(os.environ.get("NUM_RDT_BUFFERS", "2")),
                arena_presize_gb=float(os.environ.get("RDT_ARENA_PRESIZE_GB", "2.6")),
                pack_check=os.environ.get("RDT_PACK_CHECK", "0") == "1",
            ),
            client=HTTPVLLMWeightSyncClient(vllm_endpoint),
            source=source,
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
    # Ray workers are spawned by the raylet and inherit ITS environment, not the
    # driver's, so the checkpoint location has to be forwarded explicitly: each
    # trainer rank resolves the snapshot itself under HF_HOME (see
    # _load_checkpoint) and would otherwise look in ~/.cache/huggingface.
    forwarded = {
        k: os.environ[k]
        for k in (
            "HF_HOME",
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "NCCL_CUMEM_ENABLE",
            "LD_LIBRARY_PATH",
        )
        if k in os.environ
    }
    if forwarded:
        runtime_env["env_vars"] = forwarded
    if not ray.is_initialized():
        ray.init(address="auto", runtime_env=runtime_env, namespace=RAY_NAMESPACE)

    # The trainer ranks dial this endpoint from other nodes, so it must be this
    # node's address rather than loopback.
    global VLLM_ENDPOINT
    VLLM_ENDPOINT = f"http://{ray.util.get_node_ip_address()}:{VLLM_PORT}"

    local_model_path = MODEL_NAME
    print(
        f"[init] Qwen3-235B trainer = raw bf16 sharded checkpoint server "
        f"({FSDP_WORLD_SIZE} ranks / 2 nodes, {NUM_PP_STAGES} pipeline stage(s) "
        f"x {STAGE_SIZE} ranks)",
        flush=True,
    )

    # Trainer placement. ``vllm serve`` runs with the ``mp`` executor (see below),
    # so its 8 workers are plain local processes on THIS driver's node — the
    # trainer ranks must therefore land on other nodes, or the two fleets fight
    # over the same GPUs.
    #
    # Default: 16 single-GPU bundles PACK across exactly two 8-GPU nodes
    # (STRICT_PACK cannot span nodes), leaving the driver's node free. That works
    # because 16 bundles cannot fit on one node. With FSDP_WORLD_SIZE=8 they can,
    # and PACK would happily choose the driver's own node — so a 2-node run must
    # say where the trainers go. RDT_TRAINER_NODE_IP pins them by node affinity
    # instead (also what rlhf_sharded_rdt_mn.py does).
    trainer_ip = os.environ.get("RDT_TRAINER_NODE_IP")
    if trainer_ip:
        trainer_node_id = next(
            n["NodeID"]
            for n in ray.nodes()
            if n["Alive"] and n["NodeManagerAddress"] == trainer_ip
        )
        trainer_sched: object = NodeAffinitySchedulingStrategy(
            node_id=trainer_node_id, soft=False
        )
        fsdp_master_addr = trainer_ip
        rank_sched = [trainer_sched] * FSDP_WORLD_SIZE
    else:
        trainer_pg = placement_group(
            [{"GPU": 1, "CPU": 1}] * FSDP_WORLD_SIZE, strategy="PACK"
        )
        ray.get(trainer_pg.ready())
        pg_node_id = next(
            iter(placement_group_table(trainer_pg)["bundles_to_node_id"].values())
        )
        fsdp_master_addr = next(
            n["NodeManagerAddress"] for n in ray.nodes() if n["NodeID"] == pg_node_id
        )
        trainer_sched = PlacementGroupSchedulingStrategy(placement_group=trainer_pg)
        rank_sched = [
            PlacementGroupSchedulingStrategy(
                placement_group=trainer_pg, placement_group_bundle_index=rank
            )
            for rank in range(FSDP_WORLD_SIZE)
        ]

    @ray.remote(num_cpus=0, scheduling_strategy=trainer_sched)
    def _free_port():
        return get_open_port()

    fsdp_master_port = ray.get(_free_port.remote())
    print(f"[init] trainer on {fsdp_master_addr}:{fsdp_master_port}", flush=True)

    workers = []
    for rank in range(FSDP_WORLD_SIZE):
        h = QwenTrainWorker.options(
            num_gpus=1,
            scheduling_strategy=rank_sched[rank],
        ).remote(
            local_model_path, rank, FSDP_WORLD_SIZE, fsdp_master_addr, fsdp_master_port
        )
        workers.append(h)
    ray.get([w.get_rank.remote() for w in workers])
    print(
        f"[init] {FSDP_WORLD_SIZE} Qwen trainer workers ready (weights resident).",
        flush=True,
    )

    print("[engine] Launching vllm serve (Qwen3-235B bf16, TP8)...", flush=True)
    server = launch_vllm_serve(
        local_model_path,
        tensor_parallel_size=INFERENCE_TP_SIZE,
        data_parallel_size=INFERENCE_DP_SIZE,
        enable_expert_parallel=False,
        port=VLLM_PORT,
        gpu_memory_utilization=0.85,
        extra_args=[
            "--trust-remote-code",
            "--max-model-len",
            "2048",
            "--max-num-seqs",
            "4",
            # vLLM must not take a Ray placement group here: the trainer PG holds
            # 16 of the cluster's GPUs and vLLM's own PG could not be scheduled.
            "--distributed-executor-backend",
            "mp",
        ],
    )
    prompts = ["The capital of France is", "The future of AI is"]
    try:
        wait_for_server(VLLM_ENDPOINT, server, timeout=3600)

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
            print(f"[sync] iter {sync_iter}: gather + serve + update...", flush=True)
            _sync_t0 = time.perf_counter()
            ray.get([w.sync_weights.remote() for w in workers])
            print(
                f"[sync] iter {sync_iter} took "
                f"{time.perf_counter() - _sync_t0:.3f}s",
                flush=True,
            )

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
