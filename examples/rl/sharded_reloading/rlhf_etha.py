# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
RLHF with Etha M-to-N weight transfer.

Topology (8 GPUs, single node):
  Training  — 4 Ray-actor workers, each holds a sharded DTensor view of
              Qwen3-30B-A3B-Instruct-2507 (att_mesh = (2, 2),
              moe_mesh = (1, 2, 2)). Weights are loaded once via
              `dcp.load(HuggingFaceStorageReader)` straight onto the
              `meta`-built DTensors — no full-precision CPU staging.
  Inference — 4 vLLM workers, DP=2 TP=2 with expert parallelism
              (EP_SIZE = TP * DP = 4).

All 8 ranks join one `StatelessProcessGroup`; the Etha engine plans an
M-to-N redistribution per pair (qkv_proj, o_proj, embed_tokens, lm_head,
router, layernorm, experts_gate_up, experts_down) at init time and
reuses the plan for every sync round.

Steps:
  1. Launch 4 Ray-actor training workers.
  2. Launch `AsyncLLMEngine` with `weight_transfer_config=WeightTransferConfig(
     backend="etha")` and `load_format="dummy"`.
  3. Generate from prompts with dummy weights → gibberish.
  4. Pause generation, init weight-transfer rendezvous, ship weights, resume.
  5. Generate again → coherent Qwen3 output.

Mirrors `rlhf_nccl_fsdp_ep.py` in shape but uses Etha's M-to-N planning
instead of rank-0 gather + broadcast.
"""

import asyncio
import os
import re
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path

# Etha lives next to this script. Put its directory on sys.path so the
# local `etha_*` modules import cleanly here AND in any Ray actor that
# inherits this runtime_env (we forward PYTHONPATH below).
_EXAMPLE_DIR = str(Path(__file__).resolve().parent)
if _EXAMPLE_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLE_DIR)

import ray
import torch
import torch.distributed.checkpoint as dcp
from huggingface_hub import snapshot_download
from torch.distributed.checkpoint import DefaultLoadPlanner, HuggingFaceStorageReader
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.placement_types import Placement
from transformers import AutoConfig, AutoModelForCausalLM

import vllm
from vllm import SamplingParams
from vllm.config import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
)
from vllm.utils.network_utils import get_ip, get_open_port
from vllm.v1.executor import Executor

# Local etha modules. Importing `etha_engine` here also registers "etha"
# as a weight-transfer backend in this process (see the bottom of that
# file). Workers register the backend on their own when they import
# `etha_engine.EthaWorkerExtension` via the `worker_extension_cls`
# we pass to AsyncEngineArgs below.
from etha_engine import (  # noqa: E402
    EthaTrainerWeightTransferEngine,
    EthaWeightTransferInitInfo,
    EthaWeightTransferUpdateInfo,
)
from etha_sharding import (  # noqa: E402
    MOE_HANDLERS,
    TRAINER_HANDLER_PLACEMENTS,
    get_handler_name,
)

MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"

# Trainer topology — match TRAINER_HANDLER_PLACEMENTS' assumptions.
TRAINER_WORLD_SIZE = 4
TRAINER_ATTN_DP_REPLICATE = 2
TRAINER_ATTN_DP_SHARD = 2
TRAINER_MOE_DP_REPLICATE = 1
TRAINER_MOE_DP_SHARD = 2
TRAINER_EP_SIZE = 2

# Inference topology.
VLLM_TP_SIZE = 2
VLLM_DP_SIZE = 2
VLLM_WORLD_SIZE = VLLM_TP_SIZE * VLLM_DP_SIZE  # 4

# Joint group size.
TRANSFER_WORLD_SIZE = TRAINER_WORLD_SIZE + VLLM_WORLD_SIZE


# ============================================================================
# Trainer-side state-dict building (ported from Etha example trainer)
# ============================================================================

_GROUPED_RE = re.compile(r"^(.+\.mlp\.experts)\.(gate_up_proj|down_proj)$")


def _explode_grouped_moe(state_dict: dict) -> dict:
    """Grouped MoE entries → per-expert HF-keyed views into the grouped buffer.

    DCP's HuggingFaceStorageReader expects per-expert keys
    (`...experts.{i}.gate_proj.weight`); we present views into the
    grouped tensor under those names so the load writes through into
    the grouped storage.
    """
    out = {}
    for key, value in state_dict.items():
        m = _GROUPED_RE.match(key)
        if m is None:
            out[key] = value
            continue
        prefix, role = m.group(1), m.group(2)
        local = value._local_tensor if isinstance(value, DTensor) else value
        e_local = local.shape[0]
        chunk_idx = 0
        if isinstance(value, DTensor):
            mesh = value.device_mesh
            for mesh_dim, pl in enumerate(value.placements):
                if isinstance(pl, Shard) and pl.dim == 0:
                    rank_in_dim = mesh.get_local_rank(mesh.mesh_dim_names[mesh_dim])
                    chunk_idx = chunk_idx * mesh.size(mesh_dim) + rank_in_dim
        e_start = chunk_idx * e_local
        if role == "gate_up_proj":
            half = local.shape[1] // 2
            for i in range(e_local):
                out[f"{prefix}.{e_start + i}.gate_proj.weight"] = local[i, :half, :]
                out[f"{prefix}.{e_start + i}.up_proj.weight"] = local[i, half:, :]
        else:  # down_proj
            for i in range(e_local):
                out[f"{prefix}.{e_start + i}.down_proj.weight"] = local[i]
    return out


class GroupedMoEPlanner(DefaultLoadPlanner):
    def set_up_planner(self, state_dict, *args, **kwargs) -> None:
        super().set_up_planner(_explode_grouped_moe(state_dict), *args, **kwargs)


def _local_shape(
    global_shape: tuple[int, ...],
    mesh_shape: tuple[int, ...],
    placements: tuple[Placement, ...],
) -> tuple[int, ...]:
    shape = list(global_shape)
    for dim_idx, p in enumerate(placements):
        if isinstance(p, Shard):
            shape[p.dim] //= mesh_shape[dim_idx]
    return tuple(shape)


# ============================================================================
# Ray trainer actor
# ============================================================================


@ray.remote(num_gpus=1)
class EthaTrainerActor:
    """One training worker per GPU; 4 actors form the trainer side."""

    def __init__(
        self,
        rank: int,
        world_size: int,
        master_addr: str,
        master_port: int,
    ):
        self.rank = rank
        self.world_size = world_size

        # Trainer-internal NCCL PG (needed by DTensor / dcp.load).
        # This is separate from the cross-cluster StatelessProcessGroup
        # that ships weights to vLLM — that one is set up in
        # init_transfer().
        import torch.distributed as dist
        from torch.distributed import DeviceMesh

        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        torch.cuda.set_device(0)  # Ray gave us one GPU
        # Multi-backend PG: NCCL for tensor collectives, gloo for the
        # object collectives that dcp.load uses internally. Pure-NCCL
        # crashes inside gather_object during planning.
        dist.init_process_group(
            backend="cuda:nccl,cpu:gloo", rank=rank, world_size=world_size
        )
        self.device = torch.device("cuda:0")

        self.att_mesh = DeviceMesh(
            "cuda",
            torch.arange(TRAINER_ATTN_DP_REPLICATE * TRAINER_ATTN_DP_SHARD).view(
                TRAINER_ATTN_DP_REPLICATE, TRAINER_ATTN_DP_SHARD
            ),
            mesh_dim_names=("dp_replicate", "dp_shard"),
        )
        self.moe_mesh = DeviceMesh(
            "cuda",
            torch.arange(
                TRAINER_MOE_DP_REPLICATE * TRAINER_MOE_DP_SHARD * TRAINER_EP_SIZE
            ).view(TRAINER_MOE_DP_REPLICATE, TRAINER_MOE_DP_SHARD, TRAINER_EP_SIZE),
            mesh_dim_names=("dp_replicate", "dp_shard", "ep"),
        )

        self.state: dict[str, torch.Tensor] | None = None
        self.engine: EthaTrainerWeightTransferEngine | None = None

    def get_ip_port(self) -> tuple[str, int]:
        """Called only on rank 0."""
        return get_ip(), get_open_port()

    def load_weights(self) -> int:
        """Build a meta model, wrap each param as a DTensor under the
        right mesh, then dcp.load straight into the local shards."""
        hf_config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if getattr(hf_config, "model_type", "") != "qwen3_moe":
            mt = hf_config.model_type
            raise RuntimeError(f"Hard-coded for Qwen3-MoE, got model_type={mt!r}")
        model_dir = Path(snapshot_download(MODEL_NAME))

        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(
                hf_config, torch_dtype=torch.float32, trust_remote_code=True
            )

        state: dict[str, torch.Tensor] = {}
        for name, meta_param in model.named_parameters():
            handler = get_handler_name(name)
            if handler is None:
                continue
            placements = TRAINER_HANDLER_PLACEMENTS[handler]
            mesh = self.moe_mesh if handler in MOE_HANDLERS else self.att_mesh
            local_shape = _local_shape(
                tuple(meta_param.shape), tuple(mesh.mesh.shape), placements
            )
            local = torch.empty(local_shape, dtype=meta_param.dtype, device=self.device)
            state[name] = DTensor.from_local(local, mesh, placements)

        dcp.load(
            state,
            storage_reader=HuggingFaceStorageReader(str(model_dir)),
            planner=GroupedMoEPlanner(),
        )
        # Keep DTensors as the original; the trainer engine works
        # against `.to_local()` views.
        self.state = state
        return len(state)

    def init_transfer(self, master_address: str, master_port: int) -> None:
        # Convert DTensors → local shards. We deliberately keep the
        # grouped `experts.gate_up_proj` name here; the trainer-side
        # sharding strategy mirrors the engine-side split into
        # `gate_proj`/`up_proj` views and groups them under the same handler.
        assert self.state is not None
        local_state: dict[str, torch.Tensor] = {
            name: (dt.to_local() if isinstance(dt, DTensor) else dt)
            for name, dt in self.state.items()
        }

        init_info = EthaWeightTransferInitInfo(
            master_address=master_address,
            master_port=master_port,
            rank_offset=0,
            world_size=TRANSFER_WORLD_SIZE,
            trainer_attn_dp_replicate=TRAINER_ATTN_DP_REPLICATE,
            trainer_attn_dp_shard=TRAINER_ATTN_DP_SHARD,
            trainer_moe_dp_replicate=TRAINER_MOE_DP_REPLICATE,
            trainer_moe_dp_shard=TRAINER_MOE_DP_SHARD,
            trainer_ep_size=TRAINER_EP_SIZE,
            vllm_dp_size=VLLM_DP_SIZE,
            vllm_tp_size=VLLM_TP_SIZE,
            vllm_ep_size=VLLM_TP_SIZE * VLLM_DP_SIZE,
        )
        self.engine = EthaTrainerWeightTransferEngine.trainer_init(
            init_info,
            rank=self.rank,
            device_index=0,
            state_dict=local_state,
        )

    def send_weights(self) -> None:
        assert self.engine is not None
        self.engine.send_weights()


# ============================================================================
# vLLM engine wrapper
# ============================================================================


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
        output = None
        async for request_output in engine.generate(
            {"prompt": prompt},
            sampling_params,
            request_id=str(uuid.uuid4()),
        ):
            output = request_output
        return output

    return await asyncio.gather(*[gen_one(p) for p in prompts])


# ============================================================================
# Driver
# ============================================================================


async def main():
    # If a Ray cluster is already up but the workers can't import our
    # venv, point them at our interpreter via runtime_env. Also forward
    # NCCL env knobs to actors (Ray doesn't propagate env vars by default).
    # Forward PYTHONPATH so trainer and vLLM Ray actors find the local
    # `etha_*` modules without needing them to be installed.
    runtime_env: dict = {}
    venv_python = os.environ.get("VLLM_VENV_PYTHON")
    if venv_python:
        runtime_env["py_executable"] = venv_python
    env_vars: dict[str, str] = {
        k: v for k, v in os.environ.items() if k.startswith("NCCL_")
    }
    existing_pp = os.environ.get("PYTHONPATH", "")
    env_vars["PYTHONPATH"] = (
        _EXAMPLE_DIR + (os.pathsep + existing_pp if existing_pp else "")
    )
    runtime_env["env_vars"] = env_vars
    ray.init(runtime_env=runtime_env or None)

    snapshot_download(MODEL_NAME)

    # Pick a master addr/port for the trainer-internal NCCL PG before
    # any actor starts so all 4 can rendezvous in their __init__.
    trainer_master_addr = get_ip()
    trainer_master_port = get_open_port()

    # Launch 4 trainer actors.
    trainers = [
        EthaTrainerActor.remote(
            rank=r,
            world_size=TRAINER_WORLD_SIZE,
            master_addr=trainer_master_addr,
            master_port=trainer_master_port,
        )
        for r in range(TRAINER_WORLD_SIZE)
    ]
    print(f"[init] {TRAINER_WORLD_SIZE} trainer actors created; loading weights...")
    counts = ray.get([t.load_weights.remote() for t in trainers])
    print(f"[init] trainer state_dicts loaded: {counts[0]} tensors per rank")

    # Launch vLLM AsyncLLMEngine.
    # `worker_extension_cls` is the per-worker registration hook: vLLM's
    # worker_base resolves this qualname via importlib, and importing
    # `etha_engine` runs the WeightTransferEngineFactory.register_engine("etha", ...)
    # call at the bottom of that module — so the worker's factory sees
    # the backend before it constructs the engine. PYTHONPATH set above
    # makes the bare module name resolvable in the Ray actor.
    engine = create_async_engine(
        model=MODEL_NAME,
        enforce_eager=True,
        tensor_parallel_size=VLLM_TP_SIZE,
        data_parallel_size=VLLM_DP_SIZE,
        enable_expert_parallel=True,
        distributed_executor_backend="ray",
        data_parallel_backend="ray",
        weight_transfer_config=WeightTransferConfig(backend="etha"),
        worker_extension_cls="etha_engine.EthaWorkerExtension",
        load_format="dummy",
        gpu_memory_utilization=0.7,
    )
    print("[engine] AsyncLLMEngine ready (load_format=dummy).")

    prompts = [
        "The capital of France is",
        "Q: What is 2 + 2?\nA:",
        "The president of the United States is",
        "Once upon a time",
    ]
    sampling_params = SamplingParams(temperature=0, max_tokens=64)

    # Pre-sync generation: expect gibberish.
    print("[generate] BEFORE sync (dummy weights):")
    outs = await generate_batch(engine, prompts, sampling_params)
    for o in outs:
        print(f"  {o.prompt!r} -> {o.outputs[0].text!r}")

    # Rendezvous endpoint.
    master_address, master_port = ray.get(trainers[0].get_ip_port.remote())
    print(f"[transfer] rendezvous at {master_address}:{master_port}")

    # Trainer-side rendezvous (concurrent across all 4 trainer ranks).
    # Dispatch the ray tasks first so they run in the background, then
    # await the vLLM-side init — both halves rendezvous over NCCL while
    # this coroutine is suspended.
    trainer_init_handles = [
        t.init_transfer.remote(master_address, master_port) for t in trainers
    ]
    await engine.init_weight_transfer_engine(
        WeightTransferInitRequest(
            init_info=asdict(
                EthaWeightTransferInitInfo(
                    master_address=master_address,
                    master_port=master_port,
                    rank_offset=TRAINER_WORLD_SIZE,
                    world_size=TRANSFER_WORLD_SIZE,
                    trainer_attn_dp_replicate=TRAINER_ATTN_DP_REPLICATE,
                    trainer_attn_dp_shard=TRAINER_ATTN_DP_SHARD,
                    trainer_moe_dp_replicate=TRAINER_MOE_DP_REPLICATE,
                    trainer_moe_dp_shard=TRAINER_MOE_DP_SHARD,
                    trainer_ep_size=TRAINER_EP_SIZE,
                    vllm_dp_size=VLLM_DP_SIZE,
                    vllm_tp_size=VLLM_TP_SIZE,
                    vllm_ep_size=VLLM_TP_SIZE * VLLM_DP_SIZE,
                )
            )
        )
    )
    ray.get(trainer_init_handles)
    print("[transfer] rendezvous complete; plan baked.")

    # Pause, transfer, resume. Time each phase so we can see where wall
    # clock goes.
    print("[sync] pausing...")
    t_total = time.monotonic()
    t = time.monotonic()
    await engine.pause_generation(mode="abort", clear_cache=True)
    print(f"[sync]   pause_generation       {time.monotonic() - t:.3f}s")

    t = time.monotonic()
    await engine.start_weight_update(is_checkpoint_format=False)
    print(f"[sync]   start_weight_update    {time.monotonic() - t:.3f}s")

    t = time.monotonic()
    send_handles = [t_.send_weights.remote() for t_ in trainers]
    await engine.update_weights(
        WeightTransferUpdateRequest(
            update_info=asdict(EthaWeightTransferUpdateInfo(version=1))
        )
    )
    ray.get(send_handles)
    print(f"[sync]   update_weights (xfer)  {time.monotonic() - t:.3f}s")

    t = time.monotonic()
    await engine.finish_weight_update()
    print(f"[sync]   finish_weight_update   {time.monotonic() - t:.3f}s")

    t = time.monotonic()
    await engine.resume_generation()
    print(f"[sync]   resume_generation      {time.monotonic() - t:.3f}s")
    print(f"[sync] weights shipped (total {time.monotonic() - t_total:.3f}s).")

    # Post-sync generation: expect coherent Qwen3.
    print("[generate] AFTER sync (real weights):")
    outs = await generate_batch(engine, prompts, sampling_params)
    for o in outs:
        print(f"  {o.prompt!r} -> {o.outputs[0].text!r}")


if __name__ == "__main__":
    asyncio.run(main())
