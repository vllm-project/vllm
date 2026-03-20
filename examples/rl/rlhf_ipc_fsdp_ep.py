# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
RLHF with FSDP2 training and vLLM expert-parallel inference using **CUDA IPC**
weight transfer and **packed** tensors.

This mirrors ``rlhf_nccl_fsdp_ep.py`` in flow (FSDP2 → pause → sync → resume)
but uses IPC instead of NCCL. IPC requires the trainer and each vLLM worker to
share the **same physical GPU**, so this script uses **4 GPUs total** (not 8):
each GPU runs one FSDP rank and the matching DP inference rank.

Layout (4 GPUs, TP=1, DP=4, EP):
  * One Ray placement group per GPU, matching vLLM's DP PG shape:
    ``world_size`` GPU bundles + one CPU bundle for ``EngineCoreActor``.
  * FSDP worker ``rank`` is scheduled on bundle 0 of PG ``rank`` with a
    fractional GPU; vLLM's Ray worker uses the same bundle with a complementary
    fraction (``VLLM_RAY_PER_WORKER_GPUS``).

Uses the built-in ``ray`` send_mode: the engine is wrapped in a Ray actor,
and FSDP rank 0 calls ``trainer_send_weights`` with ``send_mode="ray"`` and
``llm_handle=engine_actor``.

**Memory**: You colocate a full training replica and inference on each GPU.
The same model name as the NCCL example is used; reduce size or fractions if
you hit OOM.

Assumes a single-node cluster with at least 4 GPUs.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import asdict

import ray
import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torch.distributed.fsdp import fully_shard
from transformers import AutoModelForCausalLM

import vllm
import vllm.envs as vllm_envs
from vllm import SamplingParams
from vllm.config import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
)
from vllm.distributed.weight_transfer.ipc_engine import (
    IPCTrainerSendWeightsArgs,
    IPCWeightTransferEngine,
    IPCWeightTransferInitInfo,
)
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_ip, get_open_port
from vllm.v1.executor import Executor

# Fractional GPUs: trainer + vLLM Ray worker must fit in bundle 0 (sum <= 1.0).
TRAIN_GPU_FRACTION = float(os.environ.get("RLHF_IPC_TRAIN_GPU_FRACTION", "0.42"))
VLLM_GPU_FRACTION = float(os.environ.get("RLHF_IPC_VLLM_GPU_FRACTION", "0.42"))

MODEL_NAME = "Qwen/Qwen3-30B-A3B"

FSDP_WORLD_SIZE = 4
INFERENCE_TP_SIZE = 1
INFERENCE_DP_SIZE = 4


def build_colocated_dp_placement_groups(dp_size: int, world_size: int):
    """
    Build one placement group per DP rank, same bundle pattern as
    ``CoreEngineActorManager.create_dp_placement_groups`` (non-span path).
    """
    pack = vllm_envs.VLLM_RAY_DP_PACK_STRATEGY
    strategy = "STRICT_PACK" if pack in ("strict", "fill") else "PACK"
    node_ip = get_ip()
    node_key = f"node:{node_ip}"
    device_str = current_platform.ray_device_key
    pgs = []
    for _ in range(dp_size):
        device_bundle = [{device_str: 1.0, node_key: 0.001}]
        bundles = device_bundle * world_size + [{"CPU": 1.0}]
        pgs.append(placement_group(bundles=bundles, strategy=strategy))
    return pgs


@ray.remote(num_cpus=1)
class EngineActor:
    """Ray actor wrapping AsyncLLMEngine for use with send_mode='ray'."""

    def __init__(self, placement_groups, local_dp_ranks, **engine_kwargs):
        # Remove CUDA_VISIBLE_DEVICES so get_env_vars_to_copy doesn't
        # propagate the empty value (this actor has no GPU) to
        # DPMoEEngineCoreActors and their child RayWorkerWrapper actors.
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        engine_args = vllm.AsyncEngineArgs(**engine_kwargs)
        vllm_config = engine_args.create_engine_config()
        vllm_config.parallel_config.ray_data_parallel_placement_groups = (
            placement_groups
        )
        vllm_config.parallel_config.ray_data_parallel_local_dp_ranks = local_dp_ranks
        executor_class = Executor.get_class(vllm_config)
        self._engine = vllm.AsyncLLMEngine(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_requests=engine_args.enable_log_requests,
            log_stats=not engine_args.disable_log_stats,
        )

    async def update_weights(self, request: dict) -> None:
        await self._engine.update_weights(
            WeightTransferUpdateRequest(update_info=request["update_info"])
        )

    async def init_weight_transfer_engine(self, request: dict) -> None:
        await self._engine.init_weight_transfer_engine(
            WeightTransferInitRequest(init_info=request["init_info"])
        )

    async def pause_generation(self, mode: str = "abort") -> None:
        await self._engine.pause_generation(mode=mode)

    async def resume_generation(self) -> None:
        await self._engine.resume_generation()

    async def generate(self, prompts: list[str], sampling_params) -> list:
        results = []
        for prompt in prompts:
            output = None
            async for req_out in self._engine.generate(
                {"prompt": prompt},
                sampling_params,
                request_id=str(uuid.uuid4()),
            ):
                output = req_out
            results.append(output)
        return results


@ray.remote(num_cpus=0, num_gpus=TRAIN_GPU_FRACTION)
class FSDPTrainWorker:
    """One FSDP2 worker per GPU; colocated with vLLM DP rank via placement group."""

    def __init__(
        self,
        model_name: str,
        rank: int,
        fsdp_world_size: int,
        fsdp_master_addr: str,
        fsdp_master_port: int,
    ):
        self.rank = rank

        os.environ["MASTER_ADDR"] = fsdp_master_addr
        os.environ["MASTER_PORT"] = str(fsdp_master_port)

        dist.init_process_group(backend="nccl", rank=rank, world_size=fsdp_world_size)
        torch.accelerator.set_device_index(0)

        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        )

        self.weight_names = [n for n, _ in model.named_parameters()]
        self.weight_dtype_names = [
            str(p.dtype).split(".")[-1] for _, p in model.named_parameters()
        ]
        self.weight_shapes = [list(p.shape) for _, p in model.named_parameters()]

        for layer in model.model.layers:
            fully_shard(layer)
        fully_shard(model)

        self.model = model

    def get_rank(self):
        return self.rank

    def get_weight_metadata(self):
        return self.weight_names, self.weight_dtype_names, self.weight_shapes

    def gather_and_broadcast_weights_ipc(
        self, llm_handle: ray.actor.ActorHandle, packed: bool = True
    ):
        """All-gather full params; rank 0 sends packed IPC chunks via ray send_mode.

        All ranks must call trainer_send_weights so they participate in the
        all_gather_object collective inside _all_gather_and_merge_handles.
        Only rank 0 actually sends the payload to vLLM (gated by _is_rank_zero).
        """

        def _full_param_iter():
            for name, param in self.model.named_parameters():
                yield name, param.full_tensor()

        trainer_args = IPCTrainerSendWeightsArgs(
            send_mode="ray",
            llm_handle=llm_handle,
            packed=packed,
        )
        IPCWeightTransferEngine.trainer_send_weights(
            iterator=_full_param_iter(),
            trainer_args=trainer_args,
        )


async def main():
    ray.init(
        runtime_env={
            "env_vars": {
                "PYTHONPATH": "/home/ray/default/personal/vllm",
                "VLLM_RAY_PER_WORKER_GPUS": str(VLLM_GPU_FRACTION),
                "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
            }
        }
    )

    assert TRAIN_GPU_FRACTION + VLLM_GPU_FRACTION <= 1.0, (
        "Train + vLLM GPU fractions must sum to at most 1.0 per bundle."
    )

    local_model_path = snapshot_download(MODEL_NAME)
    print(f"[init] Model downloaded to {local_model_path}")

    fsdp_master_addr = get_ip()
    fsdp_master_port = get_open_port()

    placement_groups = build_colocated_dp_placement_groups(
        FSDP_WORLD_SIZE, INFERENCE_TP_SIZE
    )
    ray.get([pg.ready() for pg in placement_groups])
    print(f"[init] {len(placement_groups)} colocated placement groups ready.")

    scheduling = [
        PlacementGroupSchedulingStrategy(
            placement_group=placement_groups[r],
            placement_group_bundle_index=0,
            placement_group_capture_child_tasks=True,
        )
        for r in range(FSDP_WORLD_SIZE)
    ]

    fsdp_workers = [
        FSDPTrainWorker.options(scheduling_strategy=scheduling[r]).remote(
            local_model_path,
            r,
            FSDP_WORLD_SIZE,
            fsdp_master_addr,
            fsdp_master_port,
        )
        for r in range(FSDP_WORLD_SIZE)
    ]
    ray.get([w.get_rank.remote() for w in fsdp_workers])
    print(f"[init] {FSDP_WORLD_SIZE} FSDP workers colocated on DP bundles.")

    local_dp_ranks = list(range(FSDP_WORLD_SIZE))

    print("[engine] Creating EngineActor (AsyncLLMEngine + colocation PGs)...")
    engine_actor = EngineActor.remote(
        placement_groups,
        local_dp_ranks,
        model=local_model_path,
        enforce_eager=True,
        tensor_parallel_size=INFERENCE_TP_SIZE,
        data_parallel_size=INFERENCE_DP_SIZE,
        enable_expert_parallel=True,
        distributed_executor_backend="ray",
        data_parallel_backend="ray",
        weight_transfer_config=WeightTransferConfig(backend="ipc"),
        load_format="dummy",
        gpu_memory_utilization=0.35,
    )
    print("[engine] EngineActor created (PGs provided).")

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0)

    print("[generate] Starting generation with dummy weights...")
    outputs = ray.get(engine_actor.generate.remote(prompts, sampling_params))
    print("[generate] Generation complete.")

    print("-" * 60)
    print("BEFORE weight sync (dummy weights):")
    print("-" * 60)
    for output in outputs:
        print(f"Prompt: {output.prompt!r}")
        print(f"Generated: {output.outputs[0].text!r}")
        print("-" * 60)

    print("[transfer] Initializing IPC weight transfer (no-op init)...")
    ray.get(
        engine_actor.init_weight_transfer_engine.remote(
            dict(init_info=asdict(IPCWeightTransferInitInfo()))
        )
    )

    print("[sync] Pausing generation...")
    ray.get(engine_actor.pause_generation.remote(mode="abort"))
    print("[sync] Generation paused.")

    names, dtype_names, shapes = ray.get(fsdp_workers[0].get_weight_metadata.remote())
    print(f"[sync] Got metadata for {len(names)} parameters.")

    print("[sync] Packed IPC transfer FSDP → vLLM (ray send_mode)...")
    ray.get(
        [
            w.gather_and_broadcast_weights_ipc.remote(engine_actor, packed=True)
            for w in fsdp_workers
        ]
    )
    print("[sync] Weight transfer complete.")

    print("[sync] Resuming generation...")
    ray.get(engine_actor.resume_generation.remote())
    print("[sync] Generation resumed.")

    print("[generate] Starting generation with synced weights...")
    outputs_updated = ray.get(engine_actor.generate.remote(prompts, sampling_params))
    print("[generate] Generation complete.")

    print("-" * 60)
    print("AFTER weight sync (real weights):")
    print("-" * 60)
    for output in outputs_updated:
        print(f"Prompt: {output.prompt!r}")
        print(f"Generated: {output.outputs[0].text!r}")
        print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main())
