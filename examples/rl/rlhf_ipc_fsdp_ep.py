# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
RLHF with FSDP2 training and vLLM expert-parallel inference using **CUDA IPC**
weight transfer and **packed** tensors.

Layout (4 GPUs, TP=1, DP=4, EP):
  * One Ray placement group per GPU.
  * Each PG holds one FSDP training worker and one vLLM ``LLM`` instance
    (sync API) using fractional GPUs so both fit on the same device.
  * The 4 ``LLM`` instances form a DP group via env-var-based SPMD
    coordination (``VLLM_DP_RANK``, ``VLLM_DP_SIZE``, etc.), the same
    mechanism used by ``examples/offline_inference/data_parallel.py``.
  * A ``DataParallelInferenceEngine`` actor spawns all 4 LLM actors,
    waits for initialization, and orchestrates generation / weight-sync.

Uses the built-in ``ray`` send_mode: each FSDP worker calls
``trainer_send_weights`` targeting its colocated LLM actor.

Assumes a single-node cluster with at least 4 GPUs.
"""

from __future__ import annotations

import os
from dataclasses import asdict

import ray
import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torch.distributed._tensor import DTensor
from torch.distributed.fsdp import fully_shard
from transformers import AutoModelForCausalLM

from vllm import LLM, SamplingParams
from vllm.config import WeightTransferConfig
from vllm.distributed.weight_transfer.ipc_engine import (
    IPCTrainerSendWeightsArgs,
    IPCWeightTransferEngine,
    IPCWeightTransferInitInfo,
)
from vllm.utils.network_utils import get_ip, get_open_port

TRAIN_GPU_FRACTION = float(os.environ.get("RLHF_IPC_TRAIN_GPU_FRACTION", "0.42"))
VLLM_GPU_FRACTION = float(os.environ.get("RLHF_IPC_VLLM_GPU_FRACTION", "0.42"))

MODEL_NAME = "Qwen/Qwen3-30B-A3B"

FSDP_WORLD_SIZE = 4
INFERENCE_TP_SIZE = 1
INFERENCE_DP_SIZE = 4


class MyLLM(LLM):
    """LLM subclass that configures DP env vars for SPMD coordination."""

    def __init__(
        self,
        *args,
        dp_rank: int = 0,
        dp_size: int = 1,
        dp_master_ip: str = "127.0.0.1",
        dp_master_port: int = 0,
        **kwargs,
    ):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(VLLM_GPU_FRACTION)
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = "0"
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        os.environ["VLLM_DP_RANK"] = str(dp_rank)
        os.environ["VLLM_DP_RANK_LOCAL"] = str(dp_rank)
        os.environ["VLLM_DP_SIZE"] = str(dp_size)
        os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
        os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

        super().__init__(*args, **kwargs)

    def ready(self):
        return True


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

    def gather_and_broadcast_weights_ipc(self, llm_handle, packed: bool = True):
        """All-gather full params; all ranks create IPC handles, rank 0 sends.

        All ranks must call trainer_send_weights so they participate in the
        all_gather_object collective inside _all_gather_and_merge_handles.
        Only rank 0 actually sends the payload to vLLM (gated by _is_rank_zero).
        """

        def _full_param_iter():
            params = self.model.state_dict()
            for name in list(params.keys()):
                param = params.pop(name)
                if isinstance(param, DTensor):
                    tensor = param.full_tensor().detach().contiguous()
                else:
                    tensor = param.detach().contiguous()
                del param
                yield name, tensor

        trainer_args = IPCTrainerSendWeightsArgs(
            send_mode="ray",
            llm_handle=llm_handle,
            packed=packed,
            packed_buffer_size_bytes=1024 * 1024 * 1024,  # 1 GB
        )
        IPCWeightTransferEngine.trainer_send_weights(
            iterator=_full_param_iter(),
            trainer_args=trainer_args,
        )


@ray.remote(num_cpus=1)
class DataParallelInferenceEngine:
    """Manages a pool of DP-sharded vLLM LLM actors.

    Spawns one MyLLM actor per placement group, waits for all engines to
    finish initializing, and exposes generation / weight-sync helpers.
    """

    def __init__(
        self,
        model: str,
        pgs: list,
        dp_master_ip: str,
        dp_master_port: int,
    ):
        dp_size = len(pgs)
        self.llm_actors = []
        for r in range(dp_size):
            sched = PlacementGroupSchedulingStrategy(
                placement_group=pgs[r],
                placement_group_capture_child_tasks=True,
            )
            actor = (
                ray.remote(num_cpus=0, num_gpus=0)(MyLLM)
                .options(scheduling_strategy=sched)
                .remote(
                    model=model,
                    enforce_eager=True,
                    tensor_parallel_size=INFERENCE_TP_SIZE,
                    distributed_executor_backend="ray",
                    enable_expert_parallel=True,
                    gpu_memory_utilization=0.35,
                    weight_transfer_config=WeightTransferConfig(backend="ipc"),
                    enable_sleep_mode=True,
                    load_format="dummy",
                    dp_rank=r,
                    dp_size=dp_size,
                    dp_master_ip=dp_master_ip,
                    dp_master_port=dp_master_port,
                )
            )
            self.llm_actors.append(actor)

        ray.get([actor.ready.remote() for actor in self.llm_actors])

    def get_llm_actors(self):
        return self.llm_actors

    def generate(self, prompts: list[str], sampling_params):
        """Distribute prompts round-robin across DP ranks and collect results."""
        dp_size = len(self.llm_actors)
        per_rank: list[list[str]] = [[] for _ in range(dp_size)]
        indices: list[list[int]] = [[] for _ in range(dp_size)]

        for i, prompt in enumerate(prompts):
            rank = i % dp_size
            per_rank[rank].append(prompt)
            indices[rank].append(i)

        refs = [
            actor.generate.remote(per_rank[r], sampling_params)
            for r, actor in enumerate(self.llm_actors)
            if per_rank[r]
        ]
        all_outputs = ray.get(refs)

        ordered = [None] * len(prompts)
        rank_idx = 0
        for r in range(dp_size):
            if per_rank[r]:
                for local_i, orig_i in enumerate(indices[r]):
                    ordered[orig_i] = all_outputs[rank_idx][local_i]
                rank_idx += 1
        return ordered

    def init_weight_transfer(self):
        ray.get(
            [
                actor.init_weight_transfer_engine.remote(
                    dict(init_info=asdict(IPCWeightTransferInitInfo()))
                )
                for actor in self.llm_actors
            ]
        )

    def sleep(self, level: int = 0):
        ray.get([actor.sleep.remote(level=level) for actor in self.llm_actors])

    def wake_up(self, tags: list[str] | None = None):
        ray.get([actor.wake_up.remote(tags=tags) for actor in self.llm_actors])


def main():
    ray.init(
        runtime_env={
            "env_vars": {
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
    dp_master_port = get_open_port()
    dp_master_ip = get_ip()

    # Create one placement group per DP rank (one GPU each).
    pgs = []
    for _ in range(INFERENCE_DP_SIZE):
        pg = placement_group([{"GPU": 1, "CPU": 1}])
        pgs.append(pg)
    ray.get([pg.ready() for pg in pgs])
    print(f"[init] {len(pgs)} placement groups ready.")

    # Launch FSDP training workers, one per PG.
    scheduling = [
        PlacementGroupSchedulingStrategy(
            placement_group=pgs[r],
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
    print(f"[init] {FSDP_WORLD_SIZE} FSDP workers ready.")

    # Launch DP inference engine (spawns and initializes all LLM actors).
    inference_engine = DataParallelInferenceEngine.remote(
        model=local_model_path,
        pgs=pgs,
        dp_master_ip=dp_master_ip,
        dp_master_port=dp_master_port,
    )
    llm_actors = ray.get(inference_engine.get_llm_actors.remote())
    print(f"[init] {INFERENCE_DP_SIZE} LLM actors ready.")

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0)

    print("[generate] Generating with dummy weights...")
    outputs = ray.get(inference_engine.generate.remote(prompts, sampling_params))
    print("-" * 60)
    print("BEFORE weight sync (dummy weights):")
    print("-" * 60)
    for output in outputs:
        print(f"Prompt: {output.prompt!r}")
        print(f"Generated: {output.outputs[0].text!r}")
        print("-" * 60)

    # --- Weight transfer ---
    print("[transfer] Initializing IPC weight transfer...")
    ray.get(inference_engine.init_weight_transfer.remote())

    # Two-phase sleep/wake pattern (same as SkyRL):
    # 1. sleep(level=1) — offload weights to CPU, discard KV cache
    # 2. wake_up(tags=["weights"]) — bring weights back to GPU (KV cache still free)
    # 3. IPC weight transfer — overwrite weights, plenty of room without KV cache
    # 4. wake_up(tags=["kv_cache"]) — re-allocate KV cache for inference
    print("[sync] Sleeping engines (offload weights + free KV cache)...")
    ray.get(inference_engine.sleep.remote(level=1))

    print("[sync] Waking weights (KV cache stays free)...")
    ray.get(inference_engine.wake_up.remote(tags=["weights"]))

    print("[sync] Packed IPC transfer FSDP → vLLM...")
    ray.get(
        [
            w.gather_and_broadcast_weights_ipc.remote(llm_actors, packed=True)
            for w in fsdp_workers
        ]
    )
    print("[sync] Weight transfer complete.")

    print("[sync] Waking KV cache + scheduling...")
    ray.get(inference_engine.wake_up.remote(tags=["kv_cache", "scheduling"]))

    print("[generate] Generating with synced weights...")
    outputs_updated = ray.get(
        inference_engine.generate.remote(prompts, sampling_params)
    )
    print("-" * 60)
    print("AFTER weight sync (real weights):")
    print("-" * 60)
    for output in outputs_updated:
        print(f"Prompt: {output.prompt!r}")
        print(f"Generated: {output.outputs[0].text!r}")
        print("-" * 60)


if __name__ == "__main__":
    main()
