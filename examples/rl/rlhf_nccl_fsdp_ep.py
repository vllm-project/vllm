# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
RLHF with FSDP2 training (4 GPUs) and vLLM expert-parallel inference (4 GPUs).

8-GPU layout:
  Training  — 4 GPUs, PyTorch FSDP2 (fully_shard)
  Inference — 4 GPUs, vLLM AsyncLLMEngine with expert parallelism +
              data parallelism (TP=1, DP=4, enable_expert_parallel
              → EP_SIZE = TP×DP = 4)

FSDP workers are Ray actors that form a single FSDP2 process group.
Rank 0 gathers full parameters via DTensor.full_tensor() and broadcasts
them to the vLLM inference engine through the NCCL weight-transfer API.

The inference engine uses AsyncLLMEngine which automatically spawns
DP worker processes (no manual placement group needed).  Weight sync
uses pause_generation / resume_generation.

Steps:
  1. Launch 4 FSDP training workers.
  2. Launch AsyncLLMEngine with EP+DP (dummy weights).
  3. Generate from prompts → gibberish (random weights).
  4. Pause generation, transfer weights from FSDP, resume.
  5. Generate from prompts → sensible output (synced weights).

Assumes a single-node cluster with 8 GPUs.
"""

import asyncio
import os
import uuid
from dataclasses import asdict

import ray
import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download
from torch.distributed.fsdp import fully_shard
from transformers import AutoModelForCausalLM

import vllm
from vllm import SamplingParams
from vllm.config import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
)
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLTrainerSendWeightsArgs,
    NCCLWeightTransferEngine,
    NCCLWeightTransferInitInfo,
    NCCLWeightTransferUpdateInfo,
)
from vllm.utils.network_utils import get_ip, get_open_port
from vllm.v1.executor import Executor

MODEL_NAME = "Qwen/Qwen3-30B-A3B"

FSDP_WORLD_SIZE = 4
INFERENCE_TP_SIZE = 1
INFERENCE_DP_SIZE = 4


@ray.remote(num_gpus=1)
class FSDPTrainWorker:
    """
    One FSDP2 training worker per GPU.  Four of these form the FSDP group.
    Rank 0 additionally handles weight transfer to the vLLM engine.
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

        self.transfer_port = None
        self.transfer_master_address = None
        self.model_update_group = None

    def get_rank(self):
        return self.rank

    # ---- weight-transfer setup (rank 0 only) ----

    def setup_transfer_endpoint(self):
        """Create the NCCL rendezvous endpoint for weight transfer."""
        assert self.rank == 0
        self.transfer_port = get_open_port()
        self.transfer_master_address = get_ip()
        return self.transfer_master_address, self.transfer_port

    def init_weight_transfer_group(self, transfer_world_size: int):
        """Join the weight-transfer NCCL group as rank 0 (the source)."""
        assert self.rank == 0
        self.model_update_group = NCCLWeightTransferEngine.trainer_init(
            dict(
                master_address=self.transfer_master_address,
                master_port=self.transfer_port,
                world_size=transfer_world_size,
            ),
        )

    def get_weight_metadata(self):
        """Return weight names, dtypes, and shapes captured before FSDP wrapping."""
        return self.weight_names, self.weight_dtype_names, self.weight_shapes

    # ---- collective ops (ALL FSDP ranks must call concurrently) ----

    def gather_and_broadcast_weights(self, packed: bool = True):
        """
        All-gather full parameters and broadcast them to vLLM.
        Only rank 0 performs the actual NCCL broadcast; others just
        participate in the FSDP all-gather.

        full_tensor() is a collective — all FSDP ranks must call it
        for each parameter in the same order.  Rank 0 additionally
        feeds each gathered tensor to the weight-transfer engine.
        """
        if self.rank == 0:

            def _full_param_iter():
                for name, param in self.model.named_parameters():
                    yield name, param.full_tensor()

            trainer_args = NCCLTrainerSendWeightsArgs(
                group=self.model_update_group,
                packed=packed,
            )
            NCCLWeightTransferEngine.trainer_send_weights(
                iterator=_full_param_iter(),
                trainer_args=trainer_args,
            )
        else:
            for _, param in self.model.named_parameters():
                param.full_tensor()


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
    ray.init()

    # Download model weights to local/shared disk once.
    local_model_path = snapshot_download(MODEL_NAME)
    print(f"[init] Model downloaded to {local_model_path}")

    # FSDP rendezvous address (single-node)
    fsdp_master_addr = get_ip()
    fsdp_master_port = get_open_port()

    # Launch 4 FSDP training workers.
    # Ray allocates 1 GPU per worker; AsyncLLMEngine's internal DP
    # placement groups will land on the remaining 4 GPUs.
    fsdp_workers = [
        FSDPTrainWorker.remote(
            local_model_path,
            rank,
            FSDP_WORLD_SIZE,
            fsdp_master_addr,
            fsdp_master_port,
        )
        for rank in range(FSDP_WORLD_SIZE)
    ]
    ray.get([w.get_rank.remote() for w in fsdp_workers])
    print(f"[init] {FSDP_WORLD_SIZE} FSDP training workers ready.")

    # Launch vLLM with expert parallelism + data parallelism.
    # AsyncLLMEngine with data_parallel_backend="ray" creates its own
    # placement groups internally — no manual placement group needed.
    print("[engine] Creating AsyncLLMEngine...")
    engine = create_async_engine(
        model=local_model_path,
        enforce_eager=True,
        tensor_parallel_size=INFERENCE_TP_SIZE,
        data_parallel_size=INFERENCE_DP_SIZE,
        enable_expert_parallel=True,
        distributed_executor_backend="ray",
        data_parallel_backend="ray",
        weight_transfer_config=WeightTransferConfig(backend="nccl"),
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

    # Generate with dummy weights — expect gibberish.
    print("[generate] Starting generation with dummy weights...")
    outputs = await generate_batch(engine, prompts, sampling_params)
    print("[generate] Generation complete.")

    print("-" * 60)
    print("BEFORE weight sync (dummy weights):")
    print("-" * 60)
    for output in outputs:
        print(f"Prompt: {output.prompt!r}")
        print(f"Generated: {output.outputs[0].text!r}")
        print("-" * 60)

    # --- Weight-transfer setup ---
    print("[transfer] Setting up weight-transfer endpoint...")
    transfer_addr, transfer_port = ray.get(
        fsdp_workers[0].setup_transfer_endpoint.remote()
    )
    print(f"[transfer] Endpoint ready at {transfer_addr}:{transfer_port}")

    transfer_world_size = INFERENCE_TP_SIZE * INFERENCE_DP_SIZE + 1
    print(
        f"[transfer] World size: {transfer_world_size} "
        f"(1 trainer + {INFERENCE_TP_SIZE * INFERENCE_DP_SIZE} vLLM workers)"
    )

    print("[transfer] Initializing NCCL groups...")
    train_handle = fsdp_workers[0].init_weight_transfer_group.remote(
        transfer_world_size
    )
    await engine.init_weight_transfer_engine(
        WeightTransferInitRequest(
            init_info=asdict(
                NCCLWeightTransferInitInfo(
                    master_address=transfer_addr,
                    master_port=transfer_port,
                    rank_offset=1,
                    world_size=transfer_world_size,
                )
            )
        )
    )
    ray.get(train_handle)
    print("[transfer] NCCL groups initialized.")

    # --- Pause, transfer weights, resume ---
    print("[sync] Pausing generation...")
    await engine.pause_generation(mode="abort")
    print("[sync] Generation paused.")

    names, dtype_names, shapes = ray.get(fsdp_workers[0].get_weight_metadata.remote())
    print(f"[sync] Got metadata for {len(names)} parameters.")

    print("[sync] Starting weight update...")
    await engine.start_weight_update(is_checkpoint_format=True)

    print("[sync] Broadcasting weights from FSDP → vLLM...")
    broadcast_handles = [
        w.gather_and_broadcast_weights.remote(packed=True) for w in fsdp_workers
    ]
    await engine.update_weights(
        WeightTransferUpdateRequest(
            update_info=asdict(
                NCCLWeightTransferUpdateInfo(
                    names=names,
                    dtype_names=dtype_names,
                    shapes=shapes,
                    packed=True,
                )
            )
        )
    )
    ray.get(broadcast_handles)

    await engine.finish_weight_update()
    print("[sync] Weight broadcast complete.")

    print("[sync] Resuming generation...")
    await engine.resume_generation()
    print("[sync] Generation resumed.")

    # Generate with synced weights — expect sensible output.
    print("[generate] Starting generation with synced weights...")
    outputs_updated = await generate_batch(engine, prompts, sampling_params)
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
