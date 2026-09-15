# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
RLHF with FSDP2 training and vLLM tensor-parallel inference using **NCCL M2N**
sharding-aware weight transfer.

Layout (4 GPUs, no colocation):
  * GPUs 0-1: two FSDP2 training workers, one per GPU.
  * GPUs 2-3: one vLLM ``LLM`` actor with ``tensor_parallel_size=2``.

The trainer and the inference workers share one NCCL communicator of 4 ranks:
trainer ranks ``[0, 2)`` and inference ranks ``[2, 4)``. Every parameter moves
with a single ``reshard`` that redistributes it from the FSDP layout to the
inference layout — the trainer sends its local shards and never all-gathers a
full tensor, which broadcast-based weight sync would force it to do.

Every FSDP rank builds an ``M2NTrainerWeightTransferEngine`` and calls
``send_weights()``; all ranks run every reshard, and only rank 0 drives the
inference side through its ``RayVLLMWeightSyncClient``.

Requires the ``nccl-extensions`` package (NCCL 2.30.5+) and a
``VLLM_NCCL_SO_PATH`` pointing at the same ``libnccl.so`` that
``libnccl_m2n.so`` was linked against.

This example was written for 4xH100.
"""

from __future__ import annotations

import os

import ray
import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download
from torch.distributed.fsdp import fully_shard
from transformers import AutoModelForCausalLM

from vllm import LLM, SamplingParams
from vllm.config import WeightTransferConfig
from vllm.distributed.weight_transfer import (
    RayVLLMWeightSyncClient,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.m2n_source import DTensorModuleSource
from vllm.distributed.weight_transfer.m2n_trainer import M2NTrainerInitInfo
from vllm.utils.network_utils import get_ip, get_open_port

MODEL_NAME = "Qwen/Qwen3-0.6B"

FSDP_WORLD_SIZE = 2
INFERENCE_TP_SIZE = 2


class MyLLM(LLM):
    """LLM subclass that keeps Ray from pinning it to a single device."""

    def __init__(self, *args, **kwargs):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        super().__init__(*args, **kwargs)

    def ready(self):
        return True


@ray.remote(num_cpus=1, num_gpus=1)
class FSDPTrainWorker:
    """One FSDP2 worker per GPU. Rank 0 is the weight-transfer sender."""

    def __init__(
        self,
        model_name: str,
        rank: int,
        fsdp_master_addr: str,
        fsdp_master_port: int,
    ):
        self.rank = rank

        os.environ["MASTER_ADDR"] = fsdp_master_addr
        os.environ["MASTER_PORT"] = str(fsdp_master_port)

        dist.init_process_group(backend="nccl", rank=rank, world_size=FSDP_WORLD_SIZE)
        torch.accelerator.set_device_index(0)

        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        ).cuda()
        for layer in model.model.layers:
            fully_shard(layer)
        fully_shard(model)
        self.model = model

    def ready(self):
        return True

    def setup_engine(
        self, llm_handle, master_address, master_port, world_size, num_workers
    ):
        """Build the trainer engine on every FSDP rank.

        `DTensorModuleSource` reads each parameter's FSDP device mesh and
        placements, so the engine knows the source layout without gathering
        anything. Rank 0 additionally drives the inference-side handshake.
        """
        self.engine = WeightTransferTrainerFactory.trainer_init(
            init_info=M2NTrainerInitInfo(
                master_address=master_address,
                master_port=master_port,
                world_size=world_size,
                num_trainer_ranks=FSDP_WORLD_SIZE,
                # One DP group of TP=2 workers; declared so both sides
                # describe the destination the same way.
                dst_mesh_dims=(num_workers // INFERENCE_TP_SIZE, INFERENCE_TP_SIZE),
                rank=self.rank,  # FSDP rank; sender is 0
            ),
            client=RayVLLMWeightSyncClient(llm_handle),
            source=DTensorModuleSource(self.model, FSDP_WORLD_SIZE),
        )

    def send_weights(self):
        """Called on all ranks concurrently; every rank runs every reshard."""
        self.engine.send_weights()


def main():
    ray.init()

    local_model_path = snapshot_download(MODEL_NAME)
    print(f"[init] Model downloaded to {local_model_path}")

    fsdp_master_addr = get_ip()
    fsdp_master_port = get_open_port()

    fsdp_workers = [
        FSDPTrainWorker.remote(
            local_model_path, rank, fsdp_master_addr, fsdp_master_port
        )
        for rank in range(FSDP_WORLD_SIZE)
    ]
    ray.get([w.ready.remote() for w in fsdp_workers])
    print(f"[init] {FSDP_WORLD_SIZE} FSDP workers ready.")

    llm = ray.remote(num_cpus=0, num_gpus=0)(MyLLM).remote(
        model=local_model_path,
        enforce_eager=True,
        tensor_parallel_size=INFERENCE_TP_SIZE,
        distributed_executor_backend="ray",
        weight_transfer_config=WeightTransferConfig(backend="nccl_m2n"),
        load_format="dummy",
    )
    ray.get(llm.ready.remote())
    num_workers = ray.get(llm.get_world_size.remote())
    print(f"[init] vLLM ready with {num_workers} inference workers.")

    prompts = [
        "Hello, my name is",
        "The capital of France is",
    ]
    sampling_params = SamplingParams(temperature=0)

    outputs = ray.get(llm.generate.remote(prompts, sampling_params))
    print("-" * 60)
    print("BEFORE weight sync (dummy weights):")
    for output in outputs:
        print(f"Prompt: {output.prompt!r}")
        print(f"Generated: {output.outputs[0].text!r}")
    print("-" * 60)

    # Trainer ranks [0, FSDP_WORLD_SIZE) and inference ranks after them share
    # one communicator, so the two meshes are contiguous rank intervals.
    master_address = get_ip()
    master_port = get_open_port()
    world_size = FSDP_WORLD_SIZE + num_workers

    print("[transfer] Initializing nccl_m2n weight transfer (all FSDP ranks)...")
    ray.get(
        [
            w.setup_engine.remote(
                llm, master_address, master_port, world_size, num_workers
            )
            for w in fsdp_workers
        ]
    )

    print("[sync] Resharding FSDP -> vLLM...")
    ray.get([w.send_weights.remote() for w in fsdp_workers])
    print("[sync] Weight transfer complete.")

    outputs_updated = ray.get(llm.generate.remote(prompts, sampling_params))
    print("-" * 60)
    print("AFTER weight sync (real weights):")
    for output in outputs_updated:
        print(f"Prompt: {output.prompt!r}")
        print(f"Generated: {output.outputs[0].text!r}")
    print("-" * 60)


if __name__ == "__main__":
    main()
