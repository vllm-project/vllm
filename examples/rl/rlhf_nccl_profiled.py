# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Profiled NCCL-broadcast weight sync, for direct comparison against
`rlhf_etha.py`.

Topology matches `rlhf_etha.py` (same 8-GPU footprint):
  Training  — 4 FSDP2 actors (Ray, num_gpus=1 each), Qwen3-30B-A3B
              held in bfloat16, sharded by FSDP.
  Inference — vLLM AsyncLLMEngine with DP=2, TP=2,
              enable_expert_parallel (EP_SIZE = TP*DP = 4).

This is the existing `NCCLWeightTransferEngine` pattern: FSDP rank 0
all-gathers each parameter and broadcasts the full tensor to every
vLLM worker, which then calls `model.load_weights` to re-shard. The
other 3 FSDP ranks only participate in the FSDP all-gather.

The same wall-clock timing hooks as the Etha example are wired around
`pause → start → update → finish → resume` so the two runs can be
diffed phase-by-phase.

This file carries the same env-var / runtime_env hacks as
`rlhf_etha.py` to work around the cluster's NCCL plugin mismatch and
to point Ray actors at our venv.
"""

import asyncio
import os
import time
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

MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"

FSDP_WORLD_SIZE = 4
INFERENCE_TP_SIZE = 2
INFERENCE_DP_SIZE = 2


@ray.remote(num_gpus=1)
class FSDPTrainWorker:
    """One FSDP2 training worker per GPU; 4 form the trainer side."""

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

        # Multi-backend PG: NCCL for tensor collectives, gloo for the
        # object collectives FSDP uses internally.
        dist.init_process_group(
            backend="cuda:nccl,cpu:gloo", rank=rank, world_size=fsdp_world_size
        )
        torch.cuda.set_device(0)

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

    def get_rank(self) -> int:
        return self.rank

    def setup_transfer_endpoint(self) -> tuple[str, int]:
        assert self.rank == 0
        self.transfer_port = get_open_port()
        self.transfer_master_address = get_ip()
        return self.transfer_master_address, self.transfer_port

    def init_weight_transfer_group(self, transfer_world_size: int) -> None:
        assert self.rank == 0
        self.model_update_group = NCCLWeightTransferEngine.trainer_init(
            dict(
                master_address=self.transfer_master_address,
                master_port=self.transfer_port,
                world_size=transfer_world_size,
            ),
        )

    def get_weight_metadata(self):
        return self.weight_names, self.weight_dtype_names, self.weight_shapes

    def gather_and_broadcast_weights(self, packed: bool = True) -> float:
        """All-gather full params; rank 0 also broadcasts each to vLLM.

        Returns wall-clock seconds for the full pass on this rank
        (FSDP all-gather + NCCL broadcast for rank 0; pure all-gather
        for others).
        """
        t0 = time.monotonic()
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
        torch.cuda.synchronize()
        return time.monotonic() - t0


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


async def main():
    # Forward venv interpreter + NCCL env vars to Ray actors. Same as
    # the Etha example — cluster's stock Python doesn't have our deps,
    # and the bundled NCCL plugin is incompatible with vLLM's NCCL.
    runtime_env: dict = {}
    venv_python = os.environ.get("VLLM_VENV_PYTHON")
    if venv_python:
        runtime_env["py_executable"] = venv_python
    nccl_env = {k: v for k, v in os.environ.items() if k.startswith("NCCL_")}
    if nccl_env:
        runtime_env["env_vars"] = nccl_env
    ray.init(runtime_env=runtime_env or None)

    local_model_path = snapshot_download(MODEL_NAME)
    print(f"[init] Model downloaded to {local_model_path}")

    fsdp_master_addr = get_ip()
    fsdp_master_port = get_open_port()

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
    print("[engine] AsyncLLMEngine ready.")

    prompts = [
        "The capital of France is",
        "Q: What is 2 + 2?\nA:",
        "The president of the United States is",
        "Once upon a time",
    ]
    sampling_params = SamplingParams(temperature=0, max_tokens=64)

    print("[generate] BEFORE sync (dummy weights):")
    outs = await generate_batch(engine, prompts, sampling_params)
    for o in outs:
        print(f"  {o.prompt!r} -> {o.outputs[0].text!r}")

    # --- Weight-transfer setup ---
    transfer_addr, transfer_port = ray.get(
        fsdp_workers[0].setup_transfer_endpoint.remote()
    )
    transfer_world_size = INFERENCE_TP_SIZE * INFERENCE_DP_SIZE + 1
    print(f"[transfer] rendezvous at {transfer_addr}:{transfer_port}")

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

    names, dtype_names, shapes = ray.get(fsdp_workers[0].get_weight_metadata.remote())
    print(f"[sync] {len(names)} parameters in the broadcast plan.")

    # --- Pause, transfer, resume (mirror rlhf_etha.py timing) ---
    print("[sync] pausing...")
    t_total = time.monotonic()

    t = time.monotonic()
    await engine.pause_generation(mode="abort")
    print(f"[sync]   pause_generation       {time.monotonic() - t:.3f}s")

    t = time.monotonic()
    await engine.start_weight_update(is_checkpoint_format=True)
    print(f"[sync]   start_weight_update    {time.monotonic() - t:.3f}s")

    t = time.monotonic()
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
    trainer_times = ray.get(broadcast_handles)
    print(f"[sync]   update_weights (xfer)  {time.monotonic() - t:.3f}s")
    for r, t_r in enumerate(trainer_times):
        print(f"[sync]     fsdp rank {r}: gather+broadcast {t_r:.3f}s")

    t = time.monotonic()
    await engine.finish_weight_update()
    print(f"[sync]   finish_weight_update   {time.monotonic() - t:.3f}s")

    t = time.monotonic()
    await engine.resume_generation()
    print(f"[sync]   resume_generation      {time.monotonic() - t:.3f}s")
    print(f"[sync] weights shipped (total {time.monotonic() - t_total:.3f}s).")

    print("[generate] AFTER sync (real weights):")
    outs_after = await generate_batch(engine, prompts, sampling_params)
    for o in outs_after:
        print(f"  {o.prompt!r} -> {o.outputs[0].text!r}")


if __name__ == "__main__":
    asyncio.run(main())
