# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Demonstrate checkpoint-coordinate sparse NCCL updates with expert parallelism.

The trainer and vLLM start from the same Qwen3 MoE checkpoint. The trainer
modifies rows from two global experts, converts its fused expert storage back to
per-expert checkpoint coordinates, and sends both patches through one
``send_weights()`` lifecycle. Every vLLM rank receives both patches; the native
loader applies its local expert and skips the foreign expert.

This example uses three GPUs on one node: one for the full BF16 Hugging Face
trainer model and two for a TP2/EP2 vLLM inference engine. Unspecified checkpoint
elements keep their initialized values, so sparse updates require a known shared
baseline. ``SPARSE_NCCL_MODEL`` may point to a compatible local Qwen3 MoE
checkpoint.
"""

import os
from contextlib import suppress

import ray
import torch
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoModelForCausalLM

from vllm import LLM, SamplingParams
from vllm.config import WeightTransferConfig
from vllm.distributed.weight_transfer import (
    RayVLLMWeightSyncClient,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.sparse_nccl_engine import (
    SparseNCCLTrainerInitInfo,
    SparseWeightPatch,
)
from vllm.utils.network_utils import get_ip, get_open_port

MODEL_NAME = os.environ.get("SPARSE_NCCL_MODEL", "Qwen/Qwen3-30B-A3B")
INFERENCE_TP_SIZE = 2
PATCHED_LAYER = 0
PATCHED_ROWS = 2
PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
SAMPLING_PARAMS = SamplingParams(temperature=0.0, max_tokens=1)


class MyLLM(LLM):
    """Configure the vLLM worker for Ray placement group execution."""

    def __init__(self, *args, **kwargs):
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(
            str(index) for index in range(INFERENCE_TP_SIZE)
        )
        super().__init__(*args, **kwargs)


@ray.remote(num_gpus=1)
class TrainModel:
    """Own the trainer model and sparse NCCL sender on one GPU."""

    def __init__(self, model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        ).to("cuda:0")
        self.model.eval()

        config = self.model.config
        if config.model_type != "qwen3_moe" or config.num_experts % INFERENCE_TP_SIZE:
            raise RuntimeError(
                "This recipe requires a Qwen3 MoE model whose experts divide "
                f"evenly across TP{INFERENCE_TP_SIZE}"
            )
        self.expert_ids = (0, config.num_experts // 2)
        self.expert_intermediate_size = config.moe_intermediate_size
        fused_name = f"model.layers.{PATCHED_LAYER}.mlp.experts.gate_up_proj"
        try:
            self.gate_up_proj = self.model.get_parameter(fused_name)
        except AttributeError as exc:
            raise RuntimeError(
                f"Expected trainer model to expose `{fused_name}`"
            ) from exc

        expected_shape = (
            config.num_experts,
            2 * self.expert_intermediate_size,
            config.hidden_size,
        )
        if self.gate_up_proj.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected fused expert shape: {self.gate_up_proj.shape} "
                f"!= {expected_shape}"
            )

        self.master_address = get_ip()
        self.master_port = get_open_port()
        self.engine = None

    def init_sparse_engine(self, world_size: int, llm_handle) -> None:
        self.engine = WeightTransferTrainerFactory.trainer_init(
            init_info=SparseNCCLTrainerInitInfo(
                master_address=self.master_address,
                master_port=self.master_port,
                world_size=world_size,
                rank=0,
            ),
            client=RayVLLMWeightSyncClient(llm_handle),
        )

    @torch.no_grad()
    def patch_and_send(self) -> tuple[list[str], int]:
        if self.engine is None:
            raise RuntimeError("Sparse NCCL engine is not initialized")

        patches = []
        for expert_id in self.expert_ids:
            checkpoint_weight = self.gate_up_proj[
                expert_id, : self.expert_intermediate_size
            ]
            original_rows = checkpoint_weight[:PATCHED_ROWS].clone()
            replacement_rows = original_rows.flip(0)
            if torch.equal(original_rows, replacement_rows):
                raise RuntimeError(f"Expert {expert_id} patch would be a no-op")
            checkpoint_weight[:PATCHED_ROWS] = replacement_rows

            hidden_size = checkpoint_weight.shape[1]
            flat_indices = torch.arange(
                PATCHED_ROWS * hidden_size,
                device=checkpoint_weight.device,
                dtype=torch.int32,
            )
            patches.append(
                SparseWeightPatch(
                    name=(
                        f"model.layers.{PATCHED_LAYER}.mlp.experts."
                        f"{expert_id}.gate_proj.weight"
                    ),
                    full_shape=tuple(checkpoint_weight.shape),
                    indices=flat_indices,
                    values=replacement_rows.reshape(-1).contiguous(),
                )
            )

        self.engine.send_weights(patches)
        return [patch.name for patch in patches], sum(
            patch.indices.numel() for patch in patches
        )

    def shutdown_engine(self) -> None:
        if self.engine is not None:
            self.engine.shutdown()
            self.engine = None


def launch_llm(scheduling_strategy: PlacementGroupSchedulingStrategy):
    return ray.remote(
        num_cpus=0,
        num_gpus=0,
        scheduling_strategy=scheduling_strategy,
    )(MyLLM).remote(
        model=MODEL_NAME,
        enforce_eager=True,
        tensor_parallel_size=INFERENCE_TP_SIZE,
        enable_expert_parallel=True,
        expert_placement_strategy="linear",
        moe_backend="triton",
        distributed_executor_backend="ray",
        gpu_memory_utilization=0.7,
        enable_prefix_caching=False,
        weight_transfer_config=WeightTransferConfig(backend="sparse_nccl"),
    )


def generate(llm_handle) -> list[dict[str, object]]:
    outputs = ray.get(llm_handle.generate.remote(PROMPTS, SAMPLING_PARAMS))
    return [
        {
            "token_ids": output.outputs[0].token_ids,
            "text": output.outputs[0].text,
        }
        for output in outputs
    ]


def print_generations(label: str, generations: list[dict[str, object]]) -> None:
    print(f"\n{label}")
    for prompt, generation in zip(PROMPTS, generations):
        print(
            f"  {prompt!r} -> {generation['text']!r} "
            f"(token_ids={generation['token_ids']})"
        )


def main() -> None:
    ray.init()
    train_model = None
    pg_inference = None
    llm = None
    try:
        train_model = TrainModel.remote(MODEL_NAME)
        pg_inference = placement_group(
            [{"GPU": 1, "CPU": 0}] * INFERENCE_TP_SIZE,
            strategy="STRICT_PACK",
        )
        ray.get(pg_inference.ready())
        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg_inference,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=0,
        )
        llm = launch_llm(scheduling_strategy)

        before = generate(llm)
        print_generations("BEFORE sparse update", before)

        ray.get(llm.sleep.remote(level=0))
        world_size = ray.get(llm.get_world_size.remote()) + 1
        ray.get(train_model.init_sparse_engine.remote(world_size, llm))
        patched_names, num_updates = ray.get(train_model.patch_and_send.remote())
        ray.get(llm.wake_up.remote(tags=["scheduling"]))

        after = generate(llm)
        print_generations("AFTER sparse update", after)
        print(f"patched_checkpoint_names={patched_names}")
        print(f"num_sparse_values={num_updates}")
        outputs_changed = any(
            old["token_ids"] != new["token_ids"]
            for old, new in zip(before, after, strict=True)
        )
        print(f"outputs_changed={outputs_changed}")
    finally:
        if train_model is not None:
            with suppress(Exception):
                ray.get(train_model.shutdown_engine.remote())
        if llm is not None:
            with suppress(Exception):
                ray.kill(llm)
        if train_model is not None:
            with suppress(Exception):
                ray.kill(train_model)
        if pg_inference is not None:
            with suppress(Exception):
                ray.util.remove_placement_group(pg_inference)
        ray.shutdown()


if __name__ == "__main__":
    main()
