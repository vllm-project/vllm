# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Demonstrates reinforcement learning using vLLM and Ray,
with native weight syncing APIs at engine instance.

The script separates training and inference workloads onto distinct GPUs
so that Ray can manage process placement and inter-process communication.
A Hugging Face Transformer model occupies one GPU for training, whereas a
2x tensor-parallel vLLM inference engine occupies two GPUs.

The example performs the following steps:
* Load the training model on one gpu (scheduled via ray)
* Initialize the inference model with dummy weights across
  two gpus using vLLM's tensor parallelism and Ray placement groups.
* Generate gibberish from a list of prompts using the randomly initialized
  inference engine.
* Update the weights of the training model and broadcast the updated weights
  to the inference engine by using a Ray collective RPC group.
* Generating from the list of prompts after weight sync should result
  in sensible outputs.

This example assumes a single-node cluster with three GPUs, but Ray
supports multi-node clusters. vLLM expects the GPUs are only used for vLLM
workloads. Residual GPU activity interferes with vLLM memory profiling and
causes unexpected behavior.
"""

import os

import ray
import torch
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoModelForCausalLM

from vllm import LLM, SamplingParams
from vllm.config import NCCLWeightTransferConfig
from vllm.distributed.weight_transfer import (
    RayVLLMWeightSyncClient,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.nccl_common import NCCLTrainerInitInfo
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_ip, get_open_port

MODEL_NAME = "facebook/opt-125m"
# MODEL_NAME = "inference-optimization/Qwen3-0.6B-W4A16-G128"


def get_assigned_gpu():
    """This is a temporary workaround for a runtime bug in RCCL on ROCm."""
    if not current_platform.is_rocm():
        return 0
    assigned_gpu = int(ray.get_gpu_ids()[0])
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ.pop("HIP_VISIBLE_DEVICES", None)
    torch.accelerator.set_device_idx(assigned_gpu)
    return assigned_gpu


class MyLLM(LLM):
    """Configure the vLLM worker for Ray placement group execution."""

    def __init__(self, *args, **kwargs):
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = "0,1"
        super().__init__(*args, **kwargs)


@ray.remote(num_gpus=1)
class TrainModel:
    """Ray actor that wraps the training model on a dedicated GPU."""

    def __init__(self, model_name: str):
        assigned_gpu = get_assigned_gpu()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
        ).to(f"cuda:{assigned_gpu}")

        self.port = get_open_port()
        self.master_address = get_ip()

    def get_master_address_and_port(self):
        return self.master_address, self.port

    def init_weight_transfer(self, world_size, llm_handle):
        """Build the trainer-side weight-transfer engine.

        `trainer_init` drives the full handshake: it kicks off the inference
        side's `init_weight_transfer_engine` (via the Ray client) on a worker
        thread while opening the trainer's NCCL endpoint, so both ends
        rendezvous together. After this returns, `send_weights()` is callable.
        """
        self.engine = WeightTransferTrainerFactory.trainer_init(
            backend="nccl",
            config=NCCLWeightTransferConfig(packed=True),
            init_info=NCCLTrainerInitInfo(
                master_address=self.master_address,
                master_port=self.port,
                world_size=world_size,
            ),
            client=RayVLLMWeightSyncClient(llm_handle),
            weight_iterator=self.model.named_parameters,  # bound method = factory
        )

    def broadcast_weights(self):
        """Push the current weights to the inference engine.

        Drives start/update/finish on the inference side and the NCCL
        broadcast internally — one call.
        """
        self.engine.send_weights()


# Initialize Ray and set the visible devices. The vLLM engine will
# be placed on GPUs 1 and 2.
ray.init()

# Create a placement group that reserves GPU 1–2 for the vLLM inference engine.
# Learn more about Ray placement groups:
# https://docs.ray.io/en/latest/placement-groups.html
# Launch the training model actor. Ray's resource scheduler will allocate
# 1 GPU (via num_gpus=1 in the decorator), ensuring pg_inference gets different GPUs.
train_model = TrainModel.remote(MODEL_NAME)

pg_inference = placement_group([{"GPU": 1, "CPU": 0}] * 2)
ray.get(pg_inference.ready())
scheduling_inference = PlacementGroupSchedulingStrategy(
    placement_group=pg_inference,
    placement_group_capture_child_tasks=True,
    placement_group_bundle_index=0,
)

# Launch the vLLM inference engine. The `enforce_eager` flag reduces
# start-up latency.
# Note: Weight transfer APIs (init_weight_transfer_engine, update_weights)
# are now native to vLLM workers.
llm = ray.remote(
    num_cpus=0,
    num_gpus=0,
    scheduling_strategy=scheduling_inference,
)(MyLLM).remote(
    model=MODEL_NAME,
    enforce_eager=True,
    tensor_parallel_size=2,
    data_parallel_size=1,
    distributed_executor_backend="ray",
    weight_transfer_config=NCCLWeightTransferConfig(packed=True),
    load_format="dummy",
    quantization="fp8",
)

# Generate text from the prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(temperature=0)

outputs = ray.get(llm.generate.remote(prompts, sampling_params))

# Generate text with the initial model. The output is expected to be nonsense
# because the weights are randomly initialized.
print("-" * 50)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
    print("-" * 50)

ray.get(llm.sleep.remote(level=0))

# Set up the communication channel between the training process and the
# inference engine. The trainer engine now owns the full handshake — the
# driver only has to hand it the vLLM actor handle and the world size.
world_size = ray.get(llm.get_world_size.remote()) + 1  # +1 for the trainer
ray.get(train_model.init_weight_transfer.remote(world_size, llm))

# Synchronize the updated weights to the inference engine. The engine drives
# start_weight_update / update_weights / finish_weight_update on the inference
# side internally, concurrent with the NCCL broadcast.
ray.get(train_model.broadcast_weights.remote())

ray.get(llm.wake_up.remote(tags=["scheduling"]))

# Generate text with the updated model. The output is expected to be normal
# because the weights are updated.
outputs_updated = ray.get(llm.generate.remote(prompts, sampling_params))
print("-" * 50)
for output in outputs_updated:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
    print("-" * 50)
