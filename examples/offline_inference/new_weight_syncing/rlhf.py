# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Demonstrates reinforcement learning from human feedback (RLHF) using vLLM and Ray,
with native weight syncing APIs at engine instance.

The script separates training and inference workloads onto distinct GPUs
so that Ray can manage process placement and inter-process communication.
A Hugging Face Transformer model occupies GPU 0 for training, whereas a
tensor-parallel vLLM inference engine occupies GPU 1–2.

The example performs the following steps:

* Load the training model on GPU 0.
* Split the inference model across GPUs 1–2 using vLLM's tensor parallelism
  and Ray placement groups with dummy weights.
* Generate text from a list of prompts using the inference engine.
* Update the weights of the training model and broadcast the updated weights
  to the inference engine by using a Ray collective RPC group.

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
from vllm.config import WeightTransferConfig
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLWeightTransferEngine,
)
from vllm.utils.network_utils import get_ip, get_open_port

MODEL_NAME = "facebook/opt-125m"


class MyLLM(LLM):
    """Configure the vLLM worker for Ray placement group execution."""

    def __init__(self, *args, **kwargs):
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = "0,1"
        super().__init__(*args, **kwargs)


@ray.remote(num_gpus=1)
class TrainModel:
    """Ray actor that wraps the training model on a dedicated GPU."""

    def __init__(self, model_name: str):
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16
        )
        self.model.to(self.device)

    def get_weight_metadata(self):
        """Return weight names, dtypes, and shapes for weight transfer."""
        names = []
        dtype_names = []
        shapes = []
        for name, p in self.model.named_parameters():
            names.append(name)
            dtype_names.append(str(p.dtype).split(".")[-1])
            shapes.append(list(p.shape))
        return names, dtype_names, shapes

    def init_weight_transfer_group(self, master_address, master_port, world_size):
        """Initialize the NCCL process group for weight transfer."""
        self.model_update_group = NCCLWeightTransferEngine.stateless_init_process_group(
            master_address, master_port, 0, world_size, self.device
        )

    def broadcast_weights(self, packed: bool = True):
        """Broadcast weights to the inference engine."""
        NCCLWeightTransferEngine.trainer_broadcast_weights(
            iterator=self.model.named_parameters(),
            group=self.model_update_group,
            packed=packed,
        )


# Initialize Ray and set the visible devices. The vLLM engine will
# be placed on GPUs 1 and 2.
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
ray.init(runtime_env={"excludes": [".git/objects/pack/"]})
# ray.init()

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
# Note: Weight transfer APIs (init_weight_transfer, update_weights,
# finalize_weight_update) are now native to vLLM workers.
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
    weight_transfer_config=WeightTransferConfig(backend="nccl"),
    load_format="dummy",
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

# Set up the communication channel between the training process and the
# inference engine.
master_address = get_ip()
master_port = get_open_port()

world_size = ray.get(llm.get_world_size.remote()) + 1  # +1 for the trainer
inference_handle = llm.init_weight_transfer.remote(
    dict(
        init_info=dict(
            master_address=master_address,
            master_port=master_port,
            rank_offset=1,
            world_size=world_size,
        )
    )
)

# Initialize weight transfer group on both the training actor and inference engine
train_handle = train_model.init_weight_transfer_group.remote(
    master_address, master_port, world_size
)
ray.get([train_handle, inference_handle])

# Synchronize the updated weights to the inference engine using batched API.
# Collect all weight metadata from the training actor
names, dtype_names, shapes = ray.get(train_model.get_weight_metadata.remote())

# Issue update_weights call with NCCL-specific update info
# packed=True enables efficient batched tensor broadcasting
inference_handle = llm.update_weights.remote(
    dict(
        update_info=dict(
            names=names,
            dtype_names=dtype_names,
            shapes=shapes,
            packed=True,
        )
    )
)

# Broadcast all weights from trainer using the weight transfer API
train_handle = train_model.broadcast_weights.remote(packed=True)
ray.get([train_handle, inference_handle])

# Finalize the weight update
ray.get(llm.finalize_weight_update.remote())

# Generate text with the updated model. The output is expected to be normal
# because the weights are updated.
outputs_updated = ray.get(llm.generate.remote(prompts, sampling_params))
print("-" * 50)
for output in outputs_updated:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
    print("-" * 50)
