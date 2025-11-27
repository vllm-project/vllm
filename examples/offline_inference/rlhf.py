# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Demonstrates reinforcement learning from human feedback (RLHF) using vLLM and Ray.

The script separates training and inference workloads onto distinct GPUs
so that Ray can manage process placement and inter-process communication.
A Hugging Face Transformer model occupies GPU 0 for training, whereas a
tensor-parallel vLLM inference engine occupies GPU 1–2.

The example performs the following steps:

* Load the training model on GPU 0.
* Split the inference model across GPUs 1–2 using vLLM's tensor parallelism
  and Ray placement groups.
* Generate text from a list of prompts using the inference engine.
* Update the weights of the training model and broadcast the updated weights
  to the inference engine by using a Ray collective RPC group. Note that
  for demonstration purposes we simply zero out the weights.

For a production-ready implementation that supports multiple training and
inference replicas, see the OpenRLHF framework:
https://github.com/OpenRLHF/OpenRLHF

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
from rlhf_utils import stateless_init_process_group
from transformers import AutoModelForCausalLM

from vllm import LLM, SamplingParams
from vllm.utils.network_utils import get_ip, get_open_port

from vllm.platforms import current_platform

DEVICE = current_platform.device_type

class MyLLM(LLM):
    """Configure the vLLM worker for Ray placement group execution."""

    def __init__(self, *args, **kwargs):
        # Remove the top-level CUDA_VISIBLE_DEVICES variable set by Ray
        # so that vLLM can manage its own device placement within the worker.
        if current_platform.is_xpu():
            os.environ.pop("ZE_AFFINITY_MASK", None)
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        super().__init__(*args, **kwargs)


# Load the OPT-125M model onto GPU 0 for the training workload.
train_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
train_model.to(f"{DEVICE}:0")

# Initialize Ray and set the visible devices. The vLLM engine will
# be placed on GPUs 1 and 2.
if current_platform.is_xpu():
    os.environ["ZE_AFFINITY_MASK"] = "1,2"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
ray.init()

# Create a placement group that reserves GPU 1–2 for the vLLM inference engine.
# Learn more about Ray placement groups:
# https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html
pg_inference = placement_group([{"GPU": 1, "CPU": 0}] * 2)
ray.get(pg_inference.ready())
scheduling_inference = PlacementGroupSchedulingStrategy(
    placement_group=pg_inference,
    placement_group_capture_child_tasks=True,
    placement_group_bundle_index=0,
)

# Launch the vLLM inference engine. The `enforce_eager` flag reduces
# start-up latency.
llm = ray.remote(
    num_cpus=0,
    num_gpus=0,
    scheduling_strategy=scheduling_inference,
)(MyLLM).remote(
    model="facebook/opt-125m",
    enforce_eager=True,
    worker_extension_cls="rlhf_utils.WorkerExtension",
    tensor_parallel_size=2,
    distributed_executor_backend="ray",
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

handle = llm.collective_rpc.remote(
    "init_weight_update_group", args=(master_address, master_port, 1, 3)
)

model_update_group = stateless_init_process_group(
    master_address, master_port, 0, 3, torch.device(f"{DEVICE}:0")
)
ray.get(handle)

# Simulate a training step by zeroing out all model weights.
# In a real RLHF training loop the weights would be updated using the gradient
# from an RL objective such as PPO on a reward model.
for name, p in train_model.named_parameters():
    p.data.zero_()

# Synchronize the updated weights to the inference engine.
for name, p in train_model.named_parameters():
    dtype_name = str(p.dtype).split(".")[-1]
    handle = llm.collective_rpc.remote(
        "update_weight", args=(name, dtype_name, p.shape)
    )
    if current_platform.is_xpu():
        model_update_group.broadcast(p, src=0, stream=torch.xpu.current_stream())
    else:
        model_update_group.broadcast(p, src=0, stream=torch.cuda.current_stream())
    ray.get(handle)

# Verify that the inference weights have been updated.
assert all(ray.get(llm.collective_rpc.remote("check_weights_changed")))

# Generate text with the updated model. The output is expected to be nonsense
# because the weights are zero.
outputs_updated = ray.get(llm.generate.remote(prompts, sampling_params))
print("-" * 50)
for output in outputs_updated:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
    print("-" * 50)
