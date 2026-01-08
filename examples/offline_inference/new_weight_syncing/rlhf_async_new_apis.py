# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Demonstrates asynchronous reinforcement learning from human feedback (RLHF)
using vLLM and Ray, with the new weight syncing APIs

The script separates training and inference workloads onto distinct GPUs
so that Ray can manage process placement and inter-process communication.
A Hugging Face Transformer model occupies GPU 0 for training, whereas a
tensor-parallel vLLM inference engine occupies GPU 1–2.

The example performs the following steps:

* Load the training model on GPU 0.
* Split the inference model across GPUs 1–2 using vLLM's tensor parallelism
  and Ray placement groups.
* Start generation from a list of prompts using the inference engine.
* Pause generation once generation completes for one sequence
* Update the weights of the training model and broadcast the updated weights
  to the inference engine by using a Ray collective RPC group. Note that
  for demonstration purposes we simply zero out the weights.
* Resume generation and print out the results

This example assumes a single-node cluster with three GPUs, but Ray
supports multi-node clusters. vLLM expects the GPUs are only used for vLLM
workloads. Residual GPU activity interferes with vLLM memory profiling and
causes unexpected behavior.
"""

import asyncio
import os
import uuid

import ray
import torch
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from rlhf_utils import stateless_init_process_group
from transformers import AutoModelForCausalLM

import vllm
from vllm import SamplingParams
from vllm.config import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest,
    WeightUpdateRequest,
)
from vllm.utils.network_utils import get_ip, get_open_port


class MyLLM:
    """Simple wrapper over AsyncLLM for supporting async RL."""

    def __init__(self, **kwargs):
        self.engine = vllm.AsyncLLMEngine.from_engine_args(
            vllm.AsyncEngineArgs(**kwargs)
        )
        self.generation_paused_event = asyncio.Event()

    async def generate(
        self, prompt: str, sampling_params: vllm.SamplingParams
    ) -> vllm.RequestOutput:
        async for request_output in self.engine.generate(
            prompt, sampling_params, request_id=str(uuid.uuid4())
        ):
            final_output = request_output
        return final_output

    async def generate_with_retry(
        self, prompt: str, sampling_params: vllm.SamplingParams
    ) -> vllm.RequestOutput:
        finish_reason = "abort"
        while finish_reason == "abort":
            await self._wait_for_generation_to_resume()
            output = await self.generate(prompt, sampling_params)
            finish_reason = output.outputs[0].finish_reason
            if finish_reason == "abort":
                print(f"REQ ABORTED, prompt: {prompt}, text: {output.outputs[0].text}")
            prompt += output.outputs[0].text
        return output

    async def abort_generation(self) -> None:
        self.generation_paused_event.set()
        unfinished_request_ids = list(
            self.engine.output_processor.request_states.keys()
        )
        if unfinished_request_ids:
            await self.engine.abort(unfinished_request_ids)
        await self.engine.reset_prefix_cache()
        print(
            f"abort_generation() finished, aborted"
            f"{len(unfinished_request_ids)} requests"
        )

    async def resume_generation(self) -> None:
        self.generation_paused_event.clear()

    async def collective_rpc(self, method: str, args: tuple = ()):
        return await self.engine.collective_rpc(method, args=args)

    async def _wait_for_generation_to_resume(self) -> None:
        """Waits for generation to be resumed, intended for in-flight weight updates
        and partial rollouts."""
        while self.generation_paused_event.is_set():
            await asyncio.sleep(0.5)

    async def init_weight_transfer(self, request: WeightTransferInitRequest) -> None:
        print("reached init weight transfer")
        return await self.engine.init_weight_transfer(request)

    async def update_weights(self, request: WeightUpdateRequest) -> None:
        return await self.engine.update_weights(request)

    async def finalize_weight_update(self) -> None:
        return await self.engine.finalize_weight_update()


# Load the OPT-125M model onto GPU 0 for the training workload.
train_model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-125m", dtype=torch.bfloat16
)
train_model.to("cuda:0")

# Initialize Ray and set the visible devices. The vLLM engine will
# be placed on GPUs 1 and 2.
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
ray.init(runtime_env={"excludes": [".git/objects/pack/"]})
# ray.init()

# Create a placement group that reserves GPU 1–2 for the vLLM inference engine.
# Learn more about Ray placement groups:
# https://docs.ray.io/en/latest/placement-groups.html
pg_training = placement_group([{"GPU": 1, "CPU": 0}])
ray.get(pg_training.ready())

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
    model="facebook/opt-125m",
    enforce_eager=True,
    tensor_parallel_size=2,
    distributed_executor_backend="ray",
    weight_transfer_config=WeightTransferConfig(backend="nccl"),
)

# Generate text from the prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(temperature=0)

# Set up the communication channel between the training process and the
# inference engine.
master_address = get_ip()
master_port = get_open_port()

print("reached init weight in driver")
handle = llm.init_weight_transfer.remote(
    WeightTransferInitRequest(
        init_info=dict(
            master_address=master_address,
            master_port=master_port,
            rank_offset=1,
            world_size=3,
        )
    )
)

model_update_group = stateless_init_process_group(
    master_address, master_port, 0, 3, torch.device("cuda:0")
)
ray.get(handle)


generation_futures = [
    llm.generate_with_retry.remote(prompt, sampling_params) for prompt in prompts
]

finished, pending = ray.wait(generation_futures, num_returns=1)

# Abort generation in preparation for weight sync
ray.get(llm.abort_generation.remote())

# Simulate a training step by zeroing out all model weights.
# In a real RLHF training loop the weights would be updated using the gradient
# from an RL objective such as PPO on a reward model.
for name, p in train_model.named_parameters():
    p.data.zero_()

# Synchronize the updated weights to the inference engine using batched API.
# Collect all weight metadata
names = []
dtype_names = []
shapes = []
for name, p in train_model.named_parameters():
    names.append(name)
    dtype_names.append(str(p.dtype).split(".")[-1])
    shapes.append(p.shape)

# Issue update_weights call
handle = llm.update_weights.remote(
    WeightUpdateRequest(
        update_info=dict(names=names, dtype_names=dtype_names, shapes=shapes)
    )
)

# Broadcast all weights from trainer
for name, p in train_model.named_parameters():
    model_update_group.broadcast(p, src=0, stream=torch.cuda.current_stream())

ray.get(handle)

# Finalize the weight update (processes weights for quantization/kernel format)
ray.get(llm.finalize_weight_update.remote())

# Resume generation since weight sync is complete
ray.get(llm.resume_generation.remote())

# Get all outputs
outputs = ray.get(finished) + ray.get(pending)

# We expect the first output to be normal generation.
# The other outputs should have generated regular results midway
# and then have garbage tokens because we zero'd out the weights
print("-" * 50)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
    print("-" * 50)
