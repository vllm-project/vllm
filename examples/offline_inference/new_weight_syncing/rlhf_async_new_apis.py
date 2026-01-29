# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Demonstrates async reinforcement learning using vLLM and Ray,
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
* Pause generation once generation completes for one sequence
* Update the weights of the training model and broadcast the updated weights
  to the inference engine by using a Ray collective RPC group.
* Resume generation and print out the results

This example assumes a single-node cluster with three GPUs, but Ray
supports multi-node clusters. vLLM expects the GPUs are only used for vLLM
workloads. Residual GPU activity interferes with vLLM memory profiling and
causes unexpected behavior.
"""

import os
import uuid
from dataclasses import asdict

import ray
import torch
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoModelForCausalLM, AutoTokenizer

import vllm
from vllm import SamplingParams
from vllm.config import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
)
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLWeightTransferEngine,
    NCCLWeightTransferInitInfo,
    NCCLWeightTransferUpdateInfo,
)
from vllm.utils.network_utils import get_ip, get_open_port
from vllm.v1.executor import Executor

MODEL_NAME = "facebook/opt-125m"


class MyLLM(vllm.AsyncLLMEngine):
    """Configure the vLLM worker for Ray placement group execution."""

    def __init__(self, **kwargs):
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = "0,1"
        engine_args = vllm.AsyncEngineArgs(**kwargs)
        vllm_config = engine_args.create_engine_config()
        executor_class = Executor.get_class(vllm_config)
        super().__init__(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_requests=engine_args.enable_log_requests,
            log_stats=not engine_args.disable_log_stats,
        )

    async def generate_with_retry(
        self, prompt_token_ids: list[int], sampling_params: vllm.SamplingParams
    ) -> vllm.RequestOutput:
        finish_reason = "abort"
        while finish_reason == "abort":
            async for request_output in self.generate(
                {"prompt_token_ids": prompt_token_ids},
                sampling_params,
                request_id=str(uuid.uuid4()),
            ):
                output = request_output
            finish_reason = output.outputs[0].finish_reason
            if finish_reason == "abort":
                print(
                    f"ABORT, prompt_token_ids: {prompt_token_ids}, "
                    f"generated token_ids: {list(output.outputs[0].token_ids)}"
                )
            prompt_token_ids = prompt_token_ids + list(output.outputs[0].token_ids)
        return output


@ray.remote(num_gpus=1)
class TrainModel:
    """Ray actor that wraps the training model on a dedicated GPU."""

    def __init__(self, model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16
        ).to("cuda:0")
        self.port = get_open_port()
        self.master_address = get_ip()

    def get_master_address_and_port(self):
        return self.master_address, self.port

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

    def init_weight_transfer_group(self, world_size):
        """Initialize the NCCL process group for weight transfer."""
        self.model_update_group = NCCLWeightTransferEngine.trainer_init(
            dict(
                master_address=self.master_address,
                master_port=self.port,
                world_size=world_size,
            ),
        )

    def broadcast_weights(self, packed: bool = True):
        """Broadcast weights to the inference engine."""
        NCCLWeightTransferEngine.trainer_send_weights(
            iterator=self.model.named_parameters(),
            group=self.model_update_group,
            packed=packed,
        )


# Initialize Ray and set the visible devices. The vLLM engine will
# be placed on GPUs 1 and 2.
ray.init()

# Launch the training model actor. Ray's resource scheduler will allocate
# 1 GPU (via num_gpus=1 in the decorator), ensuring pg_inference gets different GPUs.
train_model = TrainModel.remote(MODEL_NAME)

# Create a placement group that reserves GPU 1–2 for the vLLM inference engine.
# Learn more about Ray placement groups:
# https://docs.ray.io/en/latest/placement-groups.html

pg_inference = placement_group([{"GPU": 1, "CPU": 0}] * 2)
ray.get(pg_inference.ready())
scheduling_inference = PlacementGroupSchedulingStrategy(
    placement_group=pg_inference,
    placement_group_capture_child_tasks=True,
    placement_group_bundle_index=0,
)

# Launch the vLLM inference engine. The `enforce_eager` flag reduces
# start-up latency.
# Note: Weight transfer APIs (init_weight_transfer_engine, update_weights,
# finalize_weight_update) are now native to vLLM workers.
llm = ray.remote(
    num_cpus=0,
    num_gpus=0,
    scheduling_strategy=scheduling_inference,
)(MyLLM).remote(
    model=MODEL_NAME,
    enforce_eager=True,
    tensor_parallel_size=2,
    distributed_executor_backend="ray",
    load_format="dummy",
    weight_transfer_config=WeightTransferConfig(backend="nccl"),
)

# Generate text from the prompts.
prompts = [
    "My name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Tokenize prompts to token IDs
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
prompt_token_ids_list = [
    tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts
]

sampling_params = [
    SamplingParams(temperature=0, max_tokens=2),
    SamplingParams(temperature=0, max_tokens=32),
    SamplingParams(temperature=0, max_tokens=32),
    SamplingParams(temperature=0, max_tokens=32),
]

# Set up the communication channel between the training process and the
# inference engine.
master_address, master_port = ray.get(train_model.get_master_address_and_port.remote())

world_size = 3  # 1 trainer + 2 inference workers (tensor_parallel_size=2)
inference_handle = llm.init_weight_transfer_engine.remote(
    WeightTransferInitRequest(
        init_info=asdict(
            NCCLWeightTransferInitInfo(
                master_address=master_address,
                master_port=master_port,
                rank_offset=1,
                world_size=world_size,
            )
        )
    )
)

# Initialize weight transfer group on both the training actor and inference engine
train_handle = train_model.init_weight_transfer_group.remote(world_size)
ray.get([train_handle, inference_handle])


generation_futures = [
    llm.generate_with_retry.remote(prompt_token_ids, params)
    for prompt_token_ids, params in zip(prompt_token_ids_list, sampling_params)
]

finished, pending = ray.wait(generation_futures, num_returns=1)

# Pause generation in preparation for weight sync
ray.get(llm.pause_generation.remote(wait_for_inflight_requests=False))

# Synchronize the updated weights to the inference engine using batched API.
# Collect all weight metadata from the training actor
names, dtype_names, shapes = ray.get(train_model.get_weight_metadata.remote())

# Issue update_weights call with NCCL-specific update info
# packed=True enables efficient batched tensor broadcasting
inference_handle = llm.update_weights.remote(
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

# Broadcast all weights from trainer using the weight transfer API
train_handle = train_model.broadcast_weights.remote(packed=True)
ray.get([train_handle, inference_handle])

# Finalize the weight update (processes weights for quantization/kernel format)
ray.get(llm.finalize_weight_update.remote())

# Resume generation since weight sync is complete
ray.get(llm.resume_generation.remote())

# Get outputs separately - finished completed before pause, pending were paused/resumed
finished_outputs = ray.get(finished)
pending_outputs = ray.get(pending)

# Requests that finished before the pause: all generation used original weights
print("-" * 50)
print("Requests that completed BEFORE weight change:")
print("-" * 50)
for output in finished_outputs:
    prompt_text = tokenizer.decode(output.prompt_token_ids)
    print(f"Prompt: {prompt_text!r}")
    print(f"Generated (with original weights): {output.outputs[0].text!r}")
    print("-" * 50)

# Requests that were paused mid-generation: some text before, some after weight change
print("Requests that were PAUSED and RESUMED after weight change:")
print("-" * 50)
for output in pending_outputs:
    # Decode the full prompt token IDs (original + generated before pause)
    full_prompt_text = tokenizer.decode(output.prompt_token_ids)
    # Find the original prompt by checking which one this output started with
    original_prompt = next(p for p in prompts if full_prompt_text.startswith(p))
    # output.prompt_token_ids contains original prompt + tokens generated before pause
    # output.outputs[0].text is what was generated after resuming with new weights
    text_before_pause = full_prompt_text[len(original_prompt) :]
    text_after_pause = output.outputs[0].text
    print(f"Original prompt: {original_prompt!r}")
    print(f"Generated before weight change: {text_before_pause!r}")
    print(f"Generated after weight change: {text_after_pause!r}")
    print("-" * 50)
