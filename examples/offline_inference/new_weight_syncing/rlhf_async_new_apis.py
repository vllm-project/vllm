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

import asyncio
import os
import time
import uuid
from dataclasses import asdict

import ray
import torch
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

MODEL_NAME_V1 = "Qwen/Qwen1.5-MoE-A2.7B"
MODEL_NAME_V2 = "Qwen/Qwen1.5-MoE-A2.7B-Chat"


class MyLLM(vllm.AsyncLLMEngine):
    """Configure the vLLM worker for Ray placement group execution."""

    def __init__(self, **kwargs):
        # This actor runs with num_gpus=0, so Ray sets CUDA_VISIBLE_DEVICES=""
        # to hide all GPUs. Remove it so it doesn't propagate (via runtime_env
        # inheritance) to the DP engine core actors and their RayWorkerWrapper
        # children, which need full GPU visibility.
        if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
            del os.environ["CUDA_VISIBLE_DEVICES"]
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
        engine_args = vllm.AsyncEngineArgs(**kwargs)
        vllm_config = engine_args.create_engine_config()
        executor_class = Executor.get_class(vllm_config)
        super().__init__(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_requests=engine_args.enable_log_requests,
            log_stats=not engine_args.disable_log_stats,
        )
        self._generation_paused = False

    async def pause_generation(self, **kwargs):
        await super().pause_generation(**kwargs)
        # Set after super() completes so the flag is only visible to
        # do_generate coroutines once the engine is actually paused.

        # ensure that all tokens are flushed
        await asyncio.sleep(0.2)
        self._generation_paused = True

    async def do_generate(
        self, prompt_token_ids: list[int], sampling_params: vllm.SamplingParams
    ) -> tuple[vllm.RequestOutput, int]:
        """Generate and return (output, pause_token_index).

        pause_token_index is the number of tokens generated before the
        weight change, or -1 if the request completed before any pause.
        """
        pause_token_index = -1
        prev_token_count = 0
        async for request_output in self.generate(
            {"prompt_token_ids": prompt_token_ids},
            sampling_params,
            request_id=str(uuid.uuid4()),
        ):
            output = request_output
            cur_token_count = len(output.outputs[0].token_ids)
            if self._generation_paused and pause_token_index == -1:
                # First yield after resume — the boundary is the previous
                # token count (last output before the generator blocked).
                pause_token_index = prev_token_count
            prev_token_count = cur_token_count
        return output, pause_token_index


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
train_model = TrainModel.remote(MODEL_NAME_V2)

# Launch the vLLM inference engine. The `enforce_eager` flag reduces
# start-up latency.
# With data_parallel_backend="ray", vLLM's CoreEngineActorManager creates
# its own placement groups internally for each DP rank, so we must NOT
# create an outer placement group (it would reserve GPUs and hide them
# from the internal DP resource check).
llm = ray.remote(
    num_cpus=0,
    num_gpus=0,
)(MyLLM).remote(
    model=MODEL_NAME_V1,
    enforce_eager=True,
    tensor_parallel_size=1,
    data_parallel_size=2,
    distributed_executor_backend="ray",
    data_parallel_backend="ray",
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
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_V1)
prompt_token_ids_list = [
    tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts
]

sampling_params = [
    SamplingParams(temperature=0, max_tokens=8),
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
    llm.do_generate.remote(prompt_token_ids, params)
    for prompt_token_ids, params in zip(prompt_token_ids_list, sampling_params)
]

finished, pending = ray.wait(generation_futures, num_returns=1)

# Pause generation in preparation for weight sync.
# mode="keep" preserves inflight requests so they resume after the pause
# (no need for retry logic).
ray.get(llm.pause_generation.remote(mode="keep"))

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

# Resume generation since weight sync is complete
ray.get(llm.resume_generation.remote())

# Get all outputs. With mode="keep", paused requests resume and complete normally.
# Each result is (RequestOutput, pause_token_index).
finished_results = ray.get(finished)
pending_results = ray.get(pending)

print("-" * 50)
print("Requests that completed BEFORE weight change:")
print("-" * 50)
for output, pause_idx in finished_results:
    prompt_text = tokenizer.decode(output.prompt_token_ids)
    print(f"Prompt: {prompt_text!r}")
    print(f"Generated (all old weights): {output.outputs[0].text!r}")
    print("-" * 50)

print("Requests that completed AFTER weight change (kept across pause/resume):")
print("-" * 50)
validation_cases = []
for output, pause_idx in pending_results:
    prompt_text = tokenizer.decode(output.prompt_token_ids)
    all_token_ids = list(output.outputs[0].token_ids)
    if pause_idx >= 0:
        before_text = tokenizer.decode(all_token_ids[:pause_idx])
        after_text = tokenizer.decode(all_token_ids[pause_idx:])
        print(f"Prompt: {prompt_text!r}")
        print(f"  Old weights ({pause_idx} tokens): {before_text!r}")
        n_after = len(all_token_ids) - pause_idx
        print(f"  New weights ({n_after} tokens): {after_text!r}")
        validation_cases.append(
            {
                "prompt_text": prompt_text,
                "context": list(output.prompt_token_ids) + all_token_ids[:pause_idx],
                "expected": all_token_ids[pause_idx:],
            }
        )
    else:
        print(f"Prompt: {prompt_text!r}")
        print(f"  Generated (all old weights): {output.outputs[0].text!r}")
    print("-" * 50)

# ── Validation ──────────────────────────────────────────────────────
# Shut down the weight-synced engine and start a fresh one loaded
# directly with V2 weights.  This gives a ground-truth comparison
# using the exact same vLLM engine
if validation_cases:
    print()
    print("=" * 50)
    print("VALIDATION: restarting vLLM with V2 weights for comparison")
    print("=" * 50)

    # Graceful shutdown removes the DP placement groups created by
    # CoreEngineActorManager.  A hard ray.kill() would skip cleanup and
    # leave stale node:IP_group_* resources that block the next engine.
    ray.get(llm.shutdown.remote())
    ray.kill(llm)
    time.sleep(5)  # allow Ray to reclaim placement group resources

    llm_v2 = ray.remote(
        num_cpus=0,
        num_gpus=0,
    )(MyLLM).remote(
        model=MODEL_NAME_V2,
        enforce_eager=True,
        tensor_parallel_size=1,
        data_parallel_size=2,
        distributed_executor_backend="ray",
        data_parallel_backend="ray",
    )

    # Send each context to the fresh engine with the same number of
    # tokens we need to validate.
    val_futures = []
    for case in validation_cases:
        n_tokens = len(case["expected"])
        val_futures.append(
            llm_v2.do_generate.remote(
                case["context"],
                SamplingParams(temperature=0, max_tokens=n_tokens),
            )
        )

    val_results = ray.get(val_futures)

    all_passed = True
    for case, (val_output, _) in zip(validation_cases, val_results):
        actual = list(val_output.outputs[0].token_ids)
        match = actual == case["expected"]
        status = "PASS" if match else "FAIL"
        if not match:
            all_passed = False
        print(f"  [{status}] {case['prompt_text']!r}")
        if not match:
            expected_text = tokenizer.decode(case["expected"])
            actual_text = tokenizer.decode(actual)
            print(f"         weight-synced engine: {expected_text!r}")
            print(f"         fresh V2 engine:      {actual_text!r}")
    print("-" * 50)
    print(f"Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print("=" * 50)

    ray.get(llm_v2.shutdown.remote())
    ray.kill(llm_v2)
