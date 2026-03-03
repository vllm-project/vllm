# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# unit test for `examples/offline_inference/torchrun_example.py`
import os
import random

import torch.distributed as dist

from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import get_tp_group, get_world_group

dist.init_process_group(backend="gloo")

# Create prompts
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
] * 10
dp_size = int(os.getenv("DP_SIZE", "1"))
dp_rank = int(os.getenv("DP_RANK", "0"))

if dp_size > 1:
    # distribute the prompts across the data parallel ranks
    prompts = [prompt for idx, prompt in enumerate(prompts) if idx % dp_size == dp_rank]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# set different `gpu_memory_utilization` and `swap_space` for different ranks,
# to test if all ranks agree on the same kv cache configuration.
llm = LLM(
    model="microsoft/Phi-mini-MoE-instruct",
    tensor_parallel_size=int(os.getenv("TP_SIZE", "1")),
    pipeline_parallel_size=int(os.getenv("PP_SIZE", "1")),
    enable_expert_parallel=int(os.getenv("ENABLE_EP", "0")) == 1,
    distributed_executor_backend="external_launcher",
    gpu_memory_utilization=random.uniform(0.7, 0.9),
    swap_space=random.randint(1, 4),
    seed=0,
)

outputs = llm.generate(prompts, sampling_params)

group = get_world_group() if dp_size == 1 else get_tp_group()
cpu_group = group.cpu_group
group_rank = dist.get_rank(group=cpu_group)


def test_consistent_across_ranks(obj):
    if group_rank == 0:
        dist.broadcast_object_list([obj], src=group.ranks[0], group=cpu_group)
    else:
        container = [None]
        dist.broadcast_object_list(container, src=group.ranks[0], group=cpu_group)
        assert container[0] == obj


test_consistent_across_ranks(llm.llm_engine.vllm_config.cache_config.num_cpu_blocks)
test_consistent_across_ranks(llm.llm_engine.vllm_config.cache_config.num_gpu_blocks)

# make sure we can access the model parameters from the calling process
# of the `LLM` instance.
params = list(
    llm.llm_engine.model_executor.driver_worker.worker.model_runner.model.parameters()
)
test_consistent_across_ranks(len(params))

# all ranks should have the same outputs
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    test_consistent_across_ranks(prompt)
    test_consistent_across_ranks(generated_text)
    print(f"Rank {group_rank}, Prompt: {prompt!r}, Generated text: {generated_text!r}")
