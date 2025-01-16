# see https://github.com/vllm-project/vllm/issues/11400 for
# the motivation and use case for this example.
# this is a specialized implementation of tensor-parallel inference.
# run the script with `torchrun --nproc-per-node=2 torchrun_example.py`,
# the argument 2 should match the `tensor_parallel_size` below.

import torch.distributed as dist

from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import get_world_group

dist.init_process_group(backend="nccl")

torch_rank = dist.get_rank()

# Create prompts
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# important: use `distributed_executor_backend="external_launcher"` so that
# this llm engine/instance only creates one worker.
# set different `gpu_memory_utilization` and `swap_space` for different ranks,
# to test if all ranks agree on the same kv cache configuration.
llm = LLM(model="facebook/opt-125m",
          tensor_parallel_size=2,
          distributed_executor_backend="external_launcher",
          gpu_memory_utilization=0.9 if torch_rank == 0 else 0.7,
          swap_space=3 if torch_rank == 0 else 4)

# important: prompts should be the same across all ranks
# important: scheduling decisions should be deterministic
# and should be the same across all ranks
outputs = llm.generate(prompts, sampling_params)

# it is recommended to use this `cpu_group` to communicate
# control messages across all ranks, to avoid interference
# with the model's device group communication.
cpu_group = get_world_group().cpu_group


def test_consistent_across_ranks(obj):
    if torch_rank == 0:
        dist.broadcast_object_list([obj], src=0, group=cpu_group)
    else:
        container = [None]
        dist.broadcast_object_list(container, src=0, group=cpu_group)
        assert container[0] == obj


test_consistent_across_ranks(
    llm.llm_engine.vllm_config.cache_config.num_cpu_blocks)
test_consistent_across_ranks(
    llm.llm_engine.vllm_config.cache_config.num_gpu_blocks)

# all ranks should have the same outputs
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    test_consistent_across_ranks(prompt)
    test_consistent_across_ranks(generated_text)
    print(f"Rank {torch_rank}, Prompt: {prompt!r}, "
          f"Generated text: {generated_text!r}")

# all ranks can access the model directly
# via `llm.llm_engine.model_executor.driver_worker.worker.model_runner.model`
