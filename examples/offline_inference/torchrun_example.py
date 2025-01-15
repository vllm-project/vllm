import torch.distributed as dist

from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import get_world_group

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
llm = LLM(model="facebook/opt-125m",
          tensor_parallel_size=2,
          distributed_executor_backend="external_launcher")

# important: prompts should be the same across all ranks
# important: scheduling decisions should be deterministic
# and should be the same across all ranks
outputs = llm.generate(prompts, sampling_params)

# it is recommended to use this `cpu_group` to communicate
# control messages across all ranks, to avoid interference
# with the model's device group communication.
cpu_group = get_world_group().cpu_group
torch_rank = get_world_group().rank

# all ranks should have the same outputs
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    if torch_rank == 0:
        dist.broadcast_object_list([prompt, generated_text],
                                   src=0,
                                   group=cpu_group)
    else:
        container = [None, None]
        dist.broadcast_object_list(container, src=0, group=cpu_group)
        assert container[0] == prompt
        assert container[1] == generated_text
    print(f"Rank {torch_rank}, Prompt: {prompt!r}, "
          f"Generated text: {generated_text!r}")

# all ranks can access the model directly
# via `llm.llm_engine.model_executor.driver_worker.worker.model_runner.model`
