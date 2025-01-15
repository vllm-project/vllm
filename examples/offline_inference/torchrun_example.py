import torch.distributed as dist

from vllm import LLM, SamplingParams

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

# important: use `distributed_executor_backend="uni"` so that
# this llm engine/instance only creates one worker.
llm = LLM(model="facebook/opt-125m",
          tensor_parallel_size=2,
          distributed_executor_backend="uni")

# important: prompts should be the same across all ranks
# important: scheduling decisions should be deterministic
# and should be the same across all ranks
outputs = llm.generate(prompts, sampling_params)

# all ranks should have the same outputs
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    if torch_rank == 0:
        dist.broadcast_object_list([prompt, generated_text], src=0)
    else:
        container = [None, None]
        dist.broadcast_object_list(container, src=0)
        assert container[0] == prompt
        assert container[1] == generated_text
    print(f"Rank {torch_rank}, Prompt: {prompt!r}, "
          f"Generated text: {generated_text!r}")

# all ranks can access the model directly
# via `llm.llm_engine.model_executor.driver_worker.worker.model_runner.model`
