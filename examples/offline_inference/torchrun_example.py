# experimental example for tensor-parallel inference with torchrun,
# see https://github.com/vllm-project/vllm/issues/11400 for
# the motivation and use case for this example.
# run the script with `torchrun --nproc-per-node=2 torchrun_example.py`,
# the argument 2 should match the `tensor_parallel_size` below.
# see `tests/distributed/test_torchrun_example.py` for the unit test.

from vllm import LLM, SamplingParams

# Create prompts, the same across all ranks
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create sampling parameters, the same across all ranks
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Use `distributed_executor_backend="external_launcher"` so that
# this llm engine/instance only creates one worker.
llm = LLM(model="facebook/opt-125m",
          tensor_parallel_size=2,
          distributed_executor_backend="external_launcher",
        )

outputs = llm.generate(prompts, sampling_params)

# all ranks will have the same outputs
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, "
          f"Generated text: {generated_text!r}")
