from vllm import LLM, SamplingParams
import torch.distributed as dist

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

llm = LLM(model="facebook/opt-125m",
          tensor_parallel_size=2,
          distributed_executor_backend="uni")

# Generate texts from the prompts.
# The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
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
    print(
        f"Rank {torch_rank}, Prompt: {prompt!r}, "
        f"Generated text: {generated_text!r}"
    )
