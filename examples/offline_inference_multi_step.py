'''
Example of setting up LLM with multi-step enabled.
In actuality, async engine would be a more sensible choice
from a performance perspective. However this example is useful
for demonstration & debugging of multi-step code.
'''

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(
    model="JackFram/llama-160m",
    swap_space=16,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    num_scheduler_steps=8,
    use_v2_block_manager=True,
)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
