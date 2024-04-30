from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="/models/llama2-70b-chat",
    tensor_parallel_size=8,
    gpu_memory_utilization=0.85,
    disable_custom_all_reduce=True,
)

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
