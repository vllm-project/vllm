from vllm.entrypoints.llm import LLM, SamplingParams
import ray

ray.init(
    runtime_env={
        "env_vars": {"CUDA_LAUNCH_BLOCKING": "1"}
    }
)


prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model="facebook/opt-125m", tensor_parallel_size=2)

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")