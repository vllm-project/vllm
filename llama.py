from vllm import LLM, SamplingParams

llm = LLM(model='/home/largeniu/triton/llama3/Meta-Llama-3-8B-Instruct')
prompts = [
    "Hi my name is",
    # "The capital of France is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")