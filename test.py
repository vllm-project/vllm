from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
prompts = prompts * 64
sampling_params = SamplingParams(temperature=0.8, top_k=6, top_p=0.9)

llm = LLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct")
# llm = LLM(model="deepseek-ai/DeepSeek-V2-Lite")

outputs = llm.generate(prompts, sampling_params)

for i, output in enumerate(outputs):
    if i > 4:
        break
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    
