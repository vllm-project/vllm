import torch
import time
from vllm import LLM, SamplingParams

torch.random.manual_seed(999)

llm = LLM(model='/home/zhn/g/Meta-Llama-3-8B-Instruct', gpu_memory_utilization=0.5)
prompts = [
    "Hi my name is",
    "Tell me a joke",
]

texts = []
start = time.time()
for i in range(10):
    sampling_params = SamplingParams(temperature=0, top_k=1, max_tokens=200, top_p=1, repetition_penalty=0.9)
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        texts.append(generated_text)
end = time.time()
print(f"Time taken: {end - start:.2f}s")
