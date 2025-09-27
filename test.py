import os
from vllm import LLM, SamplingParams

# For V1: Turn off multiprocessing to make scheduling deterministic
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

# Set a fixed seed for reproducibility
SEED = 42

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
prompts = prompts * 64

# Add seed to sampling parameters for deterministic sampling
sampling_params = SamplingParams(temperature=0.8, top_p=0.7, seed=SEED)

# Add seed to LLM initialization for global reproducibility
# llm = LLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct", seed=SEED)
llm = LLM(model="deepseek-ai/DeepSeek-V2-Lite")

outputs = llm.generate(prompts, sampling_params)

for i, output in enumerate(outputs):
    if i > 4:
        break
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    
