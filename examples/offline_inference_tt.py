import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tt_metal.models.demos.t3000.llama2_70b.tt.llama_generation import TtLlamaModelForGeneration

from vllm import LLM, SamplingParams
from vllm import ModelRegistry
ModelRegistry.register_model("TTLlamaForCausalLM", TtLlamaModelForGeneration)

# Sample prompts.
prompts = [
    "List the first 5 prime numbers",
    "Give a brief history of the internet",
    "Describe to me some good coding practices",
    "How many countries are in Africa?",
    "what is the capital of USA?",
    "what is the capital of Canada?",
    "what is the capital of UK?",
    "what is the capital of Germany?",
    "what is the capital of France?",
    "what is the capital of Japan?",
    "what is the capital of India?",
    "what is the capital of China?",
    "what is the currency of Cuba?",
    "what is the currency of Lebanon?",
    "what is the currency of Brazil?",
    "what is the currency of Australia?",
] * 2  # [32, 8] tokens

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32)

# Create an LLM.
llm = LLM(model="meta-llama/Meta-Llama-3.1-70B", block_size=64)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
