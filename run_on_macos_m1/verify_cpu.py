"""
vLLM CPU Verification Script for macOS Apple Silicon
This script tests whether the vLLM CPU backend was successfully compiled and works correctly.
"""

from vllm import LLM, SamplingParams
import time

# We use Qwen2-0.5B-Instruct as a lightweight test model.
# The model will be automatically downloaded from HuggingFace if not already cached.
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Note: Although we are using the CPU backend, vLLM's architecture still requires 
# `gpu_memory_utilization` to manage its internal KV cache memory allocation.
# You do not need to pass `device="cpu"` if the CPU extension was correctly compiled.

print(f"[{time.strftime('%H:%M:%S')}] Initializing vLLM engine with model: {MODEL_NAME}...")
try:
    llm = LLM(
        model=MODEL_NAME, 
        trust_remote_code=True, 
        gpu_memory_utilization=0.4
    )
    
    # Configure generation parameters
    sampling_params = SamplingParams(
        temperature=0.1, 
        max_tokens=50
    )
    
    prompt = "The capital of France is"
    print(f"\n[{time.strftime('%H:%M:%S')}] Starting inference...")
    print(f"Prompt: '{prompt}'")
    
    # Generate the response
    start_time = time.time()
    outputs = llm.generate([prompt], sampling_params)
    end_time = time.time()
    
    generated_text = outputs[0].outputs[0].text
    print(f"\n[{time.strftime('%H:%M:%S')}] Inference result:")
    print(f"--------------------------------------------------")
    print(generated_text.strip())
    print(f"--------------------------------------------------")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print("\n✅ Verification Successful: CPU backend operator invoked correctly!")

except Exception as e:
    print("\n❌ Verification Failed. Please check the error below:")
    print(e)
    print("\nTroubleshooting tips:")
    print("1. Ensure you compiled vLLM using the steps in install.sh")
    print("2. Ensure you have activated the correct virtual environment")
