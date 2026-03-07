# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Example demonstrating how to load model weights from CPU using pt_load_map_location.

This is useful when:
- You want to explicitly load PyTorch checkpoints from CPU
- You need to manage memory usage during model initialization
- You want to map weights from one device to another

The pt_load_map_location parameter works the same as PyTorch's torch.load(map_location=...)
and defaults to "cpu" for most efficient loading.
"""

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "The advantages of loading weights from CPU include",
    "When should you use CPU weight loading?",
    "Memory management in machine learning is important because",
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50)


def main():
    # Example 1: Explicitly load weights from CPU (default behavior)
    print("=== Example 1: Loading weights from CPU ===")
    llm = LLM(
        model="facebook/opt-125m",
        pt_load_map_location="cpu"  # Explicitly specify CPU loading
    )
    
    outputs = llm.generate(prompts[:1], sampling_params)
    for output in outputs:
        print(f"Prompt: {output.prompt}")
        print(f"Output: {output.outputs[0].text}")
    
    # Example 2: Using device mapping (useful for multi-GPU setups)
    print("\n=== Example 2: Device mapping example ===")
    # Note: This example shows the syntax, but may not be applicable 
    # unless you have multiple CUDA devices available
    try:
        llm_mapped = LLM(
            model="facebook/opt-125m",
            pt_load_map_location={"": "cpu"}  # Alternative syntax for CPU
        )
        
        outputs = llm_mapped.generate(prompts[1:2], sampling_params)
        for output in outputs:
            print(f"Prompt: {output.prompt}")
            print(f"Output: {output.outputs[0].text}")
            
    except Exception as e:
        print(f"Device mapping example failed (this is normal if no CUDA available): {e}")
    
    # Example 3: Default behavior (pt_load_map_location="cpu" is the default)
    print("\n=== Example 3: Default behavior (CPU loading) ===")
    llm_default = LLM(model="facebook/opt-125m")  # Uses CPU loading by default
    
    outputs = llm_default.generate(prompts[2:3], sampling_params)
    for output in outputs:
        print(f"Prompt: {output.prompt}")
        print(f"Output: {output.outputs[0].text}")


if __name__ == "__main__":
    main()