"""
Example 01: Simple Offline Inference

This example demonstrates the most basic usage of vLLM for offline text generation.
Perfect for beginners starting with vLLM.

Usage:
    python 01_simple_inference.py
"""

from vllm import LLM, SamplingParams


def main():
    """Run a simple inference example."""
    # Initialize the model
    # facebook/opt-125m is a small model good for testing
    llm = LLM(
        model="facebook/opt-125m",
        trust_remote_code=True,
        gpu_memory_utilization=0.9
    )

    # Define prompts
    prompts = [
        "The capital of France is",
        "Machine learning is",
        "Python is a programming language that",
    ]

    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=0.8,  # Controls randomness (0=deterministic, higher=more random)
        top_p=0.95,       # Nucleus sampling threshold
        max_tokens=50,    # Maximum tokens to generate
    )

    # Generate completions
    print("Generating completions...\n")
    outputs = llm.generate(prompts, sampling_params)

    # Display results
    for i, output in enumerate(outputs, 1):
        prompt = output.prompt
        generated = output.outputs[0].text
        print(f"Prompt {i}: {prompt}")
        print(f"Generated: {generated}")
        print("-" * 80)


if __name__ == "__main__":
    main()
