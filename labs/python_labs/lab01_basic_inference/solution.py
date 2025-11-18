"""
Lab 01: Basic vLLM Offline Inference - Complete Solution

This file contains the complete solution for the basic inference lab.
"""

from typing import List, Optional
from vllm import LLM, SamplingParams


def create_llm(
    model_name: str = "facebook/opt-125m",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9
) -> LLM:
    """
    Initialize a vLLM LLM instance.

    Args:
        model_name: HuggingFace model identifier
        tensor_parallel_size: Number of GPUs to use for tensor parallelism
        gpu_memory_utilization: Fraction of GPU memory to use

    Returns:
        Initialized LLM instance
    """
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
    )
    return llm


def create_sampling_params(
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_tokens: int = 100
) -> SamplingParams:
    """
    Create sampling parameters for text generation.

    Args:
        temperature: Controls randomness (0.0 = deterministic, higher = more random)
        top_p: Nucleus sampling parameter
        max_tokens: Maximum number of tokens to generate

    Returns:
        SamplingParams instance
    """
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    return sampling_params


def generate_completions(
    llm: LLM,
    prompts: List[str],
    sampling_params: SamplingParams
) -> List:
    """
    Generate completions for a list of prompts.

    Args:
        llm: The LLM instance to use
        prompts: List of input prompts
        sampling_params: Sampling parameters for generation

    Returns:
        List of RequestOutput objects
    """
    outputs = llm.generate(prompts, sampling_params)
    return outputs


def display_outputs(outputs: List) -> None:
    """
    Display the generated outputs in a readable format.

    Args:
        outputs: List of RequestOutput objects from vLLM
    """
    print("\n=== Generated Completions ===\n")

    for i, output in enumerate(outputs, 1):
        prompt = output.prompt
        generated_text = output.outputs[0].text

        print(f"--- Prompt {i} ---")
        print(f"Input: {prompt}")
        print(f"Output: {generated_text}")
        print()


def main() -> None:
    """Main function to run the inference lab."""
    print("=== vLLM Basic Inference Lab ===\n")

    # Define test prompts
    prompts = [
        "Hello, my name is",
        "The future of AI is",
        "In a world where technology",
    ]

    print(f"Generating completions for {len(prompts)} prompts...\n")

    # Step 1: Initialize LLM
    llm = create_llm()

    # Step 2: Create sampling parameters
    sampling_params = create_sampling_params()

    # Step 3: Generate completions
    outputs = generate_completions(llm, prompts, sampling_params)

    # Step 4: Display results
    display_outputs(outputs)

    print("Inference completed successfully!")


if __name__ == "__main__":
    main()
