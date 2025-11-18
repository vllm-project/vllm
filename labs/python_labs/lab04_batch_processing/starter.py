"""
Lab 04: Efficient Batch Processing - Starter Code

Learn to process large datasets efficiently using batching.
"""

import time
from typing import List, Iterator, Tuple
from vllm import LLM, SamplingParams


def create_batches(data: List[str], batch_size: int) -> Iterator[List[str]]:
    """
    Create batches from a list of prompts.

    Args:
        data: List of prompts
        batch_size: Size of each batch

    Yields:
        Batches of prompts
    """
    # TODO 1: Implement batching logic
    # Hint: Use list slicing in a loop or yield from chunks
    pass


def process_batch(
    llm: LLM,
    batch: List[str],
    sampling_params: SamplingParams
) -> List[str]:
    """
    Process a batch of prompts.

    Args:
        llm: LLM instance
        batch: List of prompts
        sampling_params: Sampling parameters

    Returns:
        List of generated texts
    """
    # TODO 2: Process batch and extract outputs
    # Hint: Use llm.generate(batch, sampling_params)
    pass


def measure_throughput(
    llm: LLM,
    prompts: List[str],
    batch_size: int,
    sampling_params: SamplingParams
) -> Tuple[float, float]:
    """
    Measure throughput for given batch size.

    Args:
        llm: LLM instance
        prompts: List of all prompts
        batch_size: Batch size to test
        sampling_params: Sampling parameters

    Returns:
        (elapsed_time, throughput) tuple
    """
    # TODO 3: Measure processing time and calculate throughput
    start_time = time.time()

    results = []
    # TODO: Process all batches

    elapsed_time = time.time() - start_time
    throughput = len(prompts) / elapsed_time

    return elapsed_time, throughput


def find_optimal_batch_size(
    llm: LLM,
    prompts: List[str],
    sampling_params: SamplingParams,
    batch_sizes: List[int]
) -> int:
    """
    Find optimal batch size by testing different values.

    Args:
        llm: LLM instance
        prompts: Sample prompts for testing
        sampling_params: Sampling parameters
        batch_sizes: List of batch sizes to test

    Returns:
        Optimal batch size
    """
    # TODO 4: Test each batch size and find the best one
    pass


def main() -> None:
    """Main function."""
    print("=== vLLM Batch Processing Lab ===\n")

    # Initialize LLM
    llm = LLM(model="facebook/opt-125m", trust_remote_code=True)

    # Create sampling params
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=50
    )

    # Generate test dataset
    prompts = [f"Prompt {i}: " for i in range(100)]

    # Test different batch sizes
    batch_sizes = [4, 8, 16, 32]

    print(f"Processing {len(prompts)} prompts...\n")

    for batch_size in batch_sizes:
        elapsed, throughput = measure_throughput(
            llm, prompts, batch_size, sampling_params
        )
        print(f"Batch size: {batch_size}")
        print(f"Time: {elapsed:.2f}s, Throughput: {throughput:.2f} prompts/s\n")


if __name__ == "__main__":
    main()
