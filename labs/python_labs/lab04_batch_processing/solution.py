"""
Lab 04: Efficient Batch Processing - Complete Solution
"""

import time
from typing import List, Iterator, Tuple
from vllm import LLM, SamplingParams


def create_batches(data: List[str], batch_size: int) -> Iterator[List[str]]:
    """Create batches from a list of prompts."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def process_batch(
    llm: LLM,
    batch: List[str],
    sampling_params: SamplingParams
) -> List[str]:
    """Process a batch of prompts."""
    outputs = llm.generate(batch, sampling_params)
    return [output.outputs[0].text for output in outputs]


def measure_throughput(
    llm: LLM,
    prompts: List[str],
    batch_size: int,
    sampling_params: SamplingParams
) -> Tuple[float, float]:
    """Measure throughput for given batch size."""
    start_time = time.time()

    results = []
    for batch in create_batches(prompts, batch_size):
        batch_results = process_batch(llm, batch, sampling_params)
        results.extend(batch_results)

    elapsed_time = time.time() - start_time
    throughput = len(prompts) / elapsed_time

    return elapsed_time, throughput


def find_optimal_batch_size(
    llm: LLM,
    prompts: List[str],
    sampling_params: SamplingParams,
    batch_sizes: List[int]
) -> int:
    """Find optimal batch size by testing different values."""
    best_throughput = 0
    best_batch_size = batch_sizes[0]

    for batch_size in batch_sizes:
        _, throughput = measure_throughput(llm, prompts, batch_size, sampling_params)
        if throughput > best_throughput:
            best_throughput = throughput
            best_batch_size = batch_size

    return best_batch_size


def main() -> None:
    """Main function."""
    print("=== vLLM Batch Processing Lab ===\n")

    llm = LLM(model="facebook/opt-125m", trust_remote_code=True)

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=50
    )

    prompts = [f"Prompt {i}: " for i in range(100)]
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
