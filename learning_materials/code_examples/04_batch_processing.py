"""
Example 04: Efficient Batch Processing

Shows how to process large datasets efficiently using vLLM's batching.

Usage:
    python 04_batch_processing.py
"""

import time
from typing import List
from vllm import LLM, SamplingParams


def process_in_batches(
    llm: LLM,
    prompts: List[str],
    batch_size: int = 32
) -> List[str]:
    """Process prompts in batches for efficiency."""
    all_results = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]

        sampling_params = SamplingParams(
            temperature=0.8,
            max_tokens=50
        )

        outputs = llm.generate(batch, sampling_params)
        results = [output.outputs[0].text for output in outputs]
        all_results.extend(results)

        print(f"Processed batch {i // batch_size + 1}: {len(batch)} prompts")

    return all_results


def main():
    """Demonstrate batch processing."""
    print("Initializing model...\n")
    llm = LLM(model="facebook/opt-125m", trust_remote_code=True)

    # Generate test dataset
    prompts = [f"Sample prompt {i}: " for i in range(100)]

    print(f"Processing {len(prompts)} prompts in batches...\n")

    start_time = time.time()
    results = process_in_batches(llm, prompts, batch_size=16)
    elapsed = time.time() - start_time

    print(f"\nCompleted in {elapsed:.2f} seconds")
    print(f"Throughput: {len(prompts) / elapsed:.2f} prompts/second")
    print(f"\nSample results:")
    for i, result in enumerate(results[:3], 1):
        print(f"  {i}. {result[:60]}...")


if __name__ == "__main__":
    main()
