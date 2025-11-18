"""
Example 12: Concurrent Request Processing

Demonstrates processing multiple requests concurrently with asyncio.

Usage:
    python 12_concurrent_requests.py
"""

import asyncio
import time
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine


async def process_request(
    engine: AsyncLLMEngine,
    prompt: str,
    request_id: str
) -> tuple:
    """Process a single request."""
    start = time.time()
    sampling_params = SamplingParams(temperature=0.8, max_tokens=50)

    final_output = None
    async for output in engine.generate(prompt, sampling_params, request_id):
        final_output = output

    latency = time.time() - start
    text = final_output.outputs[0].text if final_output else ""

    return request_id, text, latency


async def main():
    """Process multiple requests concurrently."""
    print("=== Concurrent Request Processing ===\n")

    engine_args = AsyncEngineArgs(
        model="facebook/opt-125m",
        trust_remote_code=True
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # Create multiple requests
    prompts = [
        "The future of technology",
        "Artificial intelligence is",
        "Cloud computing enables",
        "Data science helps",
        "Machine learning models",
    ]

    print(f"Processing {len(prompts)} requests concurrently...\n")
    start_time = time.time()

    # Process all concurrently
    tasks = [
        process_request(engine, prompt, f"req-{i}")
        for i, prompt in enumerate(prompts)
    ]
    results = await asyncio.gather(*tasks)

    total_time = time.time() - start_time

    # Display results
    for req_id, text, latency in results:
        print(f"{req_id}: {latency:.2f}s - {text[:50]}...")

    print(f"\nTotal time: {total_time:.2f}s")
    print(f"Average latency: {sum(r[2] for r in results) / len(results):.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
