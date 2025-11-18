"""
Lab 02: Online Serving with AsyncEngine - Complete Solution

This file contains the complete solution for the async online serving lab.
"""

import asyncio
import time
import uuid
from typing import List, AsyncGenerator, Tuple
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine


async def create_async_engine(
    model_name: str = "facebook/opt-125m",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9
) -> AsyncLLMEngine:
    """
    Initialize an AsyncLLMEngine instance.

    Args:
        model_name: HuggingFace model identifier
        tensor_parallel_size: Number of GPUs to use
        gpu_memory_utilization: Fraction of GPU memory to use

    Returns:
        Initialized AsyncLLMEngine instance
    """
    engine_args = AsyncEngineArgs(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
    )

    engine = AsyncLLMEngine.from_engine_args(engine_args)
    return engine


def generate_request_id(prefix: str = "req") -> str:
    """
    Generate a unique request ID.

    Args:
        prefix: Prefix for the request ID

    Returns:
        Unique request ID string
    """
    unique_id = str(uuid.uuid4())[:8]
    return f"{prefix}-{unique_id}"


async def generate_single_request(
    engine: AsyncLLMEngine,
    prompt: str,
    sampling_params: SamplingParams,
    request_id: str
) -> str:
    """
    Generate completion for a single request.

    Args:
        engine: AsyncLLMEngine instance
        prompt: Input prompt
        sampling_params: Sampling parameters
        request_id: Unique request identifier

    Returns:
        Generated text
    """
    final_output = None

    # Submit request and iterate through results
    async for request_output in engine.generate(prompt, sampling_params, request_id):
        final_output = request_output

    # Extract generated text
    if final_output and final_output.outputs:
        return final_output.outputs[0].text
    return ""


async def generate_concurrent_requests(
    engine: AsyncLLMEngine,
    prompts: List[str],
    sampling_params: SamplingParams
) -> List[Tuple[str, str]]:
    """
    Process multiple requests concurrently.

    Args:
        engine: AsyncLLMEngine instance
        prompts: List of input prompts
        sampling_params: Sampling parameters

    Returns:
        List of (request_id, generated_text) tuples
    """
    tasks = []
    request_ids = []

    # Create tasks for each prompt
    for prompt in prompts:
        request_id = generate_request_id()
        request_ids.append(request_id)
        task = generate_single_request(engine, prompt, sampling_params, request_id)
        tasks.append(task)

    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks)

    # Combine request IDs with results
    return list(zip(request_ids, results))


async def main() -> None:
    """Main async function to run the online serving lab."""
    print("=== vLLM Async Online Serving Lab ===\n")

    # Initialize engine
    print("Initializing AsyncLLMEngine...")
    engine = await create_async_engine()
    print("Engine initialized successfully!\n")

    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=100,
    )

    # Test 1: Single request
    print("Processing single request...")
    single_prompt = "Explain async programming in Python"
    request_id = generate_request_id()
    print(f"Request ID: {request_id}")
    print(f"Prompt: {single_prompt}")

    result = await generate_single_request(
        engine, single_prompt, sampling_params, request_id
    )
    print(f"Generated: {result}\n")

    # Test 2: Concurrent requests
    print("Processing concurrent requests...")
    concurrent_prompts = [
        "The benefits of async programming are",
        "Machine learning inference requires",
        "High-performance computing involves",
    ]

    start_time = time.time()
    results = await generate_concurrent_requests(
        engine, concurrent_prompts, sampling_params
    )
    elapsed_time = time.time() - start_time

    for i, (req_id, output) in enumerate(results, 1):
        print(f"Request {i} ({req_id}): Complete")
        print(f"  Output: {output[:100]}...")

    print(f"\nAll {len(concurrent_prompts)} requests completed in {elapsed_time:.2f} seconds")
    print(f"Throughput: {len(concurrent_prompts) / elapsed_time:.2f} requests/second")


if __name__ == "__main__":
    asyncio.run(main())
