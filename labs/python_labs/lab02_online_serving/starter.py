"""
Lab 02: Online Serving with AsyncEngine - Starter Code

Implement async online serving using vLLM's AsyncLLMEngine.
Complete the TODOs to make this code functional.
"""

import asyncio
import time
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
    # TODO 1: Create AsyncEngineArgs and initialize AsyncLLMEngine
    # Hint:
    # 1. Create AsyncEngineArgs with model, tensor_parallel_size, gpu_memory_utilization
    # 2. Use AsyncLLMEngine.from_engine_args(engine_args) to create the engine
    pass


def generate_request_id(prefix: str = "req") -> str:
    """
    Generate a unique request ID.

    Args:
        prefix: Prefix for the request ID

    Returns:
        Unique request ID string
    """
    # TODO 2: Generate a unique request ID
    # Hint: Use timestamp or UUID for uniqueness
    # Example: f"{prefix}-{int(time.time() * 1000)}"
    pass


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
    # TODO 3: Submit request and collect results
    # Hint:
    # 1. Call engine.generate(prompt, sampling_params, request_id)
    # 2. This returns an async generator, iterate with 'async for'
    # 3. The final result contains the complete output

    final_output = None

    # TODO: Implement async iteration here
    # async for request_output in engine.generate(...):
    #     final_output = request_output

    # TODO 4: Extract and return the generated text
    # Hint: Access request_output.outputs[0].text
    pass


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
    # TODO 5: Create tasks for concurrent execution
    # Hint:
    # 1. Create a list of coroutines using generate_single_request
    # 2. Use asyncio.gather(*tasks) to run them concurrently
    # 3. Return results with request IDs

    tasks = []
    request_ids = []

    # TODO: Create tasks for each prompt
    # for prompt in prompts:
    #     request_id = generate_request_id()
    #     request_ids.append(request_id)
    #     task = generate_single_request(engine, prompt, sampling_params, request_id)
    #     tasks.append(task)

    # TODO: Gather results
    # results = await asyncio.gather(*tasks)

    # TODO: Return list of (request_id, result) tuples
    pass


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
