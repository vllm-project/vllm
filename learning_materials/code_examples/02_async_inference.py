"""
Example 02: Async Inference with AsyncLLMEngine

Demonstrates asynchronous inference for online serving scenarios.
This is essential for building production APIs.

Usage:
    python 02_async_inference.py
"""

import asyncio
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine


async def generate_text(
    engine: AsyncLLMEngine,
    prompt: str,
    request_id: str
) -> str:
    """Generate text asynchronously."""
    sampling_params = SamplingParams(temperature=0.8, max_tokens=100)

    # The engine.generate() returns an async generator
    # We iterate through it to get progressive results
    final_output = None
    async for request_output in engine.generate(prompt, sampling_params, request_id):
        final_output = request_output

    # Extract the generated text
    if final_output and final_output.outputs:
        return final_output.outputs[0].text
    return ""


async def main():
    """Main async function."""
    print("Initializing AsyncLLMEngine...\n")

    # Create engine arguments
    engine_args = AsyncEngineArgs(
        model="facebook/opt-125m",
        trust_remote_code=True,
        tensor_parallel_size=1,
    )

    # Initialize the async engine
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # Test prompts
    prompts = [
        "Async programming allows",
        "The benefits of concurrent execution include",
    ]

    # Process requests concurrently
    tasks = []
    for i, prompt in enumerate(prompts):
        task = generate_text(engine, prompt, f"request-{i}")
        tasks.append(task)

    # Wait for all to complete
    results = await asyncio.gather(*tasks)

    # Display results
    for i, (prompt, result) in enumerate(zip(prompts, results), 1):
        print(f"Request {i}:")
        print(f"  Prompt: {prompt}")
        print(f"  Result: {result}\n")


if __name__ == "__main__":
    asyncio.run(main())
