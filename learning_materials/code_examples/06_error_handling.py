"""
Example 06: Robust Error Handling

Shows best practices for error handling in vLLM applications.

Usage:
    python 06_error_handling.py
"""

import asyncio
from typing import Optional
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine


async def generate_with_timeout(
    engine: AsyncLLMEngine,
    prompt: str,
    timeout: float = 30.0
) -> Optional[str]:
    """Generate with timeout protection."""
    try:
        sampling_params = SamplingParams(temperature=0.8, max_tokens=100)
        request_id = f"req-{id(prompt)}"

        # Use asyncio.wait_for for timeout
        async def _generate():
            final_output = None
            async for output in engine.generate(prompt, sampling_params, request_id):
                final_output = output
            return final_output.outputs[0].text if final_output else None

        result = await asyncio.wait_for(_generate(), timeout=timeout)
        return result

    except asyncio.TimeoutError:
        print(f"ERROR: Request timed out after {timeout}s")
        return None
    except Exception as e:
        print(f"ERROR: Generation failed: {e}")
        return None


async def main():
    """Demo error handling."""
    print("=== Error Handling Demo ===\n")

    engine_args = AsyncEngineArgs(
        model="facebook/opt-125m",
        trust_remote_code=True
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # Test 1: Successful generation
    result = await generate_with_timeout(engine, "Test prompt", timeout=30.0)
    if result:
        print(f"Success: {result[:50]}...")

    # Test 2: Timeout scenario (simulated)
    # result = await generate_with_timeout(engine, "Long prompt", timeout=0.001)

    print("\nError handling complete!")


if __name__ == "__main__":
    asyncio.run(main())
