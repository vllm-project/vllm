"""Lab 05: Streaming Output - Starter Code"""

import asyncio
from typing import AsyncGenerator
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine


async def stream_generate(
    engine: AsyncLLMEngine,
    prompt: str,
    sampling_params: SamplingParams,
    request_id: str
) -> AsyncGenerator[str, None]:
    """Stream tokens as they are generated."""
    # TODO 1: Implement streaming logic
    # Hint: async for request_output in engine.generate(...)
    #       yield incremental text
    pass


async def main():
    """Main streaming demo."""
    print("=== Streaming Output Lab ===\n")

    # TODO 2: Initialize AsyncEngine
    # TODO 3: Stream output and display incrementally

    print("Streaming complete!")


if __name__ == "__main__":
    asyncio.run(main())
