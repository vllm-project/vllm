"""Lab 05: Streaming Output - Complete Solution"""

import asyncio
import sys
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
    previous_text = ""
    async for request_output in engine.generate(prompt, sampling_params, request_id):
        current_text = request_output.outputs[0].text
        new_text = current_text[len(previous_text):]
        previous_text = current_text
        yield new_text


async def main():
    """Main streaming demo."""
    print("=== Streaming Output Lab ===\n")

    engine_args = AsyncEngineArgs(
        model="facebook/opt-125m",
        trust_remote_code=True
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    sampling_params = SamplingParams(temperature=0.8, max_tokens=100)

    prompt = "Write a short story about"
    print(f"Prompt: {prompt}")
    print("\nStreaming output: ", end="", flush=True)

    async for token in stream_generate(engine, prompt, sampling_params, "req-1"):
        print(token, end="", flush=True)
        await asyncio.sleep(0.05)  # Simulate streaming delay

    print("\n\nStreaming complete!")


if __name__ == "__main__":
    asyncio.run(main())
