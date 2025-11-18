"""
Example 05: Streaming Responses

Demonstrates real-time token streaming for better user experience.

Usage:
    python 05_streaming_responses.py
"""

import asyncio
import sys
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine


async def stream_tokens(
    engine: AsyncLLMEngine,
    prompt: str
) -> None:
    """Stream tokens as they are generated."""
    sampling_params = SamplingParams(temperature=0.8, max_tokens=100)
    request_id = "stream-request"

    print(f"Prompt: {prompt}")
    print("Streaming: ", end="", flush=True)

    previous_text = ""
    async for request_output in engine.generate(prompt, sampling_params, request_id):
        current_text = request_output.outputs[0].text
        new_text = current_text[len(previous_text):]
        previous_text = current_text

        # Print new tokens
        print(new_text, end="", flush=True)
        await asyncio.sleep(0.05)  # Simulate streaming delay

    print("\n")


async def main():
    """Main streaming demo."""
    engine_args = AsyncEngineArgs(
        model="facebook/opt-125m",
        trust_remote_code=True
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    await stream_tokens(engine, "The future of AI includes")


if __name__ == "__main__":
    asyncio.run(main())
