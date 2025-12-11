# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Simple example demonstrating streaming offline inference with AsyncLLM (V1 engine).

This script shows the core functionality of vLLM's AsyncLLM engine for streaming
token-by-token output in offline inference scenarios. It demonstrates DELTA mode
streaming where you receive new tokens as they are generated.

Usage:
    python examples/offline_inference/async_llm_streaming.py
"""

import argparse
import asyncio
from enum import IntEnum

from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM


class Mode(IntEnum):
    PARALLEL = 1
    SEQUENTIAL = 2


async def stream_response(engine: AsyncLLM, prompt: str, request_id: str) -> None:
    """
    Stream response from AsyncLLM and display tokens as they arrive.

    This function demonstrates the core streaming pattern:
    1. Create SamplingParams with DELTA output kind
    2. Call engine.generate() and iterate over the async generator
    3. Print new tokens as they arrive
    4. Handle the finished flag to know when generation is complete
    """
    print(f"\n🚀 Prompt: {prompt!r}")
    print("💬 Response: ", end="", flush=True)

    # Configure sampling parameters for streaming
    sampling_params = SamplingParams(
        max_tokens=100,
        temperature=0.8,
        top_p=0.95,
        seed=42,  # For reproducible results
        output_kind=RequestOutputKind.DELTA,  # Get only new tokens each iteration
    )

    try:
        # Stream tokens from AsyncLLM
        async for output in engine.generate(
            request_id=request_id, prompt=prompt, sampling_params=sampling_params
        ):
            # Process each completion in the output
            for completion in output.outputs:
                # In DELTA mode, we get only new tokens generated since last iteration
                new_text = completion.text
                if new_text:
                    print(new_text, end="", flush=True)

            # Check if generation is finished
            if output.finished:
                print("\n✅ Generation complete!")
                break

    except Exception as e:
        print(f"\n❌ Error during streaming: {e}")
        raise


async def main(mode: Mode = Mode.SEQUENTIAL):
    print("🔧 Initializing AsyncLLM...")

    # Create AsyncLLM engine with simple configuration
    engine_args = AsyncEngineArgs(
        # model="meta-llama/Llama-3.2-1B-Instruct",
        model="facebook/opt-125m",
        enforce_eager=True,  # Faster startup for examples
    )
    engine = AsyncLLM.from_engine_args(engine_args)

    try:
        # Example prompts to demonstrate streaming
        prompts = [
            "The future of artificial intelligence is",
            "In a galaxy far, far away",
            "The key to happiness is",
        ]
        print(f"🎯 Running {len(prompts)} streaming examples...")

        if mode == Mode.SEQUENTIAL:
            # Process each prompt
            for i, prompt in enumerate(prompts, 1):
                print(f"\n{'=' * 60}")
                print(f"Example {i}/{len(prompts)}")
                print(f"{'=' * 60}")

                request_id = f"stream-example-{i}"
                await stream_response(engine, prompt, request_id)

                # Brief pause between examples
                if i < len(prompts):
                    await asyncio.sleep(0.5)

        elif mode == Mode.PARALLEL:
            request_ids = [f"stream-example-{i}" for i in range(len(prompts))]
            # Configure sampling parameters for streaming
            # We use the same sampling parameters for all requests
            # but you can also use different sampling parameters for each request
            sampling_params = SamplingParams(
                max_tokens=100,
                temperature=0.8,
                top_p=0.95,
                seed=42,  # For reproducible results
                output_kind=RequestOutputKind.DELTA,
                # Get only new tokens each iteration
            )

            # --- Rich Layout Setup ---
            layout = Layout()

            # Split the layout vertically into sections, one per prompt
            # Each section is named after the request_id for easy access
            layout.split_column(*[Layout(name=req_id) for req_id in request_ids])

            # Initialize content buffers for each request
            # content_map maps request_id -> Rich Text object
            content_map: dict[str, Text] = {}

            for request_id, prompt in zip(request_ids, prompts):
                # Create a styled Text object
                text = Text()
                text.append(f"\n🚀 Prompt: {prompt!r}\n", style="bold cyan")
                text.append("💬 Response: \n", style="dim")
                content_map[request_id] = text

                # Initialize the panel in the layout
                layout[request_id].update(
                    Panel(text, title=f"Request: {request_id}", border_style="blue")
                )

            # Use Live to update the screen dynamically
            # screen=True enables full-screen mode (alt screen)
            with Live(layout, refresh_per_second=15, screen=True):
                # Start generation
                async for output in engine.generate(
                    request_id=request_ids,
                    prompt=prompts,
                    sampling_params=sampling_params,
                ):
                    req_id = output.request_id

                    # Get the content buffer for this request
                    text_buffer = content_map[req_id]

                    # Append new tokens
                    # Assuming n=1, so we take the first completion
                    for completion in output.outputs:
                        if completion.text:
                            text_buffer.append(completion.text)

                    # Check if finished
                    if output.finished:
                        text_buffer.append(
                            "\n✅ Generation complete!", style="bold green"
                        )
                        # Update panel border to green to indicate completion
                        layout[req_id].update(
                            Panel(
                                text_buffer,
                                title=f"Request: {req_id} (Done)",
                                border_style="green",
                            )
                        )
                    else:
                        # Update the panel content (in case it wasn't automatically
                        # reflected, though modifying the Text object in-place
                        # usually works with Rich if referentially transparent,
                        # explicitly updating the Panel ensures the render tree is
                        # correct)
                        layout[req_id].update(
                            Panel(
                                text_buffer,
                                title=f"Request: {req_id}",
                                border_style="blue",
                            )
                        )

                # After generation is complete, keep the display active until Ctrl+C
                while True:
                    await asyncio.sleep(0.1)

        print("\n🎉 All streaming examples completed!")

    except KeyboardInterrupt:
        pass  # Graceful exit on Ctrl+C
    except Exception as e:
        # Print error after exiting Live context so it's visible
        print(f"\n❌ Error during streaming: {e}")
        raise

    finally:
        # Always clean up the engine
        print("🔧 Shutting down engine...")
        engine.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AsyncLLM Streaming Example")
    parser.add_argument(
        "--mode",
        type=int,
        default=1,
        choices=[1, 2],
        help="1: Parallel, 2: Sequential",
    )
    args = parser.parse_args()

    try:
        asyncio.run(main(mode=Mode(args.mode)))
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
