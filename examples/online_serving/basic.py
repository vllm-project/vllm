# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import signal

from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM

# Sample prompts.
prompts = [
    # "Hello, my name is",
    # "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
    "The future of artificial intelligence is",
    "In a galaxy far, far away",
    "The key to happiness is",
]

# Configure sampling parameters for streaming
sampling_params = SamplingParams(
    max_tokens=100,
    temperature=0.8,
    top_p=0.95,
    seed=42,  # For reproducible results
    output_kind=RequestOutputKind.DELTA,  # Get only new tokens each iteration
)


async def main():
    # Create AsyncLLM engine
    engine_args = AsyncEngineArgs(
        model="facebook/opt-125m",
        enforce_eager=True,
    )
    engine = AsyncLLM.from_engine_args(engine_args)

    request_ids = [f"req_{i}" for i in range(len(prompts))]

    # --- Rich Layout Setup ---
    layout = Layout()

    # Split the layout vertically into sections, one per prompt
    # Each section is named after the request_id for easy access
    layout.split_column(*[Layout(name=req_id) for req_id in request_ids])

    # Initialize content buffers for each request
    # content_map maps request_id -> Rich Text object
    content_map: dict[str, Text] = {}

    for req_id, prompt in zip(request_ids, prompts):
        # Create a styled Text object
        text = Text()
        text.append(f"Prompt: {prompt}\n", style="bold cyan")
        text.append("-" * 30 + "\n", style="dim")
        content_map[req_id] = text

        # Initialize the panel in the layout
        layout[req_id].update(
            Panel(text, title=f"Request: {req_id}", border_style="blue")
        )

    try:
        # Use Live to update the screen dynamically
        # screen=True enables full-screen mode (alt screen)
        with Live(layout, refresh_per_second=15, screen=True):
            # Start generation
            async for output in engine.generate(
                request_id=request_ids, prompt=prompts, sampling_params=sampling_params
            ):
                req_id = output.request_id

                # Get the content buffer for this request
                text_buffer = content_map[req_id]

                # Append new tokens
                # Assuming n=1, so we take the first completion
                for completion in output.outputs:
                    # text_buffer.append(f"{[req_id]}:")
                    if completion.text:
                        text_buffer.append(completion.text)

                # Check if finished
                if output.finished:
                    # print("\n✅ Generation complete!")
                    text_buffer.append("\n[Finished]", style="bold green")
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
                            text_buffer, title=f"Request: {req_id}", border_style="blue"
                        )
                    )

            # After generation is complete, keep the display active until Ctrl+C
            while True:

                def signal_handler(sig, frame):
                    raise KeyboardInterrupt

                signal.signal(signal.SIGINT, signal_handler)
                await asyncio.sleep(0.1)

    except KeyboardInterrupt:
        pass  # Graceful exit on Ctrl+C
    except Exception as e:
        # Print error after exiting Live context so it's visible
        print(f"\n❌ Error during streaming: {e}")
        raise

    finally:
        print("🔧 Shutting down engine...")
        engine.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
