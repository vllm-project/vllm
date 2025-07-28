# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Example demonstrating streaming offline inference with AsyncLLM (V1 engine).

This script shows how to use vLLM's AsyncLLM engine for streaming token-by-token
output in offline inference scenarios. It demonstrates both DELTA mode (new tokens only)
and CUMULATIVE mode (complete output so far).

Usage:
    python examples/offline_inference/async_llm_streaming.py
    python examples/offline_inference/async_llm_streaming.py --model meta-llama/Llama-3.2-1B-Instruct
    python examples/offline_inference/async_llm_streaming.py --streaming-mode cumulative
"""  # noqa: E501

import asyncio
import os
import time

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import RequestOutputKind
from vllm.utils import FlexibleArgumentParser
from vllm.v1.engine.async_llm import AsyncLLM


def create_parser():
    """Create argument parser with AsyncEngineArgs and streaming options."""
    parser = FlexibleArgumentParser(description="AsyncLLM Streaming Inference Example")

    AsyncEngineArgs.add_cli_args(parser)
    parser.set_defaults(
        model="meta-llama/Llama-3.2-1B-Instruct",
        enforce_eager=True,  # Faster for examples
    )

    # Add sampling parameters
    sampling_group = parser.add_argument_group("Sampling parameters")
    sampling_group.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    sampling_group.add_argument(
        "--temperature", type=float, default=0.8, help="Sampling temperature"
    )
    sampling_group.add_argument(
        "--top-p", type=float, default=0.95, help="Top-p (nucleus) sampling"
    )
    sampling_group.add_argument("--top-k", type=int, default=-1, help="Top-k sampling")

    # Add streaming options
    streaming_group = parser.add_argument_group("Streaming options")
    streaming_group.add_argument(
        "--streaming-mode",
        choices=["delta", "cumulative"],
        default="delta",
        help="Streaming mode: 'delta' for new tokens only, "
        "'cumulative' for complete output so far",
    )
    streaming_group.add_argument(
        "--show-timing",
        action="store_true",
        help="Show timing information for each token",
    )

    return parser


async def stream_response(
    engine: AsyncLLM,
    prompt: str,
    sampling_params: SamplingParams,
    request_id: str,
    show_timing: bool = False,
) -> None:
    """Stream response from AsyncLLM and display tokens as they arrive."""

    print(f"\n🚀 Prompt: {prompt!r}")
    print(f"📝 Streaming mode: {sampling_params.output_kind.name}")
    print("🔄 Generating", end="", flush=True)

    if sampling_params.output_kind == RequestOutputKind.DELTA:
        print(" (token-by-token):")
        print("💬 ", end="", flush=True)
    else:
        print(" (cumulative):")

    start_time = time.time()
    token_count = 0
    last_time = start_time

    try:
        # Stream tokens from AsyncLLM
        async for output in engine.generate(
            request_id=request_id, prompt=prompt, sampling_params=sampling_params
        ):
            current_time = time.time()

            # Process each completion in the output
            for completion in output.outputs:
                if sampling_params.output_kind == RequestOutputKind.DELTA:
                    # In DELTA mode, we get only new tokens
                    new_text = completion.text
                    if new_text:
                        print(new_text, end="", flush=True)
                        token_count += len(completion.token_ids)

                        if show_timing:
                            token_time = current_time - last_time
                            print(f" [{token_time:.3f}s]", end="", flush=True)

                        last_time = current_time

                else:  # CUMULATIVE mode
                    # In CUMULATIVE mode, we get the complete output so far
                    complete_text = completion.text
                    token_count = len(completion.token_ids)

                    # Clear the line and print the updated text
                    print(f"\r💬 {complete_text}", end="", flush=True)

                    if show_timing:
                        token_time = current_time - last_time
                        print(f" [{token_time:.3f}s]", end="", flush=True)
                        last_time = current_time

            # Check if generation is finished
            if output.finished:
                total_time = current_time - start_time
                print(
                    f"\n✅ Finished! Generated {token_count}"
                    f" tokens in {total_time:.2f}s"
                )

                if token_count > 0:
                    tokens_per_second = token_count / total_time
                    print(f"⚡️ Speed: {tokens_per_second:.1f} tokens/second")
                break

    except Exception as e:
        print(f"\n❌ Error during streaming: {e}")
        raise


async def run_streaming_examples(args) -> None:
    """Run streaming examples with different prompts and configurations."""

    # Ensure V1 is enabled
    os.environ["VLLM_USE_V1"] = "1"

    # Extract sampling parameters
    max_tokens = args.pop("max_tokens", 100)
    temperature = args.pop("temperature", 0.8)
    top_p = args.pop("top_p", 0.95)
    top_k = args.pop("top_k", -1)
    streaming_mode = args.pop("streaming_mode", "delta")
    show_timing = args.pop("show_timing", False)

    # Determine output kind
    output_kind = (
        RequestOutputKind.DELTA
        if streaming_mode == "delta"
        else RequestOutputKind.CUMULATIVE
    )

    # Create sampling parameters
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        output_kind=output_kind,
        # Use a seed for reproducible results in examples
        seed=42,
    )

    print(f"🔧 Initializing AsyncLLM with model: {args.get('model', 'default')}")

    # Create AsyncLLM engine
    engine_args = AsyncEngineArgs(**args)
    engine = AsyncLLM.from_engine_args(engine_args)

    try:
        # Sample prompts for demonstration
        prompts = [
            "The future of artificial intelligence is",
            "In a galaxy far, far away",
            "The key to happiness is",
            "Climate change solutions include",
        ]

        print(f"🎯 Running {len(prompts)} streaming examples...")

        # Process each prompt
        for i, prompt in enumerate(prompts, 1):
            print(f"\n{'=' * 60}")
            print(f"Example {i}/{len(prompts)}")
            print(f"{'=' * 60}")

            request_id = f"stream-example-{i}"

            await stream_response(
                engine=engine,
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
                show_timing=show_timing,
            )

            # Small delay between examples for better readability
            if i < len(prompts):
                await asyncio.sleep(1)

        print("\n🎉 All examples completed successfully!")
        print("💡 Try different streaming modes with --streaming-mode delta|cumulative")
        print("💡 Add --show-timing to see per-token timing information")

    finally:
        # Clean up the engine
        engine.shutdown()


def main():
    """Main function."""
    parser = create_parser()
    args = vars(parser.parse_args())

    # Run the async examples
    try:
        asyncio.run(run_streaming_examples(args))
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")


if __name__ == "__main__":
    main()
