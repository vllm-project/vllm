# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test for pause/resume with keep mode.

This test uses concurrent tasks to verify the engine truly stops generating
during pause:
1. Generator task: continuously generates and logs time between tokens
2. Controller task: sends pause/resume commands

If the engine properly pauses, we should see a gap in token timestamps
matching the pause duration.
"""

import asyncio
import time

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

PAUSE_DURATION = 3.0  # seconds


async def main():
    # Create engine with a small model
    engine_args = AsyncEngineArgs(
        model="facebook/opt-125m",
        enforce_eager=True,
    )
    engine = AsyncLLM.from_engine_args(engine_args)

    prompt = "Write a story about a dragon. Once upon a time"
    sampling_params = SamplingParams(max_tokens=30, ignore_eos=True)

    # Track token arrival times
    token_times: list[tuple[int, float]] = []  # (token_count, timestamp)
    pause_time: float = 0
    resume_time: float = 0

    async def generator_task():
        """Generate tokens and record timestamps."""
        async for output in engine.generate(
            request_id="test-req",
            prompt=prompt,
            sampling_params=sampling_params,
        ):
            token_count = len(output.outputs[0].token_ids)
            token_times.append((token_count, time.monotonic()))
            print(
                f"Token {token_count} arrived:"
                f"T={token_times[-1][1] - token_times[0][1]:.3f}s"
            )
        return output

    async def controller_task():
        """Pause and resume the engine after some tokens generated."""
        nonlocal pause_time, resume_time

        # Wait for some tokens to be generated
        while len(token_times) < 5:
            await asyncio.sleep(0.01)

        print(f"\nPausing engine (keep mode) at token {len(token_times)}")
        pause_time = time.monotonic()
        await engine.pause_generation(mode="keep")
        print(f"Paused! Sleeping for {PAUSE_DURATION}s...")

        # Sleep while paused - no tokens should be generated during this time
        await asyncio.sleep(PAUSE_DURATION)

        print("Resuming engine...")
        await engine.resume_generation()
        resume_time = time.monotonic()
        print("Resumed!\n")

    # Run both tasks concurrently
    gen_task = asyncio.create_task(generator_task())
    ctrl_task = asyncio.create_task(controller_task())

    final_output, _ = await asyncio.gather(gen_task, ctrl_task)

    # Analyze token timing gaps
    print("\n=== Token Timing Analysis ===")
    max_gap = 0.0
    max_gap_tokens = (0, 0)
    # Start from index 2 to exclude TTFT (time to first token)
    for i in range(2, len(token_times)):
        gap = token_times[i][1] - token_times[i - 1][1]
        if gap > max_gap:
            max_gap = gap
            max_gap_tokens = (token_times[i - 1][0], token_times[i][0])
        if gap > 0.5:  # Log significant gaps
            print(
                f"  Gap of {gap:.3f}s"
                f" between token {token_times[i - 1][0]} and {token_times[i][0]}"
            )

    print(
        f"\nLargest gap: {max_gap:.3f}s "
        f"(between tokens {max_gap_tokens[0]} and {max_gap_tokens[1]})"
    )
    print(f"Pause duration: {PAUSE_DURATION}s")
    print(f"Final token count: {len(final_output.outputs[0].token_ids)}")
    print(f"Request finished: {final_output.finished}")

    # Verify the pause actually stopped generation.
    # The max gap should be approximately the pause duration.
    if max_gap >= PAUSE_DURATION * 0.9:
        print(f"\n✓ Test passed! Engine paused for ~{max_gap:.1f}s")
    else:
        print(f"\n✗ Test failed! Expected ~{PAUSE_DURATION}s gap, got {max_gap:.3f}s")
        raise AssertionError("Engine did not properly pause")

    # Verify request completed
    assert final_output.finished, "Request should have finished"
    assert len(final_output.outputs[0].token_ids) == 30, "Should have all tokens"

    engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
