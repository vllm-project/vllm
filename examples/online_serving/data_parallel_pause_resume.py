# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test pause/resume with Data Parallel (DP) via HTTP API.

This example demonstrates coordinated pause/resume across multiple DP ranks.
The pause synchronizes across all DP engines via all-reduce.

Prerequisites:
    Start a vLLM server with data parallelism:

    $ VLLM_SERVER_DEV_MODE=1 vllm serve facebook/opt-125m \
        --enforce-eager \
        --data-parallel-size 4 \
        --tensor-parallel-size 1

    Then run this script:

    $ python data_parallel_pause_resume.py

The test verifies pause works by:
1. Starting a streaming generation request
2. Pausing the server mid-generation
3. Sleeping for PAUSE_DURATION seconds
4. Resuming the server
5. Verifying there was a gap in token generation matching the pause duration
"""

import argparse
import threading
import time

import requests
from openai import OpenAI

BASE_URL = "http://localhost:8000"
MODEL_NAME = "facebook/opt-125m"
PAUSE_DURATION = 3.0


def pause_generation(base_url: str, mode: str = "keep") -> None:
    """Pause generation via HTTP endpoint."""
    url = f"{base_url}/pause"
    response = requests.post(url, params={"mode": mode}, timeout=60)
    response.raise_for_status()
    print("Server paused")


def resume_generation(base_url: str) -> None:
    """Resume generation via HTTP endpoint."""
    url = f"{base_url}/resume"
    response = requests.post(url, timeout=60)
    response.raise_for_status()
    print("Server resumed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--model", default=MODEL_NAME)
    args = parser.parse_args()

    client = OpenAI(
        base_url=f"{args.base_url}/v1",
        api_key="EMPTY",
    )

    prompt = "Write a long story about a dragon. Once upon a time"
    token_times: list[float] = []
    pause_token_idx = 0
    pause_triggered = threading.Event()

    def generator_thread():
        """Stream tokens and record timestamps."""
        stream = client.completions.create(
            model=args.model,
            prompt=prompt,
            max_tokens=50,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].text:
                token_times.append(time.monotonic())
                token_count = len(token_times)
                print(f"Token {token_count}: {chunk.choices[0].text!r}")

                # Signal controller after some tokens
                if token_count >= 5 and not pause_triggered.is_set():
                    pause_triggered.set()

    def controller_thread():
        """Pause and resume the server."""
        nonlocal pause_token_idx

        # Wait for some tokens
        pause_triggered.wait()

        print(f"\nPausing server (keep mode) at token {len(token_times)}...")
        pause_generation(args.base_url, mode="keep")
        pause_token_idx = len(token_times)
        print(f"Sleeping for {PAUSE_DURATION}s...")

        time.sleep(PAUSE_DURATION)

        print("Resuming server...")
        resume_generation(args.base_url)
        print("Resumed!\n")

    # Run both threads
    gen_thread = threading.Thread(target=generator_thread)
    ctrl_thread = threading.Thread(target=controller_thread)

    gen_thread.start()
    ctrl_thread.start()

    gen_thread.join()
    ctrl_thread.join()

    # Check gap at the pause point
    pause_gap = token_times[pause_token_idx] - token_times[pause_token_idx - 1]
    print(
        f"\nGap after pause (token {pause_token_idx} -> "
        f"{pause_token_idx + 1}): {pause_gap:.3f}s"
    )
    if pause_gap >= PAUSE_DURATION * 0.9:
        print("Test passed! Pause synchronized across DP ranks.")
    else:
        print(f"Test failed! Expected ~{PAUSE_DURATION}s gap, got {pause_gap:.3f}s")


if __name__ == "__main__":
    main()
