# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Async Python client for OpenAI Chat Completion using vLLM API server

NOTE: start a supported chat completion model server with `vllm serve`, e.g.
    vllm serve meta-llama/Llama-2-7b-chat-hf

Usage:
    # Single request
    python openai_chat_async.py

    # Streaming
    python openai_chat_async.py --stream

    # Concurrent requests
    python openai_chat_async.py --concurrency 10

    # Combined
    python openai_chat_async.py --stream --concurrency 20
"""

import argparse
import asyncio
import time

from openai import AsyncOpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8080/v1"

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "How to learn english" * 1000},
]


def parse_args():
    parser = argparse.ArgumentParser(description="Async client for vLLM API server")
    parser.add_argument(
        "--stream", action="store_true", help="Enable streaming response"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent requests (default: 1)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=openai_api_key,
        help="OpenAI API key (default: EMPTY)",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=openai_api_base,
        help="OpenAI API base URL (default: http://localhost:8080/v1)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=5000,
        help="Max completion tokens (default: 5000)",
    )
    return parser.parse_args()


async def chat_completion(
    client: AsyncOpenAI,
    model: str,
    request_id: int,
    stream: bool = False,
    temperature: float = 0.0,
    max_tokens: int = 5000,
) -> None:
    """Send a single async chat completion request."""
    start = time.perf_counter()
    print(f"[Request {request_id}] Starting...")

    chat_response = await client.chat.completions.create(
        messages=messages,
        model=model,
        stream=stream,
        temperature=temperature,
        max_completion_tokens=max_tokens,
    )

    elapsed = time.perf_counter() - start

    print(f"\n{'=' * 60}")
    print(f"[Request {request_id}] Results (elapsed: {elapsed:.2f}s):")
    print(f"{'=' * 60}")

    if stream:
        full_content = []
        async for chunk in chat_response:
            delta = chunk.choices[0].delta.content
            if delta:
                full_content.append(delta)
                print(delta, flush=True, end="")
        print()  # newline after streaming
        total_elapsed = time.perf_counter() - start
        print(
            f"[Request {request_id}] "
            f"Stream done, total: {total_elapsed:.2f}s, "
            f"tokens: {len(full_content)} chunks"
        )
    else:
        result = chat_response.choices[0].message.content
        usage = chat_response.usage
        print(result)
        print(
            f"[Request {request_id}] "
            f"Usage: prompt_tokens={usage.prompt_tokens}, "
            f"completion_tokens={usage.completion_tokens}, "
            f"total_tokens={usage.total_tokens}"
        )

    print(f"{'=' * 60}\n")


async def main(args):
    client = AsyncOpenAI(
        api_key=args.api_key,
        base_url=args.api_base,
    )

    # Fetch available models
    models = await client.models.list()
    model = models.data[0].id
    print(f"Using model: {model}")
    print(f"Concurrency: {args.concurrency}, Stream: {args.stream}")
    print(f"{'=' * 60}")

    overall_start = time.perf_counter()

    # Launch concurrent requests
    tasks = [
        asyncio.create_task(
            chat_completion(
                client=client,
                model=model,
                request_id=i,
                stream=args.stream,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        )
        for i in range(args.concurrency)
    ]

    await asyncio.gather(*tasks)

    overall_elapsed = time.perf_counter() - overall_start
    print(f"{'=' * 60}")
    print(f"All {args.concurrency} requests completed in {overall_elapsed:.2f}s")
    print(
        f"Average: {overall_elapsed / args.concurrency:.2f}s per request "
        f"(wall clock: {overall_elapsed:.2f}s)"
    )


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
