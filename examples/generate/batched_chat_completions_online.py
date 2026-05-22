# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Examples of batched chat completions via the vLLM OpenAI-compatible API.

The /v1/chat/completions/batch endpoint accepts ``messages`` as a list of
conversations.  Each conversation is processed independently and the response
contains one choice per conversation, indexed 0, 1, ..., N-1.

Start a server first, e.g.:
    vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000

Current limitations compared to /v1/chat/completions:
    - Streaming is not supported.
    - Tool use is not supported.
    - Beam search is not supported.
"""

import json
import os

import httpx

BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
BATCH_URL = f"{BASE_URL}/v1/chat/completions/batch"


def post_batch(payload: dict) -> dict:
    response = httpx.post(BATCH_URL, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def main() -> None:
    print("=== Example 1a: single conversation (standard endpoint) ===")
    response = httpx.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "What is the capital of Japan?"}],
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    for choice in data["choices"]:
        print(f"  [{choice['index']}] {choice['message']['content']}")

    print("\n=== Example 1b: batched plain text (2 conversations) ===")
    data = post_batch(
        {
            "model": MODEL,
            "messages": [
                [{"role": "user", "content": "What is the capital of France?"}],
                [{"role": "user", "content": "What is the capital of Japan?"}],
            ],
        }
    )
    for choice in data["choices"]:
        print(f"  [{choice['index']}] {choice['message']['content']}")

    print("\n=== Example 2: batch with regex constraint (yes|no) ===")
    data = post_batch(
        {
            "model": MODEL,
            "messages": [
                [{"role": "user", "content": "Is the sky blue? Answer yes or no."}],
                [{"role": "user", "content": "Is fire cold? Answer yes or no."}],
            ],
            "structured_outputs": {"regex": "(yes|no)"},
        }
    )
    for choice in data["choices"]:
        print(f"  [{choice['index']}] {choice['message']['content']}")

    print("\n=== Example 3: batch with json_schema ===")
    person_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Full name of the person"},
            "age": {"type": "integer", "description": "Age in years"},
        },
        "required": ["name", "age"],
    }
    data = post_batch(
        {
            "model": MODEL,
            "messages": [
                [
                    {
                        "role": "user",
                        "content": "Describe the person: name Alice, age 30.",
                    }
                ],
                [{"role": "user", "content": "Describe the person: name Bob, age 25."}],
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "person",
                    "strict": True,
                    "schema": person_schema,
                },
            },
        }
    )
    for choice in data["choices"]:
        person = json.loads(choice["message"]["content"])
        print(f"  [{choice['index']}] {person}")

    print("\n=== Example 4: batch book summaries ===")
    book_schema = {
        "type": "object",
        "properties": {
            "author": {
                "type": "string",
                "description": "Full name of the author",
            },
            "num_pages": {
                "type": "integer",
                "description": "Number of pages in the book",
            },
            "short_summary": {
                "type": "string",
                "description": "A one-sentence summary of the book",
            },
            "long_summary": {
                "type": "string",
                "description": (
                    "A detailed two to three sentence summary covering "
                    "the main themes and plot"
                ),
            },
        },
        "required": ["author", "num_pages", "short_summary", "long_summary"],
    }
    system_msg = {
        "role": "system",
        "content": (
            "You are a literary analyst. Extract structured information "
            "from book descriptions."
        ),
    }
    data = post_batch(
        {
            "model": MODEL,
            "messages": [
                [
                    system_msg,
                    {
                        "role": "user",
                        "content": (
                            "Extract information from this book: '1984' by George"
                            " Orwell, published in 1949, 328 pages. A dystopian"
                            " novel set in a totalitarian society ruled by Big"
                            " Brother, following Winston Smith as he secretly"
                            " rebels against the oppressive Party that surveils"
                            " and controls every aspect of life."
                        ),
                    },
                ],
                [
                    system_msg,
                    {
                        "role": "user",
                        "content": (
                            "Extract information from this book: 'The Hitchhiker's"
                            " Guide to the Galaxy' by Douglas Adams, published in"
                            " 1979, 193 pages. A comedic science fiction novel"
                            " following Arthur Dent, an ordinary Englishman who is"
                            " whisked off Earth moments before it is demolished to"
                            " make way for a hyperspace bypass, and his subsequent"
                            " absurd adventures across the universe."
                        ),
                    },
                ],
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "book_summary",
                    "strict": True,
                    "schema": book_schema,
                },
            },
        }
    )
    for choice in data["choices"]:
        book = json.loads(choice["message"]["content"])
        print(f"  [{choice['index']}] {book}")


if __name__ == "__main__":
    main()
