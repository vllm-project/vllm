# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This script demonstrates how to use the vLLM API server to perform audio
transcription with the `openai/whisper-large-v3` model.

Before running this script, you must start the vLLM server with the following command:

    vllm serve openai/whisper-large-v3

Requirements:
- vLLM with audio support
- openai Python SDK
- httpx for streaming support

The script performs:
1. Synchronous transcription using OpenAI-compatible API.
2. Streaming transcription using raw HTTP request to the vLLM server.
"""

import asyncio
import json

import httpx
from openai import OpenAI

from vllm.assets.audio import AudioAsset

mary_had_lamb = AudioAsset("mary_had_lamb").get_local_path()
winning_call = AudioAsset("winning_call").get_local_path()

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


def sync_openai():
    """
    Perform synchronous transcription using OpenAI-compatible API.
    """
    with open(str(mary_had_lamb), "rb") as f:
        transcription = client.audio.transcriptions.create(
            file=f,
            model="openai/whisper-large-v3",
            language="en",
            response_format="json",
            temperature=0.0,
            # Additional sampling params not provided by OpenAI API.
            extra_body=dict(
                seed=4419,
                repetition_penalty=1.3,
            ),
        )
        print("transcription result:", transcription.text)


# OpenAI Transcription API client does not support streaming.
async def stream_openai_response():
    """
    Perform streaming transcription using vLLM's raw HTTP streaming API.
    """
    data = {
        "language": "en",
        "stream": True,
        "model": "openai/whisper-large-v3",
    }
    url = openai_api_base + "/audio/transcriptions"
    headers = {"Authorization": f"Bearer {openai_api_key}"}
    print("transcription result:", end=" ")
    async with httpx.AsyncClient() as client:
        with open(str(winning_call), "rb") as f:
            async with client.stream(
                "POST", url, files={"file": f}, data=data, headers=headers
            ) as response:
                async for line in response.aiter_lines():
                    # Each line is a JSON object prefixed with 'data: '
                    if line:
                        if line.startswith("data: "):
                            line = line[len("data: ") :]
                        # Last chunk, stream ends
                        if line.strip() == "[DONE]":
                            break
                        # Parse the JSON response
                        chunk = json.loads(line)
                        # Extract and print the content
                        content = chunk["choices"][0].get("delta", {}).get("content")
                        print(content, end="")
    print()  # Final newline after stream ends


def main():
    sync_openai()

    # Run the asynchronous function
    asyncio.run(stream_openai_response())


if __name__ == "__main__":
    main()
