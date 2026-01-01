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

import argparse
import asyncio

from openai import AsyncOpenAI, OpenAI

from vllm.assets.audio import AudioAsset


def sync_openai(audio_path: str, client: OpenAI, model: str):
    """
    Perform synchronous transcription using OpenAI-compatible API.
    """
    with open(audio_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            file=f,
            model=model,
            language="en",
            response_format="json",
            temperature=0.0,
            # Additional sampling params not provided by OpenAI API.
            extra_body=dict(
                seed=4419,
                repetition_penalty=1.3,
            ),
        )
        print("transcription result [sync]:", transcription.text)


async def stream_openai_response(audio_path: str, client: AsyncOpenAI, model: str):
    """
    Perform asynchronous transcription using OpenAI-compatible API.
    """
    print("\ntranscription result [stream]:", end=" ")
    with open(audio_path, "rb") as f:
        transcription = await client.audio.transcriptions.create(
            file=f,
            model=model,
            language="en",
            response_format="json",
            temperature=0.0,
            # Additional sampling params not provided by OpenAI API.
            extra_body=dict(
                seed=420,
                top_p=0.6,
            ),
            stream=True,
        )
        async for chunk in transcription:
            if chunk.choices:
                content = chunk.choices[0].get("delta", {}).get("content")
                print(content, end="", flush=True)

    print()  # Final newline after stream ends


def stream_api_response(audio_path: str, model: str, openai_api_base: str):
    """
    Perform streaming transcription using raw HTTP requests to the vLLM API server.
    """
    import json
    import os

    import requests

    api_url = f"{openai_api_base}/audio/transcriptions"
    headers = {"User-Agent": "Transcription-Client"}
    with open(audio_path, "rb") as f:
        files = {"file": (os.path.basename(audio_path), f)}
        data = {
            "stream": "true",
            "model": model,
            "language": "en",
            "response_format": "json",
        }

        print("\ntranscription result [stream]:", end=" ")
        response = requests.post(
            api_url, headers=headers, files=files, data=data, stream=True
        )
        for chunk in response.iter_lines(
            chunk_size=8192, decode_unicode=False, delimiter=b"\n"
        ):
            if chunk:
                data = chunk[len("data: ") :]
                data = json.loads(data.decode("utf-8"))
                data = data["choices"][0]
                delta = data["delta"]["content"]
                print(delta, end="", flush=True)

                finish_reason = data.get("finish_reason")
                if finish_reason is not None:
                    print(f"\n[Stream finished reason: {finish_reason}]")
                    break


def main(args):
    mary_had_lamb = str(AudioAsset("mary_had_lamb").get_local_path())
    winning_call = str(AudioAsset("winning_call").get_local_path())

    # Modify OpenAI's API key and API base to use vLLM's API server.
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    model = client.models.list().data[0].id
    print(f"Using model: {model}")

    # Run the synchronous function
    sync_openai(args.audio_path if args.audio_path else mary_had_lamb, client, model)

    # Run the asynchronous function
    if "openai" in model:
        client = AsyncOpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        asyncio.run(
            stream_openai_response(
                args.audio_path if args.audio_path else winning_call, client, model
            )
        )
    else:
        stream_api_response(
            args.audio_path if args.audio_path else winning_call,
            model,
            openai_api_base,
        )


if __name__ == "__main__":
    # setup argparser
    parser = argparse.ArgumentParser(
        description="OpenAI Transcription Client using vLLM API Server"
    )
    parser.add_argument(
        "--audio_path",
        type=str,
        default=None,
        help="The path to the audio file to transcribe.",
    )
    args = parser.parse_args()
    main(args)
