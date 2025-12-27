# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This script demonstrates how to use the vLLM API server to perform audio
transcription.
- Provide the model name via CLI (`--model Qwen/Qwen3-Omni-30B-A3B-Instruct`).
- Omit the model flag and let the script pick the first model returned by
  `/v1/models` (useful when the server is already serving a single model).

Example server start:
    vllm serve openai/whisper-large-v3 or vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct

Example client usage:
    python examples/online_serving/openai_transcription_client.py \\
    --base-url http://localhost:8000/v1 --api-key EMPTY \\
    --model Qwen/Qwen3-Omni-30B-A3B-Instruct

Requirements:
- vLLM with audio support
- openai Python SDK
- httpx for streaming support

The script performs:
1. Synchronous transcription using OpenAI-compatible API.
2. Streaming transcription using OpenAI-compatible API.
"""

import argparse
import asyncio

from openai import AsyncOpenAI, OpenAI

from vllm.assets.audio import AudioAsset


def sync_openai(audio_path: str, model: str, client: OpenAI):
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
        print("transcription result:", transcription.text)


async def stream_openai_response(audio_path: str, model: str, client: AsyncOpenAI):
    """
    Perform asynchronous transcription using OpenAI-compatible API.
    """
    print("\ntranscription result:", end=" ")
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible transcription client for vLLM."
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="vLLM server URL (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--api-key",
        default="EMPTY",
        help="API key for the server (default: EMPTY for local use)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model name to use. If omitted, the script will try to infer"
            "the first served model via /v1/models."
        ),
    )
    return parser.parse_args()


def resolve_model(client: OpenAI, user_model: str | None) -> str:
    if user_model:
        return user_model

    try:
        models = client.models.list()
        if not getattr(models, "data", None):
            raise RuntimeError("No models returned by /v1/models")
        discovered_model = models.data[0].id
        print(f"Using model returned by /v1/models: {discovered_model}")
        return discovered_model
    except Exception as exc:
        raise RuntimeError(
            "Unable to infer a model from /v1/models; please provide one with --model."
        ) from exc


def main():
    args = parse_args()
    mary_had_lamb = str(AudioAsset("mary_had_lamb").get_local_path())
    winning_call = str(AudioAsset("winning_call").get_local_path())

    # Modify OpenAI's API key and API base to use vLLM's API server.
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
    )

    model = resolve_model(client, args.model)
    sync_openai(mary_had_lamb, model, client)
    # Run the asynchronous function
    async_client = AsyncOpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
    )
    asyncio.run(stream_openai_response(winning_call, model, async_client))


if __name__ == "__main__":
    main()
