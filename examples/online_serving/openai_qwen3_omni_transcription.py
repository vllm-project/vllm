# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Minimal transcription example for Qwen3-Omni using the OpenAI-compatible API.

Server setup (one terminal):
    vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --port 8000 \
        --tensor-parallel-size 2
    # Adjust --tensor-parallel-size to match available GPUs.

Client usage (another terminal):
    python examples/online_serving/openai_qwen3_omni_transcription.py
"""

import asyncio

from openai import AsyncOpenAI, OpenAI

from vllm.assets.audio import AudioAsset

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


def sync_openai(audio_path: str, client: OpenAI) -> None:
    with open(audio_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            file=f,
            model=MODEL_ID,
            language="en",
            response_format="json",
            temperature=0.0,
        )
    print("sync transcription:", transcription.text)


async def stream_openai_response(audio_path: str, client: AsyncOpenAI) -> None:
    print("\nstreamed transcription:", end=" ")
    with open(audio_path, "rb") as f:
        async for chunk in await client.audio.transcriptions.create(
            file=f,
            model=MODEL_ID,
            language="en",
            response_format="text",
            temperature=0.0,
            stream=True,
        ):
            if chunk.choices:
                content = chunk.choices[0].get("delta", {}).get("content")
                if content:
                    print(content, end="", flush=True)
    print()


def main() -> None:
    sample_path = str(AudioAsset("winning_call").get_local_path())

    base_url = "http://localhost:8000/v1"

    # Point the OpenAI client at the local vLLM server.
    client = OpenAI(api_key="EMPTY", base_url=base_url)
    sync_openai(sample_path, client)

    async_client = AsyncOpenAI(api_key="EMPTY", base_url=base_url)
    asyncio.run(stream_openai_response(sample_path, async_client))


if __name__ == "__main__":
    main()
