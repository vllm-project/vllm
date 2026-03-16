# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
import json

import openai
import pytest
import pytest_asyncio

from ...conftest import VideoTestAssets
from ...utils import RemoteOpenAIServer

MODEL_NAME = "Qwen/Qwen2.5-Omni-3B"


@pytest.fixture
def server():
    args = [
        "--max-model-len",
        "18432",
        "--enforce-eager",
        "--limit-mm-per-prompt",
        json.dumps({"audio": 3, "video": 3}),
    ]

    with RemoteOpenAIServer(
        MODEL_NAME,
        args,
    ) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.core_model
@pytest.mark.asyncio
async def test_online_audio_in_video(
    client: openai.AsyncOpenAI, video_assets: VideoTestAssets
):
    """Test video input with `audio_in_video=True`"""

    # we don't use video_urls above because they missed audio stream.
    video_path = video_assets[0].video_path
    with open(video_path, "rb") as f:
        video_base64 = base64.b64encode(f.read()).decode("utf-8")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this video?"},
                {
                    "type": "video_url",
                    "video_url": {"url": f"data:video/mp4;base64,{video_base64}"},
                },
            ],
        }
    ]

    # multi-turn to test mm processor cache as well
    for _ in range(2):
        chat_completion = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=16,
            extra_body={
                "mm_processor_kwargs": {
                    "use_audio_in_video": True,
                }
            },
        )

        assert len(chat_completion.choices) == 1
        choice = chat_completion.choices[0]
        assert choice.finish_reason == "length"


@pytest.mark.core_model
@pytest.mark.asyncio
async def test_online_audio_in_video_multi_videos(
    client: openai.AsyncOpenAI, video_assets: VideoTestAssets
):
    """Test multi-video input with `audio_in_video=True`"""

    # we don't use video_urls above because they missed audio stream.
    video_path = video_assets[0].video_path
    with open(video_path, "rb") as f:
        video_base64 = base64.b64encode(f.read()).decode("utf-8")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in these two videos?"},
                {
                    "type": "video_url",
                    "video_url": {"url": f"data:video/mp4;base64,{video_base64}"},
                },
                {
                    "type": "video_url",
                    "video_url": {"url": f"data:video/mp4;base64,{video_base64}"},
                },
            ],
        }
    ]

    # multi-turn to test mm processor cache as well
    for _ in range(2):
        chat_completion = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=16,
            extra_body={
                "mm_processor_kwargs": {
                    "use_audio_in_video": True,
                }
            },
        )

        assert len(chat_completion.choices) == 1
        choice = chat_completion.choices[0]
        assert choice.finish_reason == "length"


@pytest.mark.core_model
@pytest.mark.asyncio
async def test_online_audio_in_video_interleaved(
    client: openai.AsyncOpenAI, video_assets: VideoTestAssets
):
    """Test interleaved video/audio input with `audio_in_video=True`"""

    # we don't use video_urls above because they missed audio stream.
    video_path = video_assets[0].video_path
    with open(video_path, "rb") as f:
        video_base64 = base64.b64encode(f.read()).decode("utf-8")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in these two videos?"},
                {
                    "type": "video_url",
                    "video_url": {"url": f"data:video/mp4;base64,{video_base64}"},
                },
                {
                    "type": "audio_url",
                    "audio_url": {"url": f"data:audio/mp4;base64,{video_base64}"},
                },
                {
                    "type": "video_url",
                    "video_url": {"url": f"data:video/mp4;base64,{video_base64}"},
                },
            ],
        }
    ]
    with pytest.raises(
        openai.BadRequestError,
        match="use_audio_in_video requires equal number of audio and video items",
    ):
        await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=16,
            extra_body={
                "mm_processor_kwargs": {
                    "use_audio_in_video": True,
                }
            },
        )
