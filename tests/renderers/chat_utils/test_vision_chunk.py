# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config import ModelConfig
from vllm.inputs import MultiModalDataDict
from vllm.renderers.chat_utils import (
    ChatCompletionMessageParam,
    parse_chat_messages,
    parse_chat_messages_async,
)

from .utils import assert_mm_uuids

KIMI_K2_5_MODEL_ID = "moonshotai/Kimi-K2.5"


@pytest.fixture(scope="function")
def kimi_k2_5_model_config():
    return ModelConfig(
        KIMI_K2_5_MODEL_ID,
        runner="generate",
        trust_remote_code=True,
        limit_mm_per_prompt={
            "image": 2,
        },
    )


def _assert_mm_data_is_vision_chunk_input(
    mm_data: MultiModalDataDict | None,
    vision_chunk_count: int,
) -> None:
    assert mm_data is not None
    assert set(mm_data.keys()) == {"vision_chunk"}

    vision_chunk_data = mm_data.get("vision_chunk")
    assert vision_chunk_data is not None

    assert (
        isinstance(vision_chunk_data, list)
        and len(vision_chunk_data) == vision_chunk_count
    )


def test_parse_chat_messages_image_vision_chunk(
    kimi_k2_5_model_config,
    image_url,
):
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this image."},
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
            ],
        }
    ]

    conversation, mm_data, mm_uuids = parse_chat_messages(
        messages,
        kimi_k2_5_model_config,
        content_format="string",
    )

    placeholder = "<|media_begin|>image<|media_content|><|media_pad|><|media_end|>"
    expected_conversation = [
        {
            "role": "user",
            "content": f"{placeholder}\nAnalyze this image.",
        }
    ]

    assert conversation == expected_conversation
    _assert_mm_data_is_vision_chunk_input(mm_data, 1)
    assert_mm_uuids(mm_uuids, 1, expected_uuids=[None], modality="vision_chunk")


def test_parse_chat_messages_video_vision_chunk(
    kimi_k2_5_model_config,
    video_url,
):
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this video."},
                {
                    "type": "video_url",
                    "video_url": {"url": video_url},
                },
            ],
        }
    ]

    conversation, mm_data, mm_uuids = parse_chat_messages(
        messages,
        kimi_k2_5_model_config,
        content_format="string",
    )

    placeholder = "<|kimi_k25_video_placeholder|>"
    expected_conversation = [
        {
            "role": "user",
            "content": f"{placeholder}\nAnalyze this video.",
        }
    ]

    assert conversation == expected_conversation
    _assert_mm_data_is_vision_chunk_input(mm_data, 1)
    assert_mm_uuids(mm_uuids, 1, expected_uuids=[None], modality="vision_chunk")


def test_parse_chat_messages_image_vision_chunk_with_uuid(
    kimi_k2_5_model_config,
    image_url,
):
    image_uuid = "image_123"
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this image."},
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                    "uuid": image_uuid,
                },
            ],
        }
    ]

    conversation, mm_data, mm_uuids = parse_chat_messages(
        messages,
        kimi_k2_5_model_config,
        content_format="string",
    )

    placeholder = "<|media_begin|>image<|media_content|><|media_pad|><|media_end|>"
    expected_conversation = [
        {
            "role": "user",
            "content": f"{placeholder}\nAnalyze this image.",
        }
    ]

    assert conversation == expected_conversation
    _assert_mm_data_is_vision_chunk_input(mm_data, 1)
    assert_mm_uuids(mm_uuids, 1, expected_uuids=[image_uuid], modality="vision_chunk")


def test_parse_chat_messages_video_vision_chunk_with_uuid(
    kimi_k2_5_model_config,
    video_url,
):
    video_uuid = "video_456"
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this video."},
                {
                    "type": "video_url",
                    "video_url": {"url": video_url},
                    "uuid": video_uuid,
                },
            ],
        }
    ]

    conversation, mm_data, mm_uuids = parse_chat_messages(
        messages,
        kimi_k2_5_model_config,
        content_format="string",
    )

    placeholder = "<|kimi_k25_video_placeholder|>"
    expected_conversation = [
        {
            "role": "user",
            "content": f"{placeholder}\nAnalyze this video.",
        }
    ]

    assert conversation == expected_conversation
    _assert_mm_data_is_vision_chunk_input(mm_data, 1)
    assert_mm_uuids(mm_uuids, 1, expected_uuids=[video_uuid], modality="vision_chunk")


def test_parse_chat_messages_mixed_vision_chunk(
    kimi_k2_5_model_config,
    image_url,
    video_url,
):
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this image and video."},
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
                {
                    "type": "video_url",
                    "video_url": {"url": video_url},
                },
            ],
        }
    ]

    conversation, mm_data, mm_uuids = parse_chat_messages(
        messages,
        kimi_k2_5_model_config,
        content_format="string",
    )

    image_placeholder = (
        "<|media_begin|>image<|media_content|><|media_pad|><|media_end|>"
    )
    video_placeholder = "<|kimi_k25_video_placeholder|>"
    expected_conversation = [
        {
            "role": "user",
            "content": (
                f"{image_placeholder}\n{video_placeholder}\n"
                "Analyze this image and video."
            ),
        }
    ]

    assert conversation == expected_conversation
    _assert_mm_data_is_vision_chunk_input(mm_data, 2)
    assert_mm_uuids(mm_uuids, 2, expected_uuids=[None, None], modality="vision_chunk")


def test_parse_chat_messages_mixed_vision_chunk_with_uuid(
    kimi_k2_5_model_config,
    image_url,
    video_url,
):
    image_uuid = "image_123"
    video_uuid = "video_456"
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this image and video."},
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                    "uuid": image_uuid,
                },
                {
                    "type": "video_url",
                    "video_url": {"url": video_url},
                    "uuid": video_uuid,
                },
            ],
        }
    ]

    conversation, mm_data, mm_uuids = parse_chat_messages(
        messages,
        kimi_k2_5_model_config,
        content_format="string",
    )

    image_placeholder = (
        "<|media_begin|>image<|media_content|><|media_pad|><|media_end|>"
    )
    video_placeholder = "<|kimi_k25_video_placeholder|>"
    expected_conversation = [
        {
            "role": "user",
            "content": (
                f"{image_placeholder}\n{video_placeholder}\n"
                "Analyze this image and video."
            ),
        }
    ]

    assert conversation == expected_conversation
    _assert_mm_data_is_vision_chunk_input(mm_data, 2)
    assert_mm_uuids(
        mm_uuids, 2, expected_uuids=[image_uuid, video_uuid], modality="vision_chunk"
    )


@pytest.mark.asyncio
async def test_parse_chat_messages_mixed_vision_chunk_async(
    kimi_k2_5_model_config,
    image_url,
    video_url,
):
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this image and video."},
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
                {
                    "type": "video_url",
                    "video_url": {"url": video_url},
                },
            ],
        }
    ]

    conversation, mm_data, mm_uuids = await parse_chat_messages_async(
        messages,
        kimi_k2_5_model_config,
        content_format="string",
    )

    image_placeholder = (
        "<|media_begin|>image<|media_content|><|media_pad|><|media_end|>"
    )
    video_placeholder = "<|kimi_k25_video_placeholder|>"
    expected_conversation = [
        {
            "role": "user",
            "content": (
                f"{image_placeholder}\n{video_placeholder}\n"
                "Analyze this image and video."
            ),
        }
    ]

    assert conversation == expected_conversation
    _assert_mm_data_is_vision_chunk_input(mm_data, 2)
    assert_mm_uuids(mm_uuids, 2, expected_uuids=[None, None], modality="vision_chunk")


@pytest.mark.asyncio
async def test_parse_chat_messages_mixed_vision_chunk_with_uuid_async(
    kimi_k2_5_model_config,
    image_url,
    video_url,
):
    image_uuid = "image_123"
    video_uuid = "video_456"
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this image and video."},
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                    "uuid": image_uuid,
                },
                {
                    "type": "video_url",
                    "video_url": {"url": video_url},
                    "uuid": video_uuid,
                },
            ],
        }
    ]

    conversation, mm_data, mm_uuids = await parse_chat_messages_async(
        messages,
        kimi_k2_5_model_config,
        content_format="string",
    )

    image_placeholder = (
        "<|media_begin|>image<|media_content|><|media_pad|><|media_end|>"
    )
    video_placeholder = "<|kimi_k25_video_placeholder|>"
    expected_conversation = [
        {
            "role": "user",
            "content": (
                f"{image_placeholder}\n{video_placeholder}\n"
                "Analyze this image and video."
            ),
        }
    ]

    assert conversation == expected_conversation
    _assert_mm_data_is_vision_chunk_input(mm_data, 2)
    assert_mm_uuids(
        mm_uuids, 2, expected_uuids=[image_uuid, video_uuid], modality="vision_chunk"
    )


@pytest.mark.asyncio
async def test_parse_chat_messages_image_vision_chunk_async(
    kimi_k2_5_model_config,
    image_url,
):
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this image."},
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
            ],
        }
    ]

    conversation, mm_data, mm_uuids = await parse_chat_messages_async(
        messages,
        kimi_k2_5_model_config,
        content_format="string",
    )

    placeholder = "<|media_begin|>image<|media_content|><|media_pad|><|media_end|>"
    expected_conversation = [
        {
            "role": "user",
            "content": f"{placeholder}\nAnalyze this image.",
        }
    ]

    assert conversation == expected_conversation
    _assert_mm_data_is_vision_chunk_input(mm_data, 1)
    assert_mm_uuids(mm_uuids, 1, expected_uuids=[None], modality="vision_chunk")


@pytest.mark.asyncio
async def test_parse_chat_messages_video_vision_chunk_async(
    kimi_k2_5_model_config,
    video_url,
):
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this video."},
                {
                    "type": "video_url",
                    "video_url": {"url": video_url},
                },
            ],
        }
    ]

    conversation, mm_data, mm_uuids = await parse_chat_messages_async(
        messages,
        kimi_k2_5_model_config,
        content_format="string",
    )

    placeholder = "<|kimi_k25_video_placeholder|>"
    expected_conversation = [
        {
            "role": "user",
            "content": f"{placeholder}\nAnalyze this video.",
        }
    ]

    assert conversation == expected_conversation
    _assert_mm_data_is_vision_chunk_input(mm_data, 1)
    assert_mm_uuids(mm_uuids, 1, expected_uuids=[None], modality="vision_chunk")


@pytest.mark.asyncio
async def test_parse_chat_messages_image_vision_chunk_with_uuid_async(
    kimi_k2_5_model_config,
    image_url,
):
    image_uuid = "image_123"
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this image."},
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                    "uuid": image_uuid,
                },
            ],
        }
    ]

    conversation, mm_data, mm_uuids = await parse_chat_messages_async(
        messages,
        kimi_k2_5_model_config,
        content_format="string",
    )

    placeholder = "<|media_begin|>image<|media_content|><|media_pad|><|media_end|>"
    expected_conversation = [
        {
            "role": "user",
            "content": f"{placeholder}\nAnalyze this image.",
        }
    ]

    assert conversation == expected_conversation
    _assert_mm_data_is_vision_chunk_input(mm_data, 1)
    assert_mm_uuids(mm_uuids, 1, expected_uuids=[image_uuid], modality="vision_chunk")


@pytest.mark.asyncio
async def test_parse_chat_messages_video_vision_chunk_with_uuid_async(
    kimi_k2_5_model_config,
    video_url,
):
    video_uuid = "video_456"
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this video."},
                {
                    "type": "video_url",
                    "video_url": {"url": video_url},
                    "uuid": video_uuid,
                },
            ],
        }
    ]

    conversation, mm_data, mm_uuids = await parse_chat_messages_async(
        messages,
        kimi_k2_5_model_config,
        content_format="string",
    )

    placeholder = "<|kimi_k25_video_placeholder|>"
    expected_conversation = [
        {
            "role": "user",
            "content": f"{placeholder}\nAnalyze this video.",
        }
    ]

    assert conversation == expected_conversation
    _assert_mm_data_is_vision_chunk_input(mm_data, 1)
    assert_mm_uuids(mm_uuids, 1, expected_uuids=[video_uuid], modality="vision_chunk")
