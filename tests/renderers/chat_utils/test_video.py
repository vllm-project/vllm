# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.config import ModelConfig
from vllm.exceptions import VLLMValidationError
from vllm.renderers.chat_utils import parse_chat_messages, parse_chat_messages_async
from vllm.utils.serial_utils import tensor2base64

from .utils import (
    assert_mm_data_inputs,
    assert_mm_uuids,
)

QWEN25OMNI_MODEL_ID = "Qwen/Qwen2.5-Omni-7B"


@pytest.fixture(scope="function")
def qwen25omni_model_config_image_embeds():
    return ModelConfig(
        QWEN25OMNI_MODEL_ID,
        runner="generate",
        limit_mm_per_prompt={"image": 2},
        enable_mm_embeds=True,
    )


@pytest.fixture(scope="function")
def qwen25omni_model_config_video_embeds():
    return ModelConfig(
        QWEN25OMNI_MODEL_ID,
        runner="generate",
        limit_mm_per_prompt={"video": 2},
        enable_mm_embeds=True,
    )


@pytest.fixture(scope="function")
def qwen25omni_model_config_mm_interleaved():
    return ModelConfig(
        QWEN25OMNI_MODEL_ID,
        runner="generate",
        interleave_mm_strings=True,
        limit_mm_per_prompt={
            "image": 2,
            "audio": 1,
            "video": 1,
        },
    )


def test_parse_chat_messages_empty_video_embeds_with_uuid(
    qwen25omni_model_config_video_embeds,
):
    """Test video_embeds with UUID (no actual embeds data)."""
    uuid = "test-video-uuid-123"

    conversation, mm_data, mm_uuids = parse_chat_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this video"},
                    {"type": "video_embeds", "video_embeds": None, "uuid": uuid},
                ],
            }
        ],
        qwen25omni_model_config_video_embeds,
        content_format="string",
    )

    # Should have video in mm_data as None (UUID provided)
    assert mm_data is not None
    assert "video" in mm_data
    assert isinstance(mm_data["video"], list)
    assert len(mm_data["video"]) == 1
    assert mm_data["video"][0] is None

    # UUID should be recorded
    assert_mm_uuids(mm_uuids, 1, modality="video", expected_uuids=[uuid])


def test_parse_chat_messages_video_embeds_with_string(
    qwen25omni_model_config_video_embeds,
):
    """Test video_embeds with base64 string embedding data."""
    hidden_size = qwen25omni_model_config_video_embeds.get_inputs_embeds_size()
    video_embedding = torch.randn(1, 128, hidden_size)

    base64_video_embedding = tensor2base64(video_embedding)

    conversation, mm_data, mm_uuids = parse_chat_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this video"},
                    {
                        "type": "video_embeds",
                        "video_embeds": base64_video_embedding,
                    },
                ],
            }
        ],
        qwen25omni_model_config_video_embeds,
        content_format="string",
    )

    # Should have video embedding in mm_data (single tensor, not a list)
    assert mm_data is not None
    assert "video" in mm_data
    assert isinstance(mm_data["video"], torch.Tensor)
    assert mm_data["video"].shape == video_embedding.shape
    # No UUID provided
    assert mm_uuids is not None
    assert "video" in mm_uuids
    assert_mm_uuids(mm_uuids, 1, modality="video", expected_uuids=[None])


@pytest.mark.asyncio
async def test_parse_chat_messages_video_embeds_async(
    qwen25omni_model_config_video_embeds,
):
    """Test video_embeds with async futures."""
    hidden_size = qwen25omni_model_config_video_embeds.get_inputs_embeds_size()
    video_embedding = torch.randn(1, 128, hidden_size)

    base64_video_embedding = tensor2base64(video_embedding)

    conversation, mm_data, mm_uuids = await parse_chat_messages_async(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this video"},
                    {
                        "type": "video_embeds",
                        "video_embeds": base64_video_embedding,
                    },
                ],
            }
        ],
        qwen25omni_model_config_video_embeds,
        content_format="string",
    )

    # Should have video embedding in mm_data (single tensor, not a list)
    assert mm_data is not None
    assert "video" in mm_data
    assert isinstance(mm_data["video"], torch.Tensor)
    assert mm_data["video"].shape == video_embedding.shape
    # No UUID provided
    assert mm_uuids is not None
    assert "video" in mm_uuids
    assert_mm_uuids(mm_uuids, 1, modality="video", expected_uuids=[None])


def test_parse_chat_messages_video_and_video_embeds_mixed(
    qwen25omni_model_config_video_embeds,
    video_url,
):
    """Test that mixing raw video and video_embeds is rejected."""
    with pytest.raises(VLLMValidationError, match="Mixing raw video"):
        parse_chat_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "video_url", "video_url": {"url": video_url}},
                        {"type": "video_embeds", "video_embeds": None, "uuid": "x"},
                    ],
                }
            ],
            qwen25omni_model_config_video_embeds,
            content_format="string",
        )


@pytest.mark.parametrize("json_metadata", [False, True])
def test_parse_chat_messages_multiple_dict_image_embeds(
    qwen25omni_model_config_image_embeds,
    json_metadata,
):
    """Test that multiple dictionaries for image_embeds is handled without errors."""
    # Create two sample image embedding tensors
    batch_size = 2
    hidden_size = qwen25omni_model_config_image_embeds.get_inputs_embeds_size()
    image_embeds = torch.randn(batch_size * 220, hidden_size)
    image_grid_thw = torch.tensor([[1, 22, 40] for _ in range(batch_size)])

    conversation, mm_data, mm_uuids = parse_chat_messages(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_embeds",
                        "image_embeds": {
                            "image_embeds": tensor2base64(embeds),
                            "image_grid_thw": grid_thw.tolist()
                            if json_metadata
                            else tensor2base64(grid_thw),
                        },
                    }
                    for embeds, grid_thw in zip(
                        image_embeds.chunk(batch_size), image_grid_thw
                    )
                ]
                + [
                    {"type": "text", "text": "Describe these two images."},
                ],
            }
        ],
        qwen25omni_model_config_image_embeds,
        content_format="string",
    )

    # Verify conversation structure
    assert conversation == [
        {
            "role": "user",
            "content": "<|vision_start|><|IMAGE|><|vision_end|>\n"
            "<|vision_start|><|IMAGE|><|vision_end|>\nDescribe these two images.",
        }
    ]

    # Verify mm_data contains a dictionary of multi-embeddings
    assert mm_data is not None
    assert "image" in mm_data
    assert isinstance(mm_data["image"], dict)
    assert len(mm_data["image"]) == batch_size

    # Verify each embedding has the correct shape
    assert isinstance(mm_data["image"]["image_embeds"], torch.Tensor)
    assert mm_data["image"]["image_embeds"].shape == image_embeds.shape
    assert isinstance(mm_data["image"]["image_grid_thw"], torch.Tensor)
    assert mm_data["image"]["image_grid_thw"].shape == image_grid_thw.shape

    # Verify UUIDs (None since we didn't provide any)
    assert_mm_uuids(mm_uuids, batch_size, expected_uuids=[None, None])


def test_parse_chat_messages_multiple_modals_multiple_messages_interleave(
    qwen25omni_model_config_mm_interleaved,
    image_url,
    video_url,
    audio_url,
):
    conversation, mm_data, mm_uuids = parse_chat_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's on this image?"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "Now listen to this audio"},
                    {"type": "audio_url", "audio_url": {"url": audio_url}},
                ],
            },
            {"role": "assistant", "content": "Some stuff."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's on this image?"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "And what's in the video?"},
                    {"type": "video_url", "video_url": {"url": video_url}},
                ],
            },
        ],
        qwen25omni_model_config_mm_interleaved,
        content_format="string",
    )

    assert conversation == [
        {
            "role": "user",
            "content": "What's on this image?\n<|vision_start|><|IMAGE|><|vision_end|>"
            "\nNow listen to this audio\nAudio 1: <|audio_bos|><|AUDIO|><|audio_eos|>",
        },
        {"role": "assistant", "content": "Some stuff."},
        {
            "role": "user",
            "content": "What's on this image?\n<|vision_start|><|IMAGE|><|vision_end|>"
            "\nAnd what's in the video?\n<|vision_start|><|VIDEO|><|vision_end|>",
        },
    ]

    assert_mm_data_inputs(mm_data, {"image": 2, "video": 1, "audio": 1})
    assert_mm_uuids(mm_uuids, 2, modality="image", expected_uuids=[None, None])
    assert_mm_uuids(mm_uuids, 1, modality="video", expected_uuids=[None])
    assert_mm_uuids(mm_uuids, 1, modality="audio", expected_uuids=[None])


def test_parse_chat_messages_multiple_modals_with_uuids_multiple_messages_interleave(
    qwen25omni_model_config_mm_interleaved,
    image_url,
    video_url,
    audio_url,
):
    conversation, mm_data, mm_uuids = parse_chat_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's on this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                        "uuid": "image_123",
                    },
                    {"type": "text", "text": "Now listen to this audio"},
                    {
                        "type": "audio_url",
                        "audio_url": {"url": audio_url},
                        "uuid": "audio_123",
                    },
                ],
            },
            {"role": "assistant", "content": "Some stuff."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's on this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                        "uuid": "image_123",
                    },
                    {"type": "text", "text": "And what's in the video?"},
                    {
                        "type": "video_url",
                        "video_url": {"url": video_url},
                        "uuid": "video_123",
                    },
                ],
            },
        ],
        qwen25omni_model_config_mm_interleaved,
        content_format="string",
    )

    assert conversation == [
        {
            "role": "user",
            "content": "What's on this image?\n<|vision_start|><|IMAGE|><|vision_end|>"
            "\nNow listen to this audio\nAudio 1: <|audio_bos|><|AUDIO|><|audio_eos|>",
        },
        {"role": "assistant", "content": "Some stuff."},
        {
            "role": "user",
            "content": "What's on this image?\n<|vision_start|><|IMAGE|><|vision_end|>"
            "\nAnd what's in the video?\n<|vision_start|><|VIDEO|><|vision_end|>",
        },
    ]

    assert_mm_data_inputs(mm_data, {"image": 2, "video": 1, "audio": 1})
    assert_mm_uuids(
        mm_uuids, 2, modality="image", expected_uuids=["image_123", "image_123"]
    )
    assert_mm_uuids(mm_uuids, 1, modality="video", expected_uuids=["video_123"])
    assert_mm_uuids(mm_uuids, 1, modality="audio", expected_uuids=["audio_123"])


def test_parse_chat_messages_multiple_modals_with_uuids_multiple_empty_media_messages_interleave(  # noqa: E501
    qwen25omni_model_config_mm_interleaved,
    image_url,
    video_url,
    audio_url,
):
    conversation, mm_data, mm_uuids = parse_chat_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's on this image?"},
                    {
                        "type": "image_url",
                        "image_url": None,
                        "uuid": "image_123",
                    },
                    {"type": "text", "text": "Now listen to this audio"},
                    {
                        "type": "audio_url",
                        "audio_url": None,
                        "uuid": "audio_123",
                    },
                ],
            },
            {"role": "assistant", "content": "Some stuff."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's on this image?"},
                    {
                        "type": "image_url",
                        "image_url": None,
                        "uuid": "image_123",
                    },
                    {"type": "text", "text": "And what's in the video?"},
                    {
                        "type": "video_url",
                        "video_url": None,
                        "uuid": "video_123",
                    },
                ],
            },
        ],
        qwen25omni_model_config_mm_interleaved,
        content_format="string",
    )

    assert conversation == [
        {
            "role": "user",
            "content": "What's on this image?\n<|vision_start|><|IMAGE|><|vision_end|>"
            "\nNow listen to this audio\nAudio 1: <|audio_bos|><|AUDIO|><|audio_eos|>",
        },
        {"role": "assistant", "content": "Some stuff."},
        {
            "role": "user",
            "content": "What's on this image?\n<|vision_start|><|IMAGE|><|vision_end|>"
            "\nAnd what's in the video?\n<|vision_start|><|VIDEO|><|vision_end|>",
        },
    ]

    assert_mm_data_inputs(
        mm_data,
        {"image": 2, "video": 1, "audio": 1},
        skipped_media_indices={"image": [0, 1], "video": [0], "audio": [0]},
    )
    assert_mm_uuids(
        mm_uuids, 2, modality="image", expected_uuids=["image_123", "image_123"]
    )
    assert_mm_uuids(mm_uuids, 1, modality="video", expected_uuids=["video_123"])
    assert_mm_uuids(mm_uuids, 1, modality="audio", expected_uuids=["audio_123"])


def test_parse_chat_messages_multiple_modals_with_partial_uuids_multiple_messages_interleave(  # noqa: E501
    qwen25omni_model_config_mm_interleaved,
    image_url,
    video_url,
    audio_url,
):
    conversation, mm_data, mm_uuids = parse_chat_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's on this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                        "uuid": "image_123",
                    },
                    {"type": "text", "text": "Now listen to this audio"},
                    {"type": "audio_url", "audio_url": {"url": audio_url}},
                ],
            },
            {"role": "assistant", "content": "Some stuff."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's on this image?"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "And what's in the video?"},
                    {
                        "type": "video_url",
                        "video_url": {"url": video_url},
                        "uuid": "video_123",
                    },
                ],
            },
        ],
        qwen25omni_model_config_mm_interleaved,
        content_format="string",
    )

    assert conversation == [
        {
            "role": "user",
            "content": "What's on this image?\n<|vision_start|><|IMAGE|><|vision_end|>"
            "\nNow listen to this audio\nAudio 1: <|audio_bos|><|AUDIO|><|audio_eos|>",
        },
        {"role": "assistant", "content": "Some stuff."},
        {
            "role": "user",
            "content": "What's on this image?\n<|vision_start|><|IMAGE|><|vision_end|>"
            "\nAnd what's in the video?\n<|vision_start|><|VIDEO|><|vision_end|>",
        },
    ]

    assert_mm_data_inputs(mm_data, {"image": 2, "video": 1, "audio": 1})
    assert_mm_uuids(mm_uuids, 2, modality="image", expected_uuids=["image_123", None])
    assert_mm_uuids(mm_uuids, 1, modality="video", expected_uuids=["video_123"])
    assert_mm_uuids(mm_uuids, 1, modality="audio", expected_uuids=[None])
