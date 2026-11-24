# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from http import HTTPStatus
from io import BytesIO

import pytest
import requests
import torch

import vllm.envs as envs
from tests.multimodal.utils import create_long_gop_video
from vllm.config import ModelConfig
from vllm.connections import MediaFetchDeadlineExceededError
from vllm.entrypoints.serve.exception_handling.error_response import (
    create_error_response,
)
from vllm.exceptions import VLLMUnprocessableEntityError, VLLMValidationError
from vllm.renderers.chat_utils import parse_chat_messages, parse_chat_messages_async
from vllm.renderers.chat_utils.types import (
    ChatCompletionContentPartAudioParam,
    ChatCompletionContentPartVideoParam,
)
from vllm.utils.serial_utils import tensor2base64

from .utils import (
    assert_mm_data_inputs,
    assert_mm_uuids,
)

QWEN25OMNI_MODEL_ID = "Qwen/Qwen2.5-Omni-7B"


def test_parse_chat_messages_checks_video_audio_limit_before_url_loading():
    model_config = ModelConfig(
        QWEN25OMNI_MODEL_ID,
        runner="generate",
        limit_mm_per_prompt={"video": 1, "audio": 0},
        allowed_media_domains=["allowed.example"],
    )

    # This URL cannot reach the network even if the connector is called.
    with pytest.raises(VLLMValidationError, match="At most 0 audio"):
        parse_chat_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {"url": "https://other.example/video.mp4"},
                        }
                    ],
                }
            ],
            model_config,
            content_format="string",
            mm_processor_kwargs={"load_audio_from_video": True},
        )


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


@pytest.mark.parametrize(
    ("parameter", "load_audio_from_video"),
    [("audio_url", False), ("video_url", False), ("video_url", True)],
    ids=["audio", "video", "video-soundtrack"],
)
def test_parse_chat_messages_media_deadline_reports_url_field(
    qwen25omni_model_config_mm_interleaved: ModelConfig,
    monkeypatch: pytest.MonkeyPatch,
    parameter: str,
    load_audio_from_video: bool,
) -> None:
    part: ChatCompletionContentPartAudioParam | ChatCompletionContentPartVideoParam
    if parameter == "audio_url":
        part = {
            "type": "audio_url",
            "audio_url": {"url": "https://media.invalid/audio"},
        }
    else:
        part = {
            "type": "video_url",
            "video_url": {"url": "https://media.invalid/video"},
        }

    video_response = None
    if load_audio_from_video:
        # Decode a valid video first, then time out fetching its soundtrack.
        video_response = requests.Response()
        video_response.status_code = HTTPStatus.OK
        video_response.raw = BytesIO(create_long_gop_video(num_frames=2))

    def fetch(*args: object, **kwargs: object) -> requests.Response:
        nonlocal video_response
        if video_response is not None:
            response, video_response = video_response, None
            return response
        raise MediaFetchDeadlineExceededError("Media fetch exceeded its time limit")

    monkeypatch.setattr(envs, "VLLM_MEDIA_CACHE", "")
    monkeypatch.setattr(requests.Session, "send", fetch)

    with pytest.raises(VLLMUnprocessableEntityError) as exc_info:
        parse_chat_messages(
            [{"role": "user", "content": [part]}],
            qwen25omni_model_config_mm_interleaved,
            content_format="string",
            mm_processor_kwargs={"load_audio_from_video": load_audio_from_video},
        )

    error = create_error_response(exc_info.value).error
    assert error.code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert error.param == parameter
    assert "Media fetch exceeded its time limit" in error.message


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


@pytest.mark.parametrize("count", [0, 1])
def test_parse_chat_messages_video_embeds_disabled_modality(count):
    model_config = ModelConfig(
        QWEN25OMNI_MODEL_ID,
        runner="generate",
        limit_mm_per_prompt={"video": {"count": count, "allow_embeds": False}},
        enable_mm_embeds=True,
    )

    with pytest.raises(VLLMValidationError, match="`video_embeds` inputs are disabled"):
        parse_chat_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "video_embeds", "video_embeds": None, "uuid": "x"},
                    ],
                }
            ],
            model_config,
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


def _hf_processor_unavailable(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Models whose HF processor cannot be built on the frontend, as in the
    scoring unit tests. Returns the (mutable) load count."""
    from vllm.multimodal.processing import context

    calls = [0]

    def cached_processor_from_config(*args, **kwargs):
        calls[0] += 1
        raise OSError("Can't load image processor")

    monkeypatch.setattr(
        context, "cached_processor_from_config", cached_processor_from_config
    )
    return calls


def _image_messages(image_url):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": "Describe both."},
            ],
        }
    ]


@pytest.fixture(scope="function")
def qwen3vl_model_config():
    return ModelConfig(
        "Qwen/Qwen3-VL-4B-Instruct",
        runner="generate",
        limit_mm_per_prompt={"image": 2, "video": 1},
    )


def test_parse_chat_messages_without_video_never_builds_the_hf_processor(
    qwen3vl_model_config,
    image_url,
    monkeypatch,
):
    calls = _hf_processor_unavailable(monkeypatch)

    _, mm_data, _ = parse_chat_messages(
        _image_messages(image_url),
        qwen3vl_model_config,
        content_format="string",
    )

    assert_mm_data_inputs(mm_data, {"image": 2})
    assert calls == [0]


def test_parse_chat_messages_video_passes_the_frame_caps_once(
    image_url,
    video_url,
    monkeypatch,
):
    from vllm.multimodal.media.connector import MediaConnector
    from vllm.multimodal.processing import BaseProcessingInfo

    get_video_frame_limits = BaseProcessingInfo.get_video_frame_limits
    limits = []

    def counting_get_video_frame_limits(self):
        limits.append(get_video_frame_limits(self))
        return limits[-1]

    monkeypatch.setattr(
        BaseProcessingInfo, "get_video_frame_limits", counting_get_video_frame_limits
    )
    seen = []

    def fetch_video(self, video_url, **kwargs):
        seen.append(kwargs["video_io_defaults"])
        return "video"

    monkeypatch.setattr(MediaConnector, "fetch_video", fetch_video)

    _, mm_data, _ = parse_chat_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "video_url", "video_url": {"url": video_url}},
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "Describe."},
                ],
            },
            {"role": "assistant", "content": "Some stuff."},
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": video_url}},
                    {"type": "text", "text": "And this?"},
                ],
            },
        ],
        ModelConfig(
            QWEN25OMNI_MODEL_ID,
            runner="generate",
            limit_mm_per_prompt={"image": 2, "video": 2},
        ),
        content_format="string",
    )

    assert_mm_data_inputs(mm_data, {"image": 2, "video": 2})
    assert len(limits) == 1 and limits[0]["max_frames"] > 0
    assert seen == limits * 2
