# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import subprocess
from typing import Any

import pybase64 as base64
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


@pytest.fixture(scope="module")
def qwen25omni_model_config_audio_video():
    return ModelConfig(
        QWEN25OMNI_MODEL_ID,
        runner="generate",
        limit_mm_per_prompt={"audio": 2, "video": 2},
    )


def _make_video_data_url(*, with_audio: bool, tmp_path) -> str:
    """Generate a real 2-second MP4, with or without an audio track."""
    video_src = ["-f", "lavfi", "-i", "testsrc=duration=2:size=64x64:rate=4"]
    path = tmp_path / ("with_audio.mp4" if with_audio else "silent.mp4")
    if with_audio:
        cmd = [
            *video_src,
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-c:v",
            "mpeg4",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ]
    else:
        cmd = [*video_src, "-c:v", "mpeg4", "-an", str(path)]
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", *cmd], check=True)
    return "data:video/mp4;base64," + base64.b64encode(path.read_bytes()).decode()


def _video_messages(*data_urls: str) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = [{"type": "text", "text": "Describe."}]
    parts += [{"type": "video_url", "video_url": {"url": url}} for url in data_urls]
    return [{"role": "user", "content": parts}]


@pytest.mark.asyncio
async def test_parse_chat_messages_silent_video_degrades_to_video_only(
    qwen25omni_model_config_audio_video, tmp_path
):
    """A silent video must not fail the request when `use_audio_in_video` is on.

    The missing audio track is dropped (with a warning), the video is kept, and
    a second video with audio still contributes its track. Sync and async
    parsers must behave alike.
    """
    silent = _make_video_data_url(with_audio=False, tmp_path=tmp_path)
    with_audio = _make_video_data_url(with_audio=True, tmp_path=tmp_path)
    kwargs = {"use_audio_in_video": True}

    cases = {
        "silent only": (_video_messages(silent), {"video": 1, "audio": 1}),
        "with audio": (_video_messages(with_audio), {"video": 1, "audio": 1}),
        "mixed": (_video_messages(silent, with_audio), {"video": 2, "audio": 2}),
    }
    for name, (messages, expected) in cases.items():
        for mode in ("sync", "async"):
            if mode == "sync":
                _, mm_data, _ = parse_chat_messages(
                    messages,
                    qwen25omni_model_config_audio_video,
                    content_format="string",
                    mm_processor_kwargs=kwargs,
                )
            else:
                _, mm_data, _ = await parse_chat_messages_async(
                    messages,
                    qwen25omni_model_config_audio_video,
                    content_format="string",
                    mm_processor_kwargs=kwargs,
                )
            counts = {
                modality: len(items) for modality, items in (mm_data or {}).items()
            }
            assert counts == expected, f"{name} ({mode}): {counts}"


@pytest.mark.asyncio
async def test_parse_chat_messages_silent_video_uuid_bookkeeping(
    qwen25omni_model_config_audio_video, tmp_path
):
    """A dropped audio track keeps mm_data/mm_uuids aligned via a placeholder
    UUID so the engine cache can match it."""
    silent = _make_video_data_url(with_audio=False, tmp_path=tmp_path)
    kwargs = {"use_audio_in_video": True}

    for mode in ("sync", "async"):
        if mode == "sync":
            _, mm_data, mm_uuids = parse_chat_messages(
                _video_messages(silent),
                qwen25omni_model_config_audio_video,
                content_format="string",
                mm_processor_kwargs=kwargs,
            )
        else:
            _, mm_data, mm_uuids = await parse_chat_messages_async(
                _video_messages(silent),
                qwen25omni_model_config_audio_video,
                content_format="string",
                mm_processor_kwargs=kwargs,
            )
        assert mm_data is not None and mm_uuids is not None
        assert mm_data["audio"] == [None], f"{mode}: {mm_data['audio']}"
        assert len(mm_uuids["audio"]) == 1
        assert mm_uuids["audio"][0], f"{mode}: placeholder uuid missing"


@pytest.mark.asyncio
async def test_parse_chat_messages_video_audio_decode_errors_raise(
    qwen25omni_model_config_audio_video, tmp_path
):
    """Only a missing audio stream degrades; other decode failures propagate."""
    corrupt = "data:video/mp4;base64," + base64.b64encode(b"not a real mp4").decode()

    for mode in ("sync", "async"):
        with pytest.raises(ValueError):
            if mode == "sync":
                parse_chat_messages(
                    _video_messages(corrupt),
                    qwen25omni_model_config_audio_video,
                    content_format="string",
                    mm_processor_kwargs={"use_audio_in_video": True},
                )
            else:
                await parse_chat_messages_async(
                    _video_messages(corrupt),
                    qwen25omni_model_config_audio_video,
                    content_format="string",
                    mm_processor_kwargs={"use_audio_in_video": True},
                )


@pytest.mark.asyncio
async def test_parse_chat_messages_sync_async_parity_for_silent_video(tmp_path):
    """Sync and async must produce identical prompts and mm_data for a silent
    video under `use_audio_in_video`, and the audio placeholder count in the
    prompt must equal the number of audio items tracked (including the None
    sentinel for a silent track).
    """
    config = ModelConfig(
        QWEN25OMNI_MODEL_ID,
        runner="generate",
        interleave_mm_strings=True,
        limit_mm_per_prompt={"audio": 2, "video": 2},
    )
    silent = _make_video_data_url(with_audio=False, tmp_path=tmp_path)
    with_audio = _make_video_data_url(with_audio=True, tmp_path=tmp_path)
    kwargs = {"use_audio_in_video": True}

    audio_token = "<|AUDIO|>"

    cases = {
        "silent only": _video_messages(silent),
        "with audio": _video_messages(with_audio),
        "mixed": _video_messages(silent, with_audio),
    }
    for name, messages in cases.items():
        conv_s, data_s, _ = parse_chat_messages(
            messages,
            config,
            content_format="string",
            mm_processor_kwargs=kwargs,
        )
        conv_a, data_a, _ = await parse_chat_messages_async(
            messages,
            config,
            content_format="string",
            mm_processor_kwargs=kwargs,
        )
        prompt_s = conv_s[0]["content"]
        prompt_a = conv_a[0]["content"]
        assert isinstance(prompt_s, str) and isinstance(prompt_a, str)
        assert prompt_s == prompt_a, f"{name}: sync/async prompt differ"

        audio_items_s = (data_s or {}).get("audio", [])
        audio_items_a = (data_a or {}).get("audio", [])
        assert len(audio_items_s) == len(audio_items_a), (
            f"{name}: audio item count differs sync={len(audio_items_s)} "
            f"async={len(audio_items_a)}"
        )

        # Each audio item (including the None sentinel for a silent track)
        # contributes exactly one audio placeholder token span to the prompt,
        # so the processor sees a 1:1 mapping between prompt and mm_data.
        assert prompt_s.count(audio_token) == len(audio_items_s), (
            f"{name}: prompt has {prompt_s.count(audio_token)} audio "
            f"placeholders but {len(audio_items_s)} audio items"
        )
