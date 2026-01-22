# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import warnings
from collections.abc import Mapping
from typing import Literal

import pytest
import torch

from vllm.assets.audio import AudioAsset
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import (
    parse_chat_messages,
    parse_chat_messages_async,
)
from vllm.multimodal import MultiModalDataDict, MultiModalUUIDDict
from vllm.multimodal.utils import (
    encode_audio_url,
    encode_image_url,
    encode_video_url,
)
from vllm.utils.serial_utils import tensor2base64

PHI3V_MODEL_ID = "microsoft/Phi-3.5-vision-instruct"
QWEN2AUDIO_MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
QWEN25OMNI_MODEL_ID = "Qwen/Qwen2.5-Omni-7B"
MISTRAL_MODEL_ID = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"


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


@pytest.fixture(scope="function")
def phi3v_model_config():
    return ModelConfig(
        PHI3V_MODEL_ID,
        runner="generate",
        trust_remote_code=True,
        limit_mm_per_prompt={
            "image": 2,
        },
    )


@pytest.fixture(scope="function")
def phi3v_model_config_mm_interleaved():
    return ModelConfig(
        PHI3V_MODEL_ID,
        runner="generate",
        trust_remote_code=True,
        interleave_mm_strings=True,
        limit_mm_per_prompt={
            "image": 2,
        },
    )


@pytest.fixture(scope="function")
def phi3v_model_config_image_embeds():
    return ModelConfig(
        PHI3V_MODEL_ID,
        runner="generate",
        trust_remote_code=True,
        limit_mm_per_prompt={
            "image": 2,
        },
        enable_mm_embeds=True,
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
def qwen2_audio_model_config():
    return ModelConfig(
        QWEN2AUDIO_MODEL_ID,
        runner="generate",
        trust_remote_code=True,
        limit_mm_per_prompt={
            "audio": 1,
        },
    )


@pytest.fixture(scope="function")
def audio_embeds_model_config():
    return ModelConfig(
        QWEN2AUDIO_MODEL_ID,
        runner="generate",
        trust_remote_code=True,
        limit_mm_per_prompt={
            "audio": 2,
        },
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


@pytest.fixture(scope="function")
def mistral_model_config():
    return ModelConfig(
        MISTRAL_MODEL_ID,
        runner="generate",
        limit_mm_per_prompt={
            "image": 2,
        },
    )


@pytest.fixture(scope="module")
def image_url():
    image = ImageAsset("cherry_blossom")
    return encode_image_url(image.pil_image)


@pytest.fixture(scope="module")
def video_url():
    video = VideoAsset("baby_reading", 1)
    return encode_video_url(video.np_ndarrays)


@pytest.fixture(scope="module")
def audio_url():
    audio = AudioAsset("mary_had_lamb")
    return encode_audio_url(*audio.audio_and_sample_rate)


def _assert_mm_data_is_image_input(
    mm_data: MultiModalDataDict | None,
    image_count: int,
    skipped_image_indices: list | None = None,
) -> None:
    assert mm_data is not None
    assert set(mm_data.keys()) == {"image"}

    image_data = mm_data.get("image")
    assert image_data is not None

    assert isinstance(image_data, list) and len(image_data) == image_count
    if skipped_image_indices is not None:
        for i in skipped_image_indices:
            assert image_data[i] is None


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


def _assert_mm_uuids(
    mm_uuids: MultiModalUUIDDict | None,
    media_count: int,
    expected_uuids: list[str | None],
    modality: str = "image",
) -> None:
    if len(expected_uuids) > 0:
        assert mm_uuids is not None
        assert modality in mm_uuids

        image_uuids = mm_uuids.get(modality)
        assert image_uuids is not None

        assert isinstance(image_uuids, list) and len(image_uuids) == media_count

        assert image_uuids == expected_uuids
    else:
        assert mm_uuids is None


ModalityType = Literal["image", "video", "audio"]
MultiModalDataCounts = Mapping[ModalityType, int]


def _assert_mm_data_inputs(
    mm_data: MultiModalDataDict | None,
    data_count: MultiModalDataCounts,
    skipped_media_indices: dict[str, list] | None = None,  # modality -> list[int]
) -> None:
    assert mm_data is not None
    assert set(data_count.keys()) == (set(mm_data.keys()))

    for modality, n in data_count.items():
        modality_data = mm_data.get(modality)
        assert modality_data is not None
        assert isinstance(modality_data, list) and len(modality_data) == n

        if skipped_media_indices is not None:
            skipped_media_indices_for_modality = skipped_media_indices.get(modality)
            assert skipped_media_indices_for_modality is not None
            for i in skipped_media_indices_for_modality:
                assert modality_data[i] is None


def test_parse_chat_messages_single_image(
    phi3v_model_config,
    image_url,
):
    conversation, mm_data, mm_uuids = parse_chat_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "What's in the image?"},
                ],
            }
        ],
        phi3v_model_config,
        content_format="string",
    )

    assert conversation == [
        {"role": "user", "content": "<|image_1|>\nWhat's in the image?"}
    ]
    _assert_mm_data_is_image_input(mm_data, 1)
    _assert_mm_uuids(mm_uuids, 1, expected_uuids=[None])


def test_parse_chat_messages_single_image_with_uuid(
    phi3v_model_config,
    image_url,
):
    image_uuid = str(hash(image_url))
    conversation, mm_data, mm_uuids = parse_chat_messages(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                        },
                        "uuid": image_uuid,
                    },
                    {"type": "text", "text": "What's in the image?"},
                ],
            }
        ],
        phi3v_model_config,
        content_format="string",
    )

    assert conversation == [
        {"role": "user", "content": "<|image_1|>\nWhat's in the image?"}
    ]
    _assert_mm_data_is_image_input(mm_data, 1)
    _assert_mm_uuids(mm_uuids, 1, expected_uuids=[image_uuid])


def test_parse_chat_messages_single_empty_image_with_uuid(
    phi3v_model_config,
    image_url,
):
    image_uuid = str(hash(image_url))
    conversation, mm_data, mm_uuids = parse_chat_messages(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": None,
                        "uuid": image_uuid,
                    },
                    {"type": "text", "text": "What's in the image?"},
                ],
            }
        ],
        phi3v_model_config,
        content_format="string",
    )

    assert conversation == [
        {"role": "user", "content": "<|image_1|>\nWhat's in the image?"}
    ]
    _assert_mm_data_is_image_input(mm_data, 1, skipped_image_indices=[0])
    _assert_mm_uuids(mm_uuids, 1, expected_uuids=[image_uuid])


def test_parse_chat_messages_single_image_with_bad_uuid_format(
    phi3v_model_config,
    image_url,
):
    image_uuid = str(hash(image_url))
    conversation, mm_data, mm_uuids = parse_chat_messages(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                            "uuid": image_uuid,
                        },
                        "bad_uuid_key": image_uuid,
                    },
                    {"type": "text", "text": "What's in the image?"},
                ],
            }
        ],
        phi3v_model_config,
        content_format="string",
    )

    assert conversation == [
        {"role": "user", "content": "<|image_1|>\nWhat's in the image?"}
    ]
    _assert_mm_data_is_image_input(mm_data, 1)
    _assert_mm_uuids(mm_uuids, 1, expected_uuids=[None])


def test_parse_chat_messages_multiple_images_with_uuids(
    phi3v_model_config,
    image_url,
):
    image_uuid1 = "my_uuid_1"
    image_uuid2 = "my_uuid_2"

    conversation, mm_data, mm_uuids = parse_chat_messages(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                        },
                        "uuid": image_uuid1,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                        },
                        "uuid": image_uuid2,
                    },
                    {"type": "text", "text": "What's in the image?"},
                ],
            }
        ],
        phi3v_model_config,
        content_format="string",
    )

    assert conversation == [
        {
            "role": "user",
            "content": "<|image_1|>\n<|image_2|>\nWhat's in the image?",
        }
    ]
    _assert_mm_data_is_image_input(mm_data, 2)
    _assert_mm_uuids(mm_uuids, 2, expected_uuids=[image_uuid1, image_uuid2])


def test_parse_chat_messages_multiple_empty_images_with_uuids(
    phi3v_model_config,
    image_url,
):
    image_uuid1 = "my_uuid_1"
    image_uuid2 = "my_uuid_2"

    conversation, mm_data, mm_uuids = parse_chat_messages(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": None,
                        "uuid": image_uuid1,
                    },
                    {
                        "type": "image_url",
                        "image_url": None,
                        "uuid": image_uuid2,
                    },
                    {"type": "text", "text": "What's in the image?"},
                ],
            }
        ],
        phi3v_model_config,
        content_format="string",
    )

    assert conversation == [
        {
            "role": "user",
            "content": "<|image_1|>\n<|image_2|>\nWhat's in the image?",
        }
    ]
    _assert_mm_data_is_image_input(mm_data, 2, skipped_image_indices=[0, 1])
    _assert_mm_uuids(mm_uuids, 2, expected_uuids=[image_uuid1, image_uuid2])


def test_parse_chat_messages_mixed_empty_images_with_uuids(
    phi3v_model_config,
    image_url,
):
    image_uuid1 = "my_uuid_1"
    image_uuid2 = "my_uuid_2"

    conversation, mm_data, mm_uuids = parse_chat_messages(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                        },
                        "uuid": image_uuid1,
                    },
                    {
                        "type": "image_url",
                        "image_url": None,
                        "uuid": image_uuid2,
                    },
                    {"type": "text", "text": "What's in the image?"},
                ],
            }
        ],
        phi3v_model_config,
        content_format="string",
    )

    assert conversation == [
        {
            "role": "user",
            "content": "<|image_1|>\n<|image_2|>\nWhat's in the image?",
        }
    ]
    _assert_mm_data_is_image_input(mm_data, 2, skipped_image_indices=[1])
    _assert_mm_uuids(mm_uuids, 2, expected_uuids=[image_uuid1, image_uuid2])


@pytest.mark.asyncio
async def test_parse_chat_messages_single_image_with_uuid_async(
    phi3v_model_config,
    image_url,
):
    image_uuid = str(hash(image_url))
    conversation, mm_data, mm_uuids = await parse_chat_messages_async(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                        "uuid": image_uuid,
                    },
                    {"type": "text", "text": "What's in the image?"},
                ],
            }
        ],
        phi3v_model_config,
        content_format="string",
    )

    assert conversation == [
        {"role": "user", "content": "<|image_1|>\nWhat's in the image?"}
    ]
    _assert_mm_data_is_image_input(mm_data, 1)
    _assert_mm_uuids(mm_uuids, 1, expected_uuids=[image_uuid])


@pytest.mark.asyncio
async def test_parse_chat_messages_empty_image_with_uuid_async(
    phi3v_model_config,
    image_url,
):
    image_uuid = str(hash(image_url))
    conversation, mm_data, mm_uuids = await parse_chat_messages_async(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": None,
                        "uuid": image_uuid,
                    },
                    {"type": "text", "text": "What's in the image?"},
                ],
            }
        ],
        phi3v_model_config,
        content_format="string",
    )

    assert conversation == [
        {"role": "user", "content": "<|image_1|>\nWhat's in the image?"}
    ]
    _assert_mm_data_is_image_input(mm_data, 1, skipped_image_indices=[0])
    _assert_mm_uuids(mm_uuids, 1, expected_uuids=[image_uuid])


@pytest.mark.asyncio
async def test_parse_chat_messages_multiple_images_with_uuids_async(
    phi3v_model_config,
    image_url,
):
    image_uuid1 = "my_uuid_1"
    image_uuid2 = "my_uuid_2"

    conversation, mm_data, mm_uuids = await parse_chat_messages_async(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                        "uuid": image_uuid1,
                    },
                    {
                        "type": "image_pil",
                        "image_pil": ImageAsset("cherry_blossom").pil_image,
                        "uuid": image_uuid2,
                    },
                    {"type": "text", "text": "What's in these images?"},
                ],
            }
        ],
        phi3v_model_config,
        content_format="string",
    )

    assert conversation == [
        {
            "role": "user",
            "content": "<|image_1|>\n<|image_2|>\nWhat's in these images?",
        }
    ]
    _assert_mm_data_is_image_input(mm_data, 2)
    _assert_mm_uuids(mm_uuids, 2, expected_uuids=[image_uuid1, image_uuid2])


@pytest.mark.asyncio
async def test_parse_chat_messages_multiple_empty_images_with_uuids_async(
    phi3v_model_config,
    image_url,
):
    image_uuid1 = "my_uuid_1"
    image_uuid2 = "my_uuid_2"

    conversation, mm_data, mm_uuids = await parse_chat_messages_async(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": None,
                        "uuid": image_uuid1,
                    },
                    {
                        "type": "image_pil",
                        "image_pil": None,
                        "uuid": image_uuid2,
                    },
                    {"type": "text", "text": "What's in these images?"},
                ],
            }
        ],
        phi3v_model_config,
        content_format="string",
    )

    assert conversation == [
        {
            "role": "user",
            "content": "<|image_1|>\n<|image_2|>\nWhat's in these images?",
        }
    ]
    _assert_mm_data_is_image_input(mm_data, 2, skipped_image_indices=[0, 1])
    _assert_mm_uuids(mm_uuids, 2, expected_uuids=[image_uuid1, image_uuid2])


@pytest.mark.asyncio
async def test_parse_chat_messages_multiple_images_with_partial_uuids_async(
    phi3v_model_config,
    image_url,
):
    image_uuid2 = "my_uuid_2"

    conversation, mm_data, mm_uuids = await parse_chat_messages_async(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                    {
                        "type": "image_pil",
                        "image_pil": ImageAsset("cherry_blossom").pil_image,
                        "uuid": image_uuid2,
                    },
                    {"type": "text", "text": "What's in these images?"},
                ],
            }
        ],
        phi3v_model_config,
        content_format="string",
    )

    assert conversation == [
        {
            "role": "user",
            "content": "<|image_1|>\n<|image_2|>\nWhat's in these images?",
        }
    ]
    _assert_mm_data_is_image_input(mm_data, 2)
    _assert_mm_uuids(mm_uuids, 2, expected_uuids=[None, image_uuid2])


def test_parse_chat_messages_empty_system(
    mistral_model_config,
):
    # Test string format
    conversation, _, _ = parse_chat_messages(
        [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": [{"type": "text", "text": "Who are you?"}],
            },
        ],
        mistral_model_config,
        content_format="string",
    )
    assert conversation == [
        {"role": "system", "content": ""},
        {"role": "user", "content": "Who are you?"},
    ]

    # Test openai format
    conversation, _, _ = parse_chat_messages(
        [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": [{"type": "text", "text": "Who are you?"}],
            },
        ],
        mistral_model_config,
        content_format="openai",
    )
    assert conversation == [
        {"role": "system", "content": [{"type": "text", "text": ""}]},
        {"role": "user", "content": [{"type": "text", "text": "Who are you?"}]},
    ]


@pytest.mark.asyncio
async def test_parse_chat_messages_single_image_async(
    phi3v_model_config,
    image_url,
):
    conversation, mm_data, mm_uuids = await parse_chat_messages_async(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "What's in the image?"},
                ],
            }
        ],
        phi3v_model_config,
        content_format="string",
    )

    assert conversation == [
        {"role": "user", "content": "<|image_1|>\nWhat's in the image?"}
    ]
    _assert_mm_data_is_image_input(mm_data, 1)
    _assert_mm_uuids(mm_uuids, 1, expected_uuids=[None])


def test_parse_chat_messages_multiple_images(
    phi3v_model_config,
    image_url,
):
    conversation, mm_data, mm_uuids = parse_chat_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {
                        "type": "image_pil",
                        "image_pil": ImageAsset("cherry_blossom").pil_image,
                    },
                    {"type": "text", "text": "What's in these images?"},
                ],
            }
        ],
        phi3v_model_config,
        content_format="string",
    )

    assert conversation == [
        {
            "role": "user",
            "content": "<|image_1|>\n<|image_2|>\nWhat's in these images?",
        }
    ]
    _assert_mm_data_is_image_input(mm_data, 2)
    _assert_mm_uuids(mm_uuids, 2, expected_uuids=[None, None])


def test_parse_chat_messages_empty_pil_image_with_uuid(
    phi3v_model_config,
):
    uuid = "abcd"
    conversation, mm_data, mm_uuids = parse_chat_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_pil", "image_pil": None, "uuid": uuid},
                    {"type": "text", "text": "What's in this image?"},
                ],
            }
        ],
        phi3v_model_config,
        content_format="string",
    )

    assert conversation == [
        {
            "role": "user",
            "content": "<|image_1|>\nWhat's in this image?",
        }
    ]
    _assert_mm_data_is_image_input(mm_data, 1, skipped_image_indices=[0])
    _assert_mm_uuids(mm_uuids, 1, expected_uuids=[uuid])


def test_parse_chat_messages_empty_image_embeds_with_uuid(
    phi3v_model_config_image_embeds,
):
    uuid = "abcd"
    conversation, mm_data, mm_uuids = parse_chat_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_embeds", "image_embeds": None, "uuid": uuid},
                    {"type": "text", "text": "What's in this image?"},
                ],
            }
        ],
        phi3v_model_config_image_embeds,
        content_format="string",
    )

    assert conversation == [
        {
            "role": "user",
            "content": "<|image_1|>\nWhat's in this image?",
        }
    ]

    assert mm_data is not None
    assert "image" in mm_data
    assert isinstance(mm_data["image"], list)
    assert len(mm_data["image"]) == 1
    assert mm_data["image"][0] is None

    _assert_mm_uuids(mm_uuids, 1, expected_uuids=[uuid])


def test_parse_chat_messages_empty_audio_embeds_with_uuid(
    audio_embeds_model_config,
):
    """Test audio_embeds with UUID (no actual embeds data)."""
    uuid = "test-audio-uuid-123"

    conversation, mm_data, mm_uuids = parse_chat_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this audio"},
                    {"type": "audio_embeds", "audio_embeds": None, "uuid": uuid},
                ],
            }
        ],
        audio_embeds_model_config,
        content_format="string",
    )

    # Should have audio in mm_data as None (UUID provided)
    assert mm_data is not None
    assert "audio" in mm_data
    assert isinstance(mm_data["audio"], list)
    assert len(mm_data["audio"]) == 1
    assert mm_data["audio"][0] is None

    # UUID should be recorded
    _assert_mm_uuids(mm_uuids, 1, modality="audio", expected_uuids=[uuid])


def test_parse_chat_messages_audio_embeds_with_string(
    audio_embeds_model_config,
):
    """Test audio_embeds with base64 string embedding data."""

    import torch

    # Create a sample audio embedding tensor
    hidden_size = audio_embeds_model_config.get_inputs_embeds_size()
    audio_embedding = torch.randn(1, 128, hidden_size)

    # Encode it as base64
    base64_audio_embedding = tensor2base64(audio_embedding)

    conversation, mm_data, mm_uuids = parse_chat_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this audio"},
                    {
                        "type": "audio_embeds",
                        "audio_embeds": base64_audio_embedding,
                    },
                ],
            }
        ],
        audio_embeds_model_config,
        content_format="string",
    )

    # Should have audio embedding in mm_data (single tensor, not a list)
    assert mm_data is not None
    assert "audio" in mm_data
    assert isinstance(mm_data["audio"], torch.Tensor)
    assert mm_data["audio"].shape == audio_embedding.shape
    # No UUID provided
    assert mm_uuids is not None
    assert "audio" in mm_uuids
    _assert_mm_uuids(mm_uuids, 1, modality="audio", expected_uuids=[None])


@pytest.mark.asyncio
async def test_parse_chat_messages_audio_embeds_async(
    audio_embeds_model_config,
):
    """Test audio_embeds with async futures."""

    import torch

    # Create a sample audio embedding tensor
    hidden_size = audio_embeds_model_config.get_inputs_embeds_size()
    audio_embedding = torch.randn(1, 128, hidden_size)

    # Encode it as base64
    base64_audio_embedding = tensor2base64(audio_embedding)

    conversation, mm_data, mm_uuids = await parse_chat_messages_async(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this audio"},
                    {
                        "type": "audio_embeds",
                        "audio_embeds": base64_audio_embedding,
                    },
                ],
            }
        ],
        audio_embeds_model_config,
        content_format="string",
    )

    # Should have audio embedding in mm_data (single tensor, not a list)
    assert mm_data is not None
    assert "audio" in mm_data
    assert isinstance(mm_data["audio"], torch.Tensor)
    assert mm_data["audio"].shape == audio_embedding.shape
    # No UUID provided
    assert mm_uuids is not None
    assert "audio" in mm_uuids
    _assert_mm_uuids(mm_uuids, 1, modality="audio", expected_uuids=[None])


def test_parse_chat_messages_multiple_image_embeds(
    phi3v_model_config_image_embeds,
):
    """Test that multiple image_embeds in a single message are now supported.

    This test validates the fix for the limitation that previously only allowed
    one message with {'type': 'image_embeds'}. Now multiple image embeddings
    can be provided in a single request, similar to regular images.
    """
    # Create two sample image embedding tensors
    hidden_size = phi3v_model_config_image_embeds.get_inputs_embeds_size()
    image_embedding_1 = torch.randn(256, hidden_size)
    image_embedding_2 = torch.randn(128, hidden_size)

    # Encode them as base64 using the convenience function
    base64_image_embedding_1 = tensor2base64(image_embedding_1)
    base64_image_embedding_2 = tensor2base64(image_embedding_2)

    conversation, mm_data, mm_uuids = parse_chat_messages(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_embeds",
                        "image_embeds": base64_image_embedding_1,
                    },
                    {
                        "type": "image_embeds",
                        "image_embeds": base64_image_embedding_2,
                    },
                    {"type": "text", "text": "Describe these two images."},
                ],
            }
        ],
        phi3v_model_config_image_embeds,
        content_format="string",
    )

    # Verify conversation structure
    assert conversation == [
        {
            "role": "user",
            "content": "<|image_1|>\n<|image_2|>\nDescribe these two images.",
        }
    ]

    # Verify mm_data contains a list of embeddings (not a single embedding)
    assert mm_data is not None
    assert "image" in mm_data
    assert isinstance(mm_data["image"], list)
    assert len(mm_data["image"]) == 2

    # Verify each embedding has the correct shape
    assert isinstance(mm_data["image"][0], torch.Tensor)
    assert mm_data["image"][0].shape == image_embedding_1.shape
    assert isinstance(mm_data["image"][1], torch.Tensor)
    assert mm_data["image"][1].shape == image_embedding_2.shape

    # Verify UUIDs (None since we didn't provide any)
    _assert_mm_uuids(mm_uuids, 2, expected_uuids=[None, None])


def test_parse_chat_messages_multiple_image_embeds_with_uuids(
    phi3v_model_config_image_embeds,
):
    """Test multiple image_embeds with UUIDs.

    This validates that UUIDs are properly tracked for multiple embeddings.
    """
    uuid1 = "image-uuid-1"
    uuid2 = "image-uuid-2"

    conversation, mm_data, mm_uuids = parse_chat_messages(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_embeds",
                        "image_embeds": None,
                        "uuid": uuid1,
                    },
                    {
                        "type": "image_embeds",
                        "image_embeds": None,
                        "uuid": uuid2,
                    },
                    {"type": "text", "text": "Compare these images."},
                ],
            }
        ],
        phi3v_model_config_image_embeds,
        content_format="string",
    )

    # Verify conversation structure
    assert conversation == [
        {
            "role": "user",
            "content": "<|image_1|>\n<|image_2|>\nCompare these images.",
        }
    ]

    # Verify mm_data contains a list with None values (UUID references)
    assert mm_data is not None
    assert "image" in mm_data
    assert isinstance(mm_data["image"], list)
    assert len(mm_data["image"]) == 2
    assert mm_data["image"][0] is None
    assert mm_data["image"][1] is None

    # Verify UUIDs are correctly tracked
    _assert_mm_uuids(mm_uuids, 2, expected_uuids=[uuid1, uuid2])


@pytest.mark.asyncio
async def test_parse_chat_messages_multiple_image_embeds_async(
    phi3v_model_config_image_embeds,
):
    """Test multiple image_embeds with async parsing.

    This validates the AsyncMultiModalItemTracker also supports multiple embeddings.
    """
    # Create two sample image embedding tensors
    hidden_size = phi3v_model_config_image_embeds.get_inputs_embeds_size()
    image_embedding_1 = torch.randn(200, hidden_size)
    image_embedding_2 = torch.randn(150, hidden_size)

    # Encode them as base64 using the convenience function
    base64_image_embedding_1 = tensor2base64(image_embedding_1)
    base64_image_embedding_2 = tensor2base64(image_embedding_2)

    conversation, mm_data, mm_uuids = await parse_chat_messages_async(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_embeds",
                        "image_embeds": base64_image_embedding_1,
                    },
                    {
                        "type": "image_embeds",
                        "image_embeds": base64_image_embedding_2,
                    },
                    {"type": "text", "text": "What do these images show?"},
                ],
            }
        ],
        phi3v_model_config_image_embeds,
        content_format="string",
    )

    # Verify conversation structure
    assert conversation == [
        {
            "role": "user",
            "content": "<|image_1|>\n<|image_2|>\nWhat do these images show?",
        }
    ]

    # Await the future and verify mm_data
    assert mm_data is not None
    assert "image" in mm_data
    assert isinstance(mm_data["image"], list)
    assert len(mm_data["image"]) == 2

    # Verify each embedding has the correct shape
    assert isinstance(mm_data["image"][0], torch.Tensor)
    assert mm_data["image"][0].shape == image_embedding_1.shape
    assert isinstance(mm_data["image"][1], torch.Tensor)
    assert mm_data["image"][1].shape == image_embedding_2.shape

    # Verify UUIDs
    _assert_mm_uuids(mm_uuids, 2, expected_uuids=[None, None])


@pytest.mark.asyncio
async def test_parse_chat_messages_empty_image_embeds_with_uuid_async(
    phi3v_model_config_image_embeds,
):
    uuid = "abcd"
    conversation, mm_data, mm_uuids = await parse_chat_messages_async(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_embeds", "image_embeds": None, "uuid": uuid},
                    {"type": "text", "text": "What's in this image?"},
                ],
            }
        ],
        phi3v_model_config_image_embeds,
        content_format="string",
    )

    assert conversation == [
        {
            "role": "user",
            "content": "<|image_1|>\nWhat's in this image?",
        }
    ]
    assert mm_data is not None
    assert "image" in mm_data
    assert isinstance(mm_data["image"], list)
    assert len(mm_data["image"]) == 1
    assert mm_data["image"][0] is None

    _assert_mm_uuids(mm_uuids, 1, expected_uuids=[uuid])


def test_parse_chat_messages_empty_dict_image_embeds(
    phi3v_model_config_image_embeds,
):
    """Test that empty dictionary for image_embeds is handled without errors."""
    conversation, mm_data, mm_uuids = parse_chat_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_embeds", "image_embeds": {}},
                    {"type": "text", "text": "What's in this image?"},
                ],
            }
        ],
        phi3v_model_config_image_embeds,
        content_format="string",
    )

    # Verify conversation structure
    assert conversation == [
        {
            "role": "user",
            "content": "<|image_1|>\nWhat's in this image?",
        }
    ]

    # Verify mm_data contains an empty dictionary of embeddings
    assert mm_data is not None
    assert "image" in mm_data
    assert isinstance(mm_data["image"], dict)
    assert len(mm_data["image"]) == 0

    # Verify UUIDs (None since we didn't provide any)
    _assert_mm_uuids(mm_uuids, 1, expected_uuids=[None])


def test_parse_chat_messages_multiple_dict_image_embeds(
    qwen25omni_model_config_image_embeds,
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
                            "image_grid_thw": tensor2base64(grid_thw),
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
    _assert_mm_uuids(mm_uuids, batch_size, expected_uuids=[None, None])


@pytest.mark.asyncio
async def test_parse_chat_messages_multiple_images_async(
    phi3v_model_config,
    image_url,
):
    conversation, mm_data, mm_uuids = await parse_chat_messages_async(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {
                        "type": "image_pil",
                        "image_pil": ImageAsset("cherry_blossom").pil_image,
                    },
                    {"type": "text", "text": "What's in these images?"},
                ],
            }
        ],
        phi3v_model_config,
        content_format="string",
    )

    assert conversation == [
        {
            "role": "user",
            "content": "<|image_1|>\n<|image_2|>\nWhat's in these images?",
        }
    ]
    _assert_mm_data_is_image_input(mm_data, 2)
    _assert_mm_uuids(mm_uuids, 2, expected_uuids=[None, None])


def test_parse_chat_messages_placeholder_already_in_prompt(
    phi3v_model_config,
    image_url,
):
    conversation, mm_data, mm_uuids = parse_chat_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {
                        "type": "text",
                        "text": "What's in <|image_1|> and how does it compare to <|image_2|>?",  # noqa: E501
                    },
                ],
            }
        ],
        phi3v_model_config,
        content_format="string",
    )
    assert conversation == [
        {
            "role": "user",
            "content": "What's in <|image_1|> and how does it compare to <|image_2|>?",
        }
    ]
    _assert_mm_data_is_image_input(mm_data, 2)
    _assert_mm_uuids(mm_uuids, 2, expected_uuids=[None, None])


def test_parse_chat_messages_placeholder_one_already_in_prompt(
    phi3v_model_config,
    image_url,
):
    conversation, mm_data, mm_uuids = parse_chat_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {
                        "type": "text",
                        "text": "What's in <|image_1|> and how does it compare to "
                        "the other one?",
                    },
                ],
            }
        ],
        phi3v_model_config,
        content_format="string",
    )

    assert conversation == [
        {
            "role": "user",
            "content": "<|image_2|>\nWhat's in <|image_1|> and how does it compare to "
            "the other one?",
        }
    ]
    _assert_mm_data_is_image_input(mm_data, 2)
    _assert_mm_uuids(mm_uuids, 2, expected_uuids=[None, None])


def test_parse_chat_messages_multiple_images_across_messages(
    phi3v_model_config,
    image_url,
):
    conversation, mm_data, mm_uuids = parse_chat_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "What's in this image?"},
                ],
            },
            {"role": "assistant", "content": "Some stuff."},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "What about this one?"},
                ],
            },
        ],
        phi3v_model_config,
        content_format="string",
    )

    assert conversation == [
        {"role": "user", "content": "<|image_1|>\nWhat's in this image?"},
        {"role": "assistant", "content": "Some stuff."},
        {"role": "user", "content": "<|image_2|>\nWhat about this one?"},
    ]
    _assert_mm_data_is_image_input(mm_data, 2)
    _assert_mm_uuids(mm_uuids, 2, expected_uuids=[None, None])


def test_parse_chat_messages_multiple_images_with_uuids_across_messages(
    phi3v_model_config,
    image_url,
):
    image_uuid = str(hash(image_url))
    conversation, mm_data, mm_uuids = parse_chat_messages(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                        "uuid": image_uuid,
                    },
                    {"type": "text", "text": "What's in this image?"},
                ],
            },
            {"role": "assistant", "content": "Some stuff."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                        "uuid": image_uuid,
                    },
                    {"type": "text", "text": "What about this one?"},
                ],
            },
        ],
        phi3v_model_config,
        content_format="string",
    )

    assert conversation == [
        {"role": "user", "content": "<|image_1|>\nWhat's in this image?"},
        {"role": "assistant", "content": "Some stuff."},
        {"role": "user", "content": "<|image_2|>\nWhat about this one?"},
    ]
    _assert_mm_data_is_image_input(mm_data, 2)
    _assert_mm_uuids(mm_uuids, 2, expected_uuids=[image_uuid, image_uuid])


def test_parse_chat_messages_context_text_format(
    phi3v_model_config,
):
    conversation, mm_data, mm_uuids = parse_chat_messages(
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What's in this text?"}],
            },
            {"role": "assistant", "content": "Some stuff."},
            {"role": "user", "content": "What about this one?"},
        ],
        phi3v_model_config,
        content_format="openai",
    )

    assert conversation == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "What's in this text?"}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Some stuff."}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "What about this one?"}],
        },
    ]
    assert mm_data is None
    assert mm_uuids is None


def test_parse_chat_messages_rejects_too_many_images_in_one_message(
    phi3v_model_config,
    image_url,
):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="coroutine 'async_get_and_parse_image' was never awaited",
        )
        with pytest.raises(ValueError, match="At most"):
            parse_chat_messages(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            },
                            {"type": "text", "text": "What's in these images?"},
                        ],
                    }
                ],
                phi3v_model_config,
                content_format="string",
            )


def test_parse_chat_messages_rejects_too_many_images_across_messages(
    phi3v_model_config,
    image_url,
):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="coroutine 'async_get_and_parse_image' was never awaited",
        )
        with pytest.raises(ValueError, match="At most"):
            parse_chat_messages(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            },
                            {"type": "text", "text": "What's in this image?"},
                        ],
                    },
                    {"role": "assistant", "content": "Some stuff."},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            },
                            {"type": "text", "text": "What about these two?"},
                        ],
                    },
                ],
                phi3v_model_config,
                content_format="string",
            )


def test_parse_chat_messages_multiple_images_uncommon_input(
    phi3v_model_config,
    image_url,
):
    conversation, mm_data, mm_uuids = parse_chat_messages(
        [
            {
                "role": "user",
                "content": [
                    "What's in these images?",
                    {"image_url": image_url},
                    {"image_url": image_url},
                ],
            }
        ],
        phi3v_model_config,
        content_format="string",
    )

    assert conversation == [
        {
            "role": "user",
            "content": "<|image_1|>\n<|image_2|>\nWhat's in these images?",
        }
    ]
    _assert_mm_data_is_image_input(mm_data, 2)
    _assert_mm_uuids(mm_uuids, 2, expected_uuids=[None, None])


def test_parse_chat_messages_multiple_images_interleave(
    phi3v_model_config_mm_interleaved,
    image_url,
):
    conversation, mm_data, mm_uuids = parse_chat_messages(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "I need you to compare this image",
                    },
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "and this one"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "Do they have differences?"},
                ],
            }
        ],
        phi3v_model_config_mm_interleaved,
        content_format="string",
    )

    assert conversation == [
        {
            "role": "user",
            "content": "I need you to compare this image\n<|image_1|>\nand this one\n<|image_2|>\n"  # noqa: E501
            "Do they have differences?",
        }
    ]
    _assert_mm_data_is_image_input(mm_data, 2)
    _assert_mm_uuids(mm_uuids, 2, expected_uuids=[None, None])


@pytest.mark.asyncio
async def test_parse_chat_messages_multiple_images_interleave_async(
    phi3v_model_config_mm_interleaved,
    image_url,
):
    conversation, mm_data, mm_uuids = await parse_chat_messages_async(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "I need you to compare this image",
                    },
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "and this one"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "Do they have differences?"},
                ],
            }
        ],
        phi3v_model_config_mm_interleaved,
        content_format="string",
    )

    assert conversation == [
        {
            "role": "user",
            "content": "I need you to compare this image\n<|image_1|>\nand this one\n<|image_2|>\n"  # noqa: E501
            "Do they have differences?",
        }
    ]
    _assert_mm_data_is_image_input(mm_data, 2)
    _assert_mm_uuids(mm_uuids, 2, expected_uuids=[None, None])


@pytest.mark.asyncio
async def test_parse_chat_messages_multiple_images_with_uuids_interleave_async(
    phi3v_model_config_mm_interleaved,
    image_url,
):
    image_uuid = str(hash(image_url))
    conversation, mm_data, mm_uuids = await parse_chat_messages_async(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "I need you to compare this image",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                        "uuid": image_uuid,
                    },
                    {"type": "text", "text": "and this one"},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                        "uuid": image_uuid,
                    },
                    {"type": "text", "text": "Do they have differences?"},
                ],
            }
        ],
        phi3v_model_config_mm_interleaved,
        content_format="string",
    )

    assert conversation == [
        {
            "role": "user",
            "content": "I need you to compare this image\n<|image_1|>\nand this one\n<|image_2|>\n"  # noqa: E501
            "Do they have differences?",
        }
    ]
    _assert_mm_data_is_image_input(mm_data, 2)
    _assert_mm_uuids(mm_uuids, 2, expected_uuids=[image_uuid, image_uuid])


def test_parse_chat_messages_multiple_images_multiple_messages_interleave(
    phi3v_model_config_mm_interleaved,
    image_url,
):
    conversation, mm_data, mm_uuids = parse_chat_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's on this image?"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "Be accurate."},
                ],
            },
            {"role": "assistant", "content": "Some stuff."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's on this image?"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
        phi3v_model_config_mm_interleaved,
        content_format="string",
    )

    assert conversation == [
        {
            "role": "user",
            "content": "What's on this image?\n<|image_1|>\nBe accurate.",
        },
        {"role": "assistant", "content": "Some stuff."},
        {"role": "user", "content": "What's on this image?\n<|image_2|>"},
    ]
    _assert_mm_data_is_image_input(mm_data, 2)
    _assert_mm_uuids(mm_uuids, 2, expected_uuids=[None, None])


def test_parse_chat_messages_multiple_images_with_uuids_multiple_messages_interleave(
    phi3v_model_config_mm_interleaved,
    image_url,
):
    image_uuid = str(hash(image_url))
    conversation, mm_data, mm_uuids = parse_chat_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's on this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                        "uuid": image_uuid,
                    },
                    {"type": "text", "text": "Be accurate."},
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
                        "uuid": image_uuid,
                    },
                ],
            },
        ],
        phi3v_model_config_mm_interleaved,
        content_format="string",
    )

    assert conversation == [
        {
            "role": "user",
            "content": "What's on this image?\n<|image_1|>\nBe accurate.",
        },
        {"role": "assistant", "content": "Some stuff."},
        {"role": "user", "content": "What's on this image?\n<|image_2|>"},
    ]
    _assert_mm_data_is_image_input(mm_data, 2)
    _assert_mm_uuids(mm_uuids, 2, expected_uuids=[image_uuid, image_uuid])


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

    _assert_mm_data_inputs(mm_data, {"image": 2, "video": 1, "audio": 1})
    _assert_mm_uuids(mm_uuids, 2, modality="image", expected_uuids=[None, None])
    _assert_mm_uuids(mm_uuids, 1, modality="video", expected_uuids=[None])
    _assert_mm_uuids(mm_uuids, 1, modality="audio", expected_uuids=[None])


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

    _assert_mm_data_inputs(mm_data, {"image": 2, "video": 1, "audio": 1})
    _assert_mm_uuids(
        mm_uuids, 2, modality="image", expected_uuids=["image_123", "image_123"]
    )
    _assert_mm_uuids(mm_uuids, 1, modality="video", expected_uuids=["video_123"])
    _assert_mm_uuids(mm_uuids, 1, modality="audio", expected_uuids=["audio_123"])


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

    _assert_mm_data_inputs(
        mm_data,
        {"image": 2, "video": 1, "audio": 1},
        skipped_media_indices={"image": [0, 1], "video": [0], "audio": [0]},
    )
    _assert_mm_uuids(
        mm_uuids, 2, modality="image", expected_uuids=["image_123", "image_123"]
    )
    _assert_mm_uuids(mm_uuids, 1, modality="video", expected_uuids=["video_123"])
    _assert_mm_uuids(mm_uuids, 1, modality="audio", expected_uuids=["audio_123"])


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

    _assert_mm_data_inputs(mm_data, {"image": 2, "video": 1, "audio": 1})
    _assert_mm_uuids(mm_uuids, 2, modality="image", expected_uuids=["image_123", None])
    _assert_mm_uuids(mm_uuids, 1, modality="video", expected_uuids=["video_123"])
    _assert_mm_uuids(mm_uuids, 1, modality="audio", expected_uuids=[None])


def test_parse_chat_messages_multiple_images_interleave_with_placeholders(
    phi3v_model_config_mm_interleaved,
    image_url,
):
    with pytest.raises(
        ValueError,
        match=r"Found more '<|image_1|>' placeholders in input prompt "
        "than actual multimodal data items.",
    ):
        parse_chat_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {
                            "type": "text",
                            "text": "I need you to compare this image\n<|image_1|>\nand this one\n<|image_2|>\n"  # noqa: E501
                            "Do they have differences?",
                        },
                    ],
                }
            ],
            phi3v_model_config_mm_interleaved,
            content_format="string",
        )


def test_parse_chat_messages_include_thinking_chunk(mistral_model_config):
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a helpful assistant."},
                {
                    "type": "thinking",
                    "closed": True,
                    "thinking": "Only return the answer when you are confident.",
                },
            ],
        },
        {"role": "user", "content": "What is 2+2?"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me think about it."},
                {"type": "thinking", "closed": True, "thinking": "2+2 = 4"},
                {
                    "type": "text",
                    "text": "The answer is 4.",
                },
            ],
        },
    ]

    conversation_with_thinking, _, _ = parse_chat_messages(
        messages,
        mistral_model_config,
        content_format="openai",
    )

    expected_conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a helpful assistant."},
                {
                    "type": "text",
                    "text": "Only return the answer when you are confident.",
                },
            ],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "What is 2+2?"}],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me think about it."},
                {"type": "text", "text": "2+2 = 4"},
                {"type": "text", "text": "The answer is 4."},
            ],
        },
    ]

    assert conversation_with_thinking == expected_conversation


def test_parse_chat_messages_single_empty_audio_with_uuid(
    qwen2_audio_model_config,
):
    audio_uuid = "abcd"
    conversation, mm_data, mm_uuids = parse_chat_messages(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {},
                        "uuid": audio_uuid,
                    },
                    {"type": "text", "text": "What does the audio say?"},
                ],
            }
        ],
        qwen2_audio_model_config,
        content_format="string",
    )

    assert conversation == [
        {
            "role": "user",
            "content": "Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\nWhat does the "
            "audio say?",
        }
    ]
    _assert_mm_data_inputs(mm_data, {"audio": 1})
    _assert_mm_uuids(mm_uuids, 1, modality="audio", expected_uuids=[audio_uuid])


@pytest.mark.asyncio
async def test_parse_chat_messages_single_empty_audio_with_uuid_async(
    qwen2_audio_model_config,
):
    audio_uuid = "abcd"
    conversation, mm_data, mm_uuids = await parse_chat_messages_async(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {},
                        "uuid": audio_uuid,
                    },
                    {"type": "text", "text": "What does the audio say?"},
                ],
            }
        ],
        qwen2_audio_model_config,
        content_format="string",
    )

    assert conversation == [
        {
            "role": "user",
            "content": "Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\nWhat does the "
            "audio say?",
        }
    ]
    _assert_mm_data_inputs(mm_data, {"audio": 1})
    _assert_mm_uuids(mm_uuids, 1, modality="audio", expected_uuids=[audio_uuid])


def test_parse_chat_messages_image_vision_chunk(
    kimi_k2_5_model_config,
    image_url,
):
    messages = [
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
    _assert_mm_uuids(mm_uuids, 1, expected_uuids=[None], modality="vision_chunk")


def test_parse_chat_messages_video_vision_chunk(
    kimi_k2_5_model_config,
    video_url,
):
    messages = [
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
    _assert_mm_uuids(mm_uuids, 1, expected_uuids=[None], modality="vision_chunk")


def test_parse_chat_messages_image_vision_chunk_with_uuid(
    kimi_k2_5_model_config,
    image_url,
):
    image_uuid = "image_123"
    messages = [
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
    _assert_mm_uuids(mm_uuids, 1, expected_uuids=[image_uuid], modality="vision_chunk")


def test_parse_chat_messages_video_vision_chunk_with_uuid(
    kimi_k2_5_model_config,
    video_url,
):
    video_uuid = "video_456"
    messages = [
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
    _assert_mm_uuids(mm_uuids, 1, expected_uuids=[video_uuid], modality="vision_chunk")


def test_parse_chat_messages_mixed_vision_chunk(
    kimi_k2_5_model_config,
    image_url,
    video_url,
):
    messages = [
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
    _assert_mm_uuids(mm_uuids, 2, expected_uuids=[None, None], modality="vision_chunk")


def test_parse_chat_messages_mixed_vision_chunk_with_uuid(
    kimi_k2_5_model_config,
    image_url,
    video_url,
):
    image_uuid = "image_123"
    video_uuid = "video_456"
    messages = [
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
    _assert_mm_uuids(
        mm_uuids, 2, expected_uuids=[image_uuid, video_uuid], modality="vision_chunk"
    )


@pytest.mark.asyncio
async def test_parse_chat_messages_mixed_vision_chunk_async(
    kimi_k2_5_model_config,
    image_url,
    video_url,
):
    messages = [
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
    _assert_mm_uuids(mm_uuids, 2, expected_uuids=[None, None], modality="vision_chunk")


@pytest.mark.asyncio
async def test_parse_chat_messages_mixed_vision_chunk_with_uuid_async(
    kimi_k2_5_model_config,
    image_url,
    video_url,
):
    image_uuid = "image_123"
    video_uuid = "video_456"
    messages = [
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
    _assert_mm_uuids(
        mm_uuids, 2, expected_uuids=[image_uuid, video_uuid], modality="vision_chunk"
    )


@pytest.mark.asyncio
async def test_parse_chat_messages_image_vision_chunk_async(
    kimi_k2_5_model_config,
    image_url,
):
    messages = [
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
    _assert_mm_uuids(mm_uuids, 1, expected_uuids=[None], modality="vision_chunk")


@pytest.mark.asyncio
async def test_parse_chat_messages_video_vision_chunk_async(
    kimi_k2_5_model_config,
    video_url,
):
    messages = [
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
    _assert_mm_uuids(mm_uuids, 1, expected_uuids=[None], modality="vision_chunk")


@pytest.mark.asyncio
async def test_parse_chat_messages_image_vision_chunk_with_uuid_async(
    kimi_k2_5_model_config,
    image_url,
):
    image_uuid = "image_123"
    messages = [
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
    _assert_mm_uuids(mm_uuids, 1, expected_uuids=[image_uuid], modality="vision_chunk")


@pytest.mark.asyncio
async def test_parse_chat_messages_video_vision_chunk_with_uuid_async(
    kimi_k2_5_model_config,
    video_url,
):
    video_uuid = "video_456"
    messages = [
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
    _assert_mm_uuids(mm_uuids, 1, expected_uuids=[video_uuid], modality="vision_chunk")
