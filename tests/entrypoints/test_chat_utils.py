# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import warnings
from collections.abc import Mapping
from typing import Literal

import pytest
import torch
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy

from vllm.assets.audio import AudioAsset
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import (
    _try_extract_ast,
    apply_mistral_chat_template,
    load_chat_template,
    parse_chat_messages,
    parse_chat_messages_futures,
    resolve_chat_template_content_format,
    resolve_chat_template_kwargs,
    resolve_hf_chat_template,
)
from vllm.multimodal import MultiModalDataDict, MultiModalUUIDDict
from vllm.multimodal.utils import (
    encode_audio_url,
    encode_image_url,
    encode_video_url,
)
from vllm.tokenizers import get_tokenizer
from vllm.tokenizers.mistral import MistralTokenizer
from vllm.utils.serial_utils import tensor2base64

from ..models.registry import HF_EXAMPLE_MODELS
from ..utils import VLLM_PATH

EXAMPLES_DIR = VLLM_PATH / "examples"

PHI3V_MODEL_ID = "microsoft/Phi-3.5-vision-instruct"
ULTRAVOX_MODEL_ID = "fixie-ai/ultravox-v0_5-llama-3_2-1b"
QWEN2AUDIO_MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
QWEN2VL_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
QWEN25VL_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
QWEN25OMNI_MODEL_ID = "Qwen/Qwen2.5-Omni-7B"
QWEN3_MODEL_ID = "Qwen/Qwen3-8B"
LLAMA_GUARD_MODEL_ID = "meta-llama/Llama-Guard-3-1B"
HERMES_MODEL_ID = "NousResearch/Hermes-3-Llama-3.1-8B"
MISTRAL_MODEL_ID = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"


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
    conversation, mm_future, mm_uuids = parse_chat_messages_futures(
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
    _assert_mm_data_is_image_input(await mm_future, 1)
    _assert_mm_uuids(mm_uuids, 1, expected_uuids=[image_uuid])


@pytest.mark.asyncio
async def test_parse_chat_messages_empty_image_with_uuid_async(
    phi3v_model_config,
    image_url,
):
    image_uuid = str(hash(image_url))
    conversation, mm_future, mm_uuids = parse_chat_messages_futures(
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
    _assert_mm_data_is_image_input(await mm_future, 1, skipped_image_indices=[0])
    _assert_mm_uuids(mm_uuids, 1, expected_uuids=[image_uuid])


@pytest.mark.asyncio
async def test_parse_chat_messages_multiple_images_with_uuids_async(
    phi3v_model_config,
    image_url,
):
    image_uuid1 = "my_uuid_1"
    image_uuid2 = "my_uuid_2"

    conversation, mm_future, mm_uuids = parse_chat_messages_futures(
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
    _assert_mm_data_is_image_input(await mm_future, 2)
    _assert_mm_uuids(mm_uuids, 2, expected_uuids=[image_uuid1, image_uuid2])


@pytest.mark.asyncio
async def test_parse_chat_messages_multiple_empty_images_with_uuids_async(
    phi3v_model_config,
    image_url,
):
    image_uuid1 = "my_uuid_1"
    image_uuid2 = "my_uuid_2"

    conversation, mm_future, mm_uuids = parse_chat_messages_futures(
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
    _assert_mm_data_is_image_input(await mm_future, 2, skipped_image_indices=[0, 1])
    _assert_mm_uuids(mm_uuids, 2, expected_uuids=[image_uuid1, image_uuid2])


@pytest.mark.asyncio
async def test_parse_chat_messages_multiple_images_with_partial_uuids_async(
    phi3v_model_config,
    image_url,
):
    image_uuid2 = "my_uuid_2"

    conversation, mm_future, mm_uuids = parse_chat_messages_futures(
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
    _assert_mm_data_is_image_input(await mm_future, 2)
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
    conversation, mm_future, mm_uuids = parse_chat_messages_futures(
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
    _assert_mm_data_is_image_input(await mm_future, 1)
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
    audio_embedding = torch.randn(1, 128, 768)

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
    audio_embedding = torch.randn(1, 128, 768)

    # Encode it as base64
    base64_audio_embedding = tensor2base64(audio_embedding)

    conversation, mm_future, mm_uuids = parse_chat_messages_futures(
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
    mm_data = await mm_future
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
    image_embedding_1 = torch.randn(256, 1024)
    image_embedding_2 = torch.randn(128, 1024)

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
    image_embedding_1 = torch.randn(200, 768)
    image_embedding_2 = torch.randn(150, 768)

    # Encode them as base64 using the convenience function
    base64_image_embedding_1 = tensor2base64(image_embedding_1)
    base64_image_embedding_2 = tensor2base64(image_embedding_2)

    conversation, mm_future, mm_uuids = parse_chat_messages_futures(
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
    mm_data = await mm_future
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
    conversation, mm_future, mm_uuids = parse_chat_messages_futures(
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
    mm_data = await mm_future
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
    phi3v_model_config_image_embeds,
):
    """Test that multiple dictionaries for image_embeds is handled without errors."""
    # Create two sample image embedding tensors
    batch_size = 2
    image_embedding_1 = torch.randn(batch_size, 256, 1024)
    image_embedding_2 = torch.randn(batch_size, 3)

    conversation, mm_data, mm_uuids = parse_chat_messages(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_embeds",
                        "image_embeds": {
                            "image_embedding_1": tensor2base64(p),
                            "image_embedding_2": tensor2base64(i),
                        },
                    }
                    for p, i in zip(image_embedding_1, image_embedding_2)
                ]
                + [
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

    # Verify mm_data contains a dictionary of multi-embeddings
    assert mm_data is not None
    assert "image" in mm_data
    assert isinstance(mm_data["image"], dict)
    assert len(mm_data["image"]) == batch_size

    # Verify each embedding has the correct shape
    assert isinstance(mm_data["image"]["image_embedding_1"], torch.Tensor)
    assert mm_data["image"]["image_embedding_1"].shape == image_embedding_1.shape
    assert isinstance(mm_data["image"]["image_embedding_2"], torch.Tensor)
    assert mm_data["image"]["image_embedding_2"].shape == image_embedding_2.shape

    # Verify UUIDs (None since we didn't provide any)
    _assert_mm_uuids(mm_uuids, batch_size, expected_uuids=[None, None])


@pytest.mark.asyncio
async def test_parse_chat_messages_multiple_images_async(
    phi3v_model_config,
    image_url,
):
    conversation, mm_future, mm_uuids = parse_chat_messages_futures(
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
    _assert_mm_data_is_image_input(await mm_future, 2)
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
    conversation, mm_data, mm_uuids = parse_chat_messages_futures(
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
    _assert_mm_data_is_image_input(await mm_data, 2)
    _assert_mm_uuids(mm_uuids, 2, expected_uuids=[None, None])


@pytest.mark.asyncio
async def test_parse_chat_messages_multiple_images_with_uuids_interleave_async(
    phi3v_model_config_mm_interleaved,
    image_url,
):
    image_uuid = str(hash(image_url))
    conversation, mm_data, mm_uuids = parse_chat_messages_futures(
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
    _assert_mm_data_is_image_input(await mm_data, 2)
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


@pytest.mark.parametrize(
    "model",
    [
        QWEN2VL_MODEL_ID,  # tokenizer.chat_template is of type str
        HERMES_MODEL_ID,  # tokenizer.chat_template is of type dict
    ],
)
@pytest.mark.parametrize("use_tools", [True, False])
def test_resolve_hf_chat_template(sample_json_schema, model, use_tools):
    """checks that chat_template is a dict type for HF models."""
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    model_info.check_available_online(on_fail="skip")

    model_config = ModelConfig(
        model,
        tokenizer=model_info.tokenizer or model,
        tokenizer_mode=model_info.tokenizer_mode,
        revision=model_info.revision,
        trust_remote_code=model_info.trust_remote_code,
        hf_overrides=model_info.hf_overrides,
        skip_tokenizer_init=model_info.require_embed_inputs,
        enable_prompt_embeds=model_info.require_embed_inputs,
        enable_mm_embeds=model_info.require_embed_inputs,
        enforce_eager=model_info.enforce_eager,
        dtype=model_info.dtype,
    )

    # Build the tokenizer
    tokenizer = get_tokenizer(
        model,
        trust_remote_code=model_config.trust_remote_code,
    )

    tools = (
        [
            {
                "type": "function",
                "function": {
                    "name": "dummy_function_name",
                    "description": "This is a dummy function",
                    "parameters": sample_json_schema,
                },
            }
        ]
        if use_tools
        else None
    )

    # Test detecting the tokenizer's chat_template
    chat_template = resolve_hf_chat_template(
        tokenizer,
        chat_template=None,
        tools=tools,
        model_config=model_config,
    )
    assert isinstance(chat_template, str)


@pytest.mark.parametrize(
    "model, expected_kwargs",
    [
        (
            QWEN2VL_MODEL_ID,
            {
                "add_vision_id",
                "add_generation_prompt",
                "continue_final_message",
                "tools",
            },
        ),
        (
            QWEN3_MODEL_ID,
            {
                "enable_thinking",
                "add_generation_prompt",
                "continue_final_message",
                "tools",
            },
        ),
    ],
)
def test_resolve_hf_chat_template_kwargs(sample_json_schema, model, expected_kwargs):
    """checks that chat_template is a dict type for HF models."""
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    model_info.check_available_online(on_fail="skip")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "dummy_function_name",
                "description": "This is a dummy function",
                "parameters": sample_json_schema,
            },
        }
    ]

    chat_template_kwargs = {
        # both unused
        "unsed_kwargs_1": 123,
        "unsed_kwargs_2": "abc",
        # should not appear
        "chat_template": "{% Hello world! %}",
        "tokenize": True,
        # used by tokenizer
        "continue_final_message": True,
        "tools": tools,
        # both used by Qwen2-VL and Qwen3
        "add_generation_prompt": True,
        # only used by Qwen2-VL
        "add_vision_id": True,
        # only used by Qwen3
        "enable_thinking": True,
    }

    model_config = ModelConfig(
        model,
        tokenizer=model_info.tokenizer or model,
        tokenizer_mode=model_info.tokenizer_mode,
        revision=model_info.revision,
        trust_remote_code=model_info.trust_remote_code,
        hf_overrides=model_info.hf_overrides,
        skip_tokenizer_init=model_info.require_embed_inputs,
        enable_prompt_embeds=model_info.require_embed_inputs,
        enable_mm_embeds=model_info.require_embed_inputs,
        enforce_eager=model_info.enforce_eager,
        dtype=model_info.dtype,
    )

    # Build the tokenizer
    tokenizer = get_tokenizer(
        model,
        trust_remote_code=model_config.trust_remote_code,
    )

    # Test detecting the tokenizer's chat_template
    chat_template = resolve_hf_chat_template(
        tokenizer,
        chat_template=None,
        tools=tools,
        model_config=model_config,
    )
    with pytest.raises(
        ValueError, match="Found unexpected chat template kwargs from request"
    ):
        # should raise error if `chat_template_kwargs` contains
        # `chat_template` or `tokenize`
        resolve_chat_template_kwargs(
            tokenizer,
            chat_template=chat_template,
            chat_template_kwargs=chat_template_kwargs,
        )
    resolved_chat_template_kwargs = resolve_chat_template_kwargs(
        tokenizer,
        chat_template=chat_template,
        chat_template_kwargs=chat_template_kwargs,
        raise_on_unexpected=False,
    )
    assert set(resolved_chat_template_kwargs.keys()) == expected_kwargs

    # Additional test: Verify HF base parameters work with **kwargs tokenizers
    # This validates the fix for tokenizers like Kimi K2 that use **kwargs
    # to receive standard HuggingFace parameters instead of declaring them explicitly
    from vllm.entrypoints.chat_utils import _get_hf_base_chat_template_params

    hf_base_params = _get_hf_base_chat_template_params()
    # Verify common HF parameters are in the base class
    assert {"add_generation_prompt", "tools", "continue_final_message"}.issubset(
        hf_base_params
    ), f"Expected HF base params not found in {hf_base_params}"

    # Test with a mock tokenizer that uses **kwargs (like Kimi K2)
    class MockTokenizerWithKwargs:
        def apply_chat_template(self, conversation, **kwargs):
            return "mocked_output"

    mock_tokenizer = MockTokenizerWithKwargs()
    mock_kwargs = {
        "add_generation_prompt": True,
        "tools": tools,
        "continue_final_message": False,
        "unknown_param": "should_be_filtered",
    }
    resolved_mock = resolve_chat_template_kwargs(
        mock_tokenizer, chat_template, mock_kwargs, raise_on_unexpected=False
    )
    # HF base params should pass through even with **kwargs tokenizer
    assert "add_generation_prompt" in resolved_mock
    assert "tools" in resolved_mock
    assert "continue_final_message" in resolved_mock
    # Unknown params should be filtered out
    assert "unknown_param" not in resolved_mock


# NOTE: Qwen2-Audio default chat template is specially defined inside
# processor class instead of using `tokenizer_config.json`
@pytest.mark.parametrize(
    ("model", "expected_format"),
    [
        (PHI3V_MODEL_ID, "string"),
        (QWEN2VL_MODEL_ID, "openai"),
        (QWEN25VL_MODEL_ID, "openai"),
        (ULTRAVOX_MODEL_ID, "string"),
        (QWEN2AUDIO_MODEL_ID, "openai"),
        (LLAMA_GUARD_MODEL_ID, "openai"),
    ],
)
def test_resolve_content_format_hf_defined(model, expected_format):
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    model_info.check_available_online(on_fail="skip")

    model_config = ModelConfig(
        model,
        tokenizer=model_info.tokenizer or model,
        tokenizer_mode=model_info.tokenizer_mode,
        revision=model_info.revision,
        trust_remote_code=model_info.trust_remote_code,
        hf_overrides=model_info.hf_overrides,
        skip_tokenizer_init=model_info.require_embed_inputs,
        enable_prompt_embeds=model_info.require_embed_inputs,
        enable_mm_embeds=model_info.require_embed_inputs,
        enforce_eager=model_info.enforce_eager,
        dtype=model_info.dtype,
    )

    tokenizer = get_tokenizer(
        model,
        trust_remote_code=model_config.trust_remote_code,
    )

    # Test detecting the tokenizer's chat_template
    chat_template = resolve_hf_chat_template(
        tokenizer,
        chat_template=None,
        tools=None,
        model_config=model_config,
    )
    assert isinstance(chat_template, str)

    print("[TEXT]")
    print(chat_template)
    print("[AST]")
    print(_try_extract_ast(chat_template))

    resolved_format = resolve_chat_template_content_format(
        None,  # Test detecting the tokenizer's chat_template
        None,
        "auto",
        tokenizer,
        model_config=model_config,
    )

    assert resolved_format == expected_format


@pytest.mark.parametrize(
    ("model", "expected_format"),
    [
        ("Salesforce/blip2-opt-2.7b", "string"),
        ("facebook/chameleon-7b", "string"),
        ("deepseek-ai/deepseek-vl2-tiny", "string"),
        ("adept/fuyu-8b", "string"),
        ("google/paligemma-3b-mix-224", "string"),
        ("Qwen/Qwen-VL", "string"),
        ("Qwen/Qwen-VL-Chat", "string"),
    ],
)
def test_resolve_content_format_fallbacks(model, expected_format):
    model_info = HF_EXAMPLE_MODELS.find_hf_info(model)
    model_info.check_available_online(on_fail="skip")

    model_config = ModelConfig(
        model,
        tokenizer=model_info.tokenizer or model,
        tokenizer_mode=model_info.tokenizer_mode,
        revision=model_info.revision,
        trust_remote_code=model_info.trust_remote_code,
        hf_overrides=model_info.hf_overrides,
        skip_tokenizer_init=model_info.require_embed_inputs,
        enable_prompt_embeds=model_info.require_embed_inputs,
        enable_mm_embeds=model_info.require_embed_inputs,
        enforce_eager=model_info.enforce_eager,
        dtype=model_info.dtype,
    )

    tokenizer = get_tokenizer(
        model_config.tokenizer,
        trust_remote_code=model_config.trust_remote_code,
    )

    # Test detecting the tokenizer's chat_template
    chat_template = resolve_hf_chat_template(
        tokenizer,
        chat_template=None,
        tools=None,
        model_config=model_config,
    )
    assert isinstance(chat_template, str)

    print("[TEXT]")
    print(chat_template)
    print("[AST]")
    print(_try_extract_ast(chat_template))

    resolved_format = resolve_chat_template_content_format(
        None,  # Test detecting the tokenizer's chat_template
        None,
        "auto",
        tokenizer,
        model_config=model_config,
    )

    assert resolved_format == expected_format


@pytest.mark.parametrize(
    ("template_path", "expected_format"),
    [
        ("template_alpaca.jinja", "string"),
        ("template_baichuan.jinja", "string"),
        ("template_chatglm.jinja", "string"),
        ("template_chatglm2.jinja", "string"),
        ("template_chatml.jinja", "string"),
        ("template_dse_qwen2_vl.jinja", "openai"),
        ("template_falcon_180b.jinja", "string"),
        ("template_falcon.jinja", "string"),
        ("template_inkbot.jinja", "string"),
        ("template_teleflm.jinja", "string"),
        ("template_vlm2vec_phi3v.jinja", "openai"),
        ("template_vlm2vec_qwen2vl.jinja", "openai"),
        ("tool_chat_template_granite_20b_fc.jinja", "string"),
        ("tool_chat_template_hermes.jinja", "string"),
        ("tool_chat_template_internlm2_tool.jinja", "string"),
        ("tool_chat_template_llama3.1_json.jinja", "openai"),
        ("tool_chat_template_llama3.2_json.jinja", "openai"),
        ("tool_chat_template_mistral_parallel.jinja", "string"),
        ("tool_chat_template_mistral.jinja", "string"),
    ],
)
def test_resolve_content_format_examples(template_path, expected_format):
    model_config = ModelConfig(
        PHI3V_MODEL_ID,  # Dummy
        tokenizer=PHI3V_MODEL_ID,  # Dummy
        trust_remote_code=True,
    )

    dummy_tokenizer = get_tokenizer(
        PHI3V_MODEL_ID,  # Dummy
        trust_remote_code=model_config.trust_remote_code,
    )
    dummy_tokenizer.chat_template = None

    chat_template = load_chat_template(EXAMPLES_DIR / template_path)
    assert isinstance(chat_template, str)

    print("[TEXT]")
    print(chat_template)
    print("[AST]")
    print(_try_extract_ast(chat_template))

    resolved_format = resolve_chat_template_content_format(
        chat_template,
        None,
        "auto",
        dummy_tokenizer,
        model_config=model_config,
    )

    assert resolved_format == expected_format


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


def test_apply_mistral_chat_template_thinking_chunk():
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
        {"role": "user", "content": "Thanks, what is 3+3?"},
    ]
    mistral_tokenizer = MistralTokenizer.from_pretrained(
        "mistralai/Magistral-Small-2509"
    )

    tokens_ids = apply_mistral_chat_template(
        mistral_tokenizer, messages, chat_template=None, tools=None
    )

    string_tokens = mistral_tokenizer.mistral.decode(
        tokens_ids, special_token_policy=SpecialTokenPolicy.KEEP
    )

    expected_tokens = (
        r"<s>[SYSTEM_PROMPT]You are a helpful assistant.[THINK]Only return the"
        r" answer when you are confident.[/THINK][/SYSTEM_PROMPT]"
        r"[INST]What is 2+2?[/INST]"
        r"Let me think about it.[THINK]2+2 = 4[/THINK]The answer is 4.</s>"
        r"[INST]Thanks, what is 3+3?[/INST]"
    )

    assert string_tokens == expected_tokens


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
    conversation, mm_future, mm_uuids = parse_chat_messages_futures(
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
    _assert_mm_data_inputs(await mm_future, {"audio": 1})
    _assert_mm_uuids(mm_uuids, 1, modality="audio", expected_uuids=[audio_uuid])
