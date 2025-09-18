# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import warnings
from collections.abc import Mapping
from typing import Literal, Optional

import pytest
from mistral_common.tokens.tokenizers.base import (SpecialTokenPolicy,
                                                   SpecialTokens)
from mistral_common.tokens.tokenizers.tekken import (SpecialTokenInfo,
                                                     Tekkenizer)

from vllm.assets.audio import AudioAsset
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.config import ModelConfig
from vllm.entrypoints.chat_utils import (_try_extract_ast, load_chat_template,
                                         parse_chat_messages,
                                         parse_chat_messages_futures,
                                         resolve_chat_template_content_format,
                                         resolve_hf_chat_template)
from vllm.multimodal import MultiModalDataDict, MultiModalUUIDDict
from vllm.multimodal.utils import (encode_audio_base64, encode_image_base64,
                                   encode_video_base64)
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.transformers_utils.tokenizers.mistral import MistralTokenizer

from ..models.registry import HF_EXAMPLE_MODELS
from ..utils import VLLM_PATH

EXAMPLES_DIR = VLLM_PATH / "examples"

PHI3V_MODEL_ID = "microsoft/Phi-3.5-vision-instruct"
ULTRAVOX_MODEL_ID = "fixie-ai/ultravox-v0_5-llama-3_2-1b"
QWEN2AUDIO_MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
QWEN2VL_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
QWEN25VL_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
QWEN25OMNI_MODEL_ID = "Qwen/Qwen2.5-Omni-7B"
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


@pytest.fixture(scope="module")
def phi3v_tokenizer():
    return get_tokenizer(PHI3V_MODEL_ID)


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


@pytest.fixture(scope="module")
def qwen2_audio_tokenizer():
    return get_tokenizer(QWEN2AUDIO_MODEL_ID)


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


@pytest.fixture(scope="module")
def qwen25omni_tokenizer():
    return get_tokenizer(QWEN25OMNI_MODEL_ID)


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
def mistral_tokenizer():
    return get_tokenizer(MISTRAL_MODEL_ID)


@pytest.fixture(scope="module")
def image_url():
    image = ImageAsset("cherry_blossom")
    base64 = encode_image_base64(image.pil_image)
    return f"data:image/jpeg;base64,{base64}"


@pytest.fixture(scope="module")
def video_url():
    video = VideoAsset("baby_reading", 1)
    base64 = encode_video_base64(video.np_ndarrays)
    return f"data:video/jpeg;base64,{base64}"


@pytest.fixture(scope="module")
def audio_url():
    audio = AudioAsset("mary_had_lamb")
    base64 = encode_audio_base64(*audio.audio_and_sample_rate)
    return f"data:audio/ogg;base64,{base64}"


def _assert_mm_data_is_image_input(
    mm_data: Optional[MultiModalDataDict],
    image_count: int,
    skipped_image_indices: Optional[list] = None,
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
    mm_uuids: Optional[MultiModalUUIDDict],
    media_count: int,
    expected_uuids: list[Optional[str]],
    modality: str = "image",
) -> None:
    if len(expected_uuids) > 0:
        assert mm_uuids is not None
        assert modality in mm_uuids

        image_uuids = mm_uuids.get(modality)
        assert image_uuids is not None

        assert isinstance(image_uuids,
                          list) and len(image_uuids) == media_count

        assert image_uuids == expected_uuids
    else:
        assert mm_uuids is None


ModalityType = Literal["image", "video", "audio"]
MultiModalDataCounts = Mapping[ModalityType, int]


def _assert_mm_data_inputs(
        mm_data: Optional[MultiModalDataDict],
        data_count: MultiModalDataCounts,
        skipped_media_indices: Optional[dict[
            str, list]] = None,  # modality -> list[int]
) -> None:
    assert mm_data is not None
    assert set(data_count.keys()) == (set(mm_data.keys()))

    for modality, n in data_count.items():
        modality_data = mm_data.get(modality)
        assert modality_data is not None
        assert isinstance(modality_data, list) and len(modality_data) == n

        if skipped_media_indices is not None:
            skipped_media_indices_for_modality = skipped_media_indices.get(
                modality)
            assert skipped_media_indices_for_modality is not None
            for i in skipped_media_indices_for_modality:
                assert modality_data[i] is None


def test_parse_chat_messages_single_image(
    phi3v_model_config,
    phi3v_tokenizer,
    image_url,
):
    conversation, mm_data, mm_uuids = parse_chat_messages(
        [{
            "role":
            "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                },
                {
                    "type": "text",
                    "text": "What's in the image?"
                },
            ],
        }],
        phi3v_model_config,
        phi3v_tokenizer,
        content_format="string",
    )

    assert conversation == [{
        "role": "user",
        "content": "<|image_1|>\nWhat's in the image?"
    }]
    _assert_mm_data_is_image_input(mm_data, 1)
    _assert_mm_uuids(mm_uuids, 1, expected_uuids=[None])


def test_parse_chat_messages_single_image_with_uuid(
    phi3v_model_config,
    phi3v_tokenizer,
    image_url,
):
    image_uuid = str(hash(image_url))
    conversation, mm_data, mm_uuids = parse_chat_messages(
        [{
            "role":
            "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                    },
                    "uuid": image_uuid,
                },
                {
                    "type": "text",
                    "text": "What's in the image?"
                },
            ],
        }],
        phi3v_model_config,
        phi3v_tokenizer,
        content_format="string",
    )

    assert conversation == [{
        "role": "user",
        "content": "<|image_1|>\nWhat's in the image?"
    }]
    _assert_mm_data_is_image_input(mm_data, 1)
    _assert_mm_uuids(mm_uuids, 1, expected_uuids=[image_uuid])


def test_parse_chat_messages_single_empty_image_with_uuid(
    phi3v_model_config,
    phi3v_tokenizer,
    image_url,
):
    image_uuid = str(hash(image_url))
    conversation, mm_data, mm_uuids = parse_chat_messages(
        [{
            "role":
            "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": None,
                    "uuid": image_uuid,
                },
                {
                    "type": "text",
                    "text": "What's in the image?"
                },
            ],
        }],
        phi3v_model_config,
        phi3v_tokenizer,
        content_format="string",
    )

    assert conversation == [{
        "role": "user",
        "content": "<|image_1|>\nWhat's in the image?"
    }]
    _assert_mm_data_is_image_input(mm_data, 1, skipped_image_indices=[0])
    _assert_mm_uuids(mm_uuids, 1, expected_uuids=[image_uuid])


def test_parse_chat_messages_single_image_with_bad_uuid_format(
    phi3v_model_config,
    phi3v_tokenizer,
    image_url,
):
    image_uuid = str(hash(image_url))
    conversation, mm_data, mm_uuids = parse_chat_messages(
        [{
            "role":
            "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                        "uuid": image_uuid,
                    },
                    "bad_uuid_key": image_uuid,
                },
                {
                    "type": "text",
                    "text": "What's in the image?"
                },
            ],
        }],
        phi3v_model_config,
        phi3v_tokenizer,
        content_format="string",
    )

    assert conversation == [{
        "role": "user",
        "content": "<|image_1|>\nWhat's in the image?"
    }]
    _assert_mm_data_is_image_input(mm_data, 1)
    _assert_mm_uuids(mm_uuids, 1, expected_uuids=[None])


def test_parse_chat_messages_multiple_images_with_uuids(
    phi3v_model_config,
    phi3v_tokenizer,
    image_url,
):
    image_uuid1 = "my_uuid_1"
    image_uuid2 = "my_uuid_2"

    conversation, mm_data, mm_uuids = parse_chat_messages(
        [{
            "role":
            "user",
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
                {
                    "type": "text",
                    "text": "What's in the image?"
                },
            ],
        }],
        phi3v_model_config,
        phi3v_tokenizer,
        content_format="string",
    )

    assert conversation == [{
        "role":
        "user",
        "content":
        "<|image_1|>\n<|image_2|>\nWhat's in the image?",
    }]
    _assert_mm_data_is_image_input(mm_data, 2)
    _assert_mm_uuids(mm_uuids, 2, expected_uuids=[image_uuid1, image_uuid2])


def test_parse_chat_messages_multiple_empty_images_with_uuids(
    phi3v_model_config,
    phi3v_tokenizer,
    image_url,
):
    image_uuid1 = "my_uuid_1"
    image_uuid2 = "my_uuid_2"

    conversation, mm_data, mm_uuids = parse_chat_messages(
        [{
            "role":
            "user",
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
                {
                    "type": "text",
                    "text": "What's in the image?"
                },
            ],
        }],
        phi3v_model_config,
        phi3v_tokenizer,
        content_format="string",
    )

    assert conversation == [{
        "role":
        "user",
        "content":
        "<|image_1|>\n<|image_2|>\nWhat's in the image?",
    }]
    _assert_mm_data_is_image_input(mm_data, 2, skipped_image_indices=[0, 1])
    _assert_mm_uuids(mm_uuids, 2, expected_uuids=[image_uuid1, image_uuid2])


def test_parse_chat_messages_mixed_empty_images_with_uuids(
    phi3v_model_config,
    phi3v_tokenizer,
    image_url,
):
    image_uuid1 = "my_uuid_1"
    image_uuid2 = "my_uuid_2"

    conversation, mm_data, mm_uuids = parse_chat_messages(
        [{
            "role":
            "user",
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
                {
                    "type": "text",
                    "text": "What's in the image?"
                },
            ],
        }],
        phi3v_model_config,
        phi3v_tokenizer,
        content_format="string",
    )

    assert conversation == [{
        "role":
        "user",
        "content":
        "<|image_1|>\n<|image_2|>\nWhat's in the image?",
    }]
    _assert_mm_data_is_image_input(mm_data, 2, skipped_image_indices=[1])
    _assert_mm_uuids(mm_uuids, 2, expected_uuids=[image_uuid1, image_uuid2])


@pytest.mark.asyncio
async def test_parse_chat_messages_single_image_with_uuid_async(
    phi3v_model_config,
    phi3v_tokenizer,
    image_url,
):
    image_uuid = str(hash(image_url))
    conversation, mm_future, mm_uuids = parse_chat_messages_futures(
        [{
            "role":
            "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    },
                    "uuid": image_uuid,
                },
                {
                    "type": "text",
                    "text": "What's in the image?"
                },
            ],
        }],
        phi3v_model_config,
        phi3v_tokenizer,
        content_format="string",
    )

    assert conversation == [{
        "role": "user",
        "content": "<|image_1|>\nWhat's in the image?"
    }]
    _assert_mm_data_is_image_input(await mm_future, 1)
    _assert_mm_uuids(mm_uuids, 1, expected_uuids=[image_uuid])


@pytest.mark.asyncio
async def test_parse_chat_messages_empty_image_with_uuid_async(
    phi3v_model_config,
    phi3v_tokenizer,
    image_url,
):
    image_uuid = str(hash(image_url))
    conversation, mm_future, mm_uuids = parse_chat_messages_futures(
        [{
            "role":
            "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": None,
                    "uuid": image_uuid,
                },
                {
                    "type": "text",
                    "text": "What's in the image?"
                },
            ],
        }],
        phi3v_model_config,
        phi3v_tokenizer,
        content_format="string",
    )

    assert conversation == [{
        "role": "user",
        "content": "<|image_1|>\nWhat's in the image?"
    }]
    _assert_mm_data_is_image_input(await mm_future,
                                   1,
                                   skipped_image_indices=[0])
    _assert_mm_uuids(mm_uuids, 1, expected_uuids=[image_uuid])


@pytest.mark.asyncio
async def test_parse_chat_messages_multiple_images_with_uuids_async(
    phi3v_model_config,
    phi3v_tokenizer,
    image_url,
):
    image_uuid1 = "my_uuid_1"
    image_uuid2 = "my_uuid_2"

    conversation, mm_future, mm_uuids = parse_chat_messages_futures(
        [{
            "role":
            "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    },
                    "uuid": image_uuid1,
                },
                {
                    "type": "image_pil",
                    "image_pil": ImageAsset("cherry_blossom").pil_image,
                    "uuid": image_uuid2,
                },
                {
                    "type": "text",
                    "text": "What's in these images?"
                },
            ],
        }],
        phi3v_model_config,
        phi3v_tokenizer,
        content_format="string",
    )

    assert conversation == [{
        "role":
        "user",
        "content":
        "<|image_1|>\n<|image_2|>\nWhat's in these images?",
    }]
    _assert_mm_data_is_image_input(await mm_future, 2)
    _assert_mm_uuids(mm_uuids, 2, expected_uuids=[image_uuid1, image_uuid2])


@pytest.mark.asyncio
async def test_parse_chat_messages_multiple_empty_images_with_uuids_async(
    phi3v_model_config,
    phi3v_tokenizer,
    image_url,
):
    image_uuid1 = "my_uuid_1"
    image_uuid2 = "my_uuid_2"

    conversation, mm_future, mm_uuids = parse_chat_messages_futures(
        [{
            "role":
            "user",
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
                {
                    "type": "text",
                    "text": "What's in these images?"
                },
            ],
        }],
        phi3v_model_config,
        phi3v_tokenizer,
        content_format="string",
    )

    assert conversation == [{
        "role":
        "user",
        "content":
        "<|image_1|>\n<|image_2|>\nWhat's in these images?",
    }]
    _assert_mm_data_is_image_input(await mm_future,
                                   2,
                                   skipped_image_indices=[0, 1])
    _assert_mm_uuids(mm_uuids, 2, expected_uuids=[image_uuid1, image_uuid2])


@pytest.mark.asyncio
async def test_parse_chat_messages_multiple_images_with_partial_uuids_async(
    phi3v_model_config,
    phi3v_tokenizer,
    image_url,
):
    image_uuid2 = "my_uuid_2"

    conversation, mm_future, mm_uuids = parse_chat_messages_futures(
        [{
            "role":
            "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    },
                },
                {
                    "type": "image_pil",
                    "image_pil": ImageAsset("cherry_blossom").pil_image,
                    "uuid": image_uuid2,
                },
                {
                    "type": "text",
                    "text": "What's in these images?"
                },
            ],
        }],
        phi3v_model_config,
        phi3v_tokenizer,
        content_format="string",
    )

    assert conversation == [{
        "role":
        "user",
        "content":
        "<|image_1|>\n<|image_2|>\nWhat's in these images?",
    }]
    _assert_mm_data_is_image_input(await mm_future, 2)
    _assert_mm_uuids(mm_uuids, 2, expected_uuids=[None, image_uuid2])


def test_parse_chat_messages_empty_system(
    mistral_model_config,
    mistral_tokenizer,
):
    # Test string format
    conversation, _, _ = parse_chat_messages(
        [
            {
                "role": "system",
                "content": ""
            },
            {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "Who are you?"
                }],
            },
        ],
        mistral_model_config,
        mistral_tokenizer,
        content_format="string",
    )
    assert conversation == [
        {
            "role": "system",
            "content": ""
        },
        {
            "role": "user",
            "content": "Who are you?"
        },
    ]

    # Test openai format
    conversation, _, _ = parse_chat_messages(
        [
            {
                "role": "system",
                "content": ""
            },
            {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "Who are you?"
                }],
            },
        ],
        mistral_model_config,
        mistral_tokenizer,
        content_format="openai",
    )
    assert conversation == [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": ""
            }]
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": "Who are you?"
            }]
        },
    ]


@pytest.mark.asyncio
async def test_parse_chat_messages_single_image_async(
    phi3v_model_config,
    phi3v_tokenizer,
    image_url,
):
    conversation, mm_future, mm_uuids = parse_chat_messages_futures(
        [{
            "role":
            "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                },
                {
                    "type": "text",
                    "text": "What's in the image?"
                },
            ],
        }],
        phi3v_model_config,
        phi3v_tokenizer,
        content_format="string",
    )

    assert conversation == [{
        "role": "user",
        "content": "<|image_1|>\nWhat's in the image?"
    }]
    _assert_mm_data_is_image_input(await mm_future, 1)
    _assert_mm_uuids(mm_uuids, 1, expected_uuids=[None])


def test_parse_chat_messages_multiple_images(
    phi3v_model_config,
    phi3v_tokenizer,
    image_url,
):
    conversation, mm_data, mm_uuids = parse_chat_messages(
        [{
            "role":
            "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                },
                {
                    "type": "image_pil",
                    "image_pil": ImageAsset("cherry_blossom").pil_image,
                },
                {
                    "type": "text",
                    "text": "What's in these images?"
                },
            ],
        }],
        phi3v_model_config,
        phi3v_tokenizer,
        content_format="string",
    )

    assert conversation == [{
        "role":
        "user",
        "content":
        "<|image_1|>\n<|image_2|>\nWhat's in these images?",
    }]
    _assert_mm_data_is_image_input(mm_data, 2)
    _assert_mm_uuids(mm_uuids, 2, expected_uuids=[None, None])


def test_parse_chat_messages_empty_pil_image_with_uuid(
    phi3v_model_config,
    phi3v_tokenizer,
):
    uuid = "abcd"
    conversation, mm_data, mm_uuids = parse_chat_messages(
        [{
            "role":
            "user",
            "content": [
                {
                    "type": "image_pil",
                    "image_pil": None,
                    "uuid": uuid
                },
                {
                    "type": "text",
                    "text": "What's in this image?"
                },
            ],
        }],
        phi3v_model_config,
        phi3v_tokenizer,
        content_format="string",
    )

    assert conversation == [{
        "role": "user",
        "content": "<|image_1|>\nWhat's in this image?",
    }]
    _assert_mm_data_is_image_input(mm_data, 1, skipped_image_indices=[0])
    _assert_mm_uuids(mm_uuids, 1, expected_uuids=[uuid])


def test_parse_chat_messages_empty_image_embeds_with_uuid(
    phi3v_model_config,
    phi3v_tokenizer,
):
    uuid = "abcd"
    conversation, mm_data, mm_uuids = parse_chat_messages(
        [{
            "role":
            "user",
            "content": [
                {
                    "type": "image_embeds",
                    "image_embeds": None,
                    "uuid": uuid
                },
                {
                    "type": "text",
                    "text": "What's in this image?"
                },
            ],
        }],
        phi3v_model_config,
        phi3v_tokenizer,
        content_format="string",
    )

    assert conversation == [{
        "role": "user",
        "content": "<|image_1|>\nWhat's in this image?",
    }]
    assert mm_data is not None
    assert "image" in mm_data
    assert mm_data["image"] is None
    _assert_mm_uuids(mm_uuids, 1, expected_uuids=[uuid])


@pytest.mark.asyncio
async def test_parse_chat_messages_empty_image_embeds_with_uuid_async(
    phi3v_model_config,
    phi3v_tokenizer,
):
    uuid = "abcd"
    conversation, mm_future, mm_uuids = parse_chat_messages_futures(
        [{
            "role":
            "user",
            "content": [
                {
                    "type": "image_embeds",
                    "image_embeds": None,
                    "uuid": uuid
                },
                {
                    "type": "text",
                    "text": "What's in this image?"
                },
            ],
        }],
        phi3v_model_config,
        phi3v_tokenizer,
        content_format="string",
    )

    assert conversation == [{
        "role": "user",
        "content": "<|image_1|>\nWhat's in this image?",
    }]
    mm_data = await mm_future
    assert mm_data is not None
    assert "image" in mm_data
    assert mm_data["image"] is None
    _assert_mm_uuids(mm_uuids, 1, expected_uuids=[uuid])


@pytest.mark.asyncio
async def test_parse_chat_messages_multiple_images_async(
    phi3v_model_config,
    phi3v_tokenizer,
    image_url,
):
    conversation, mm_future, mm_uuids = parse_chat_messages_futures(
        [{
            "role":
            "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                },
                {
                    "type": "image_pil",
                    "image_pil": ImageAsset("cherry_blossom").pil_image,
                },
                {
                    "type": "text",
                    "text": "What's in these images?"
                },
            ],
        }],
        phi3v_model_config,
        phi3v_tokenizer,
        content_format="string",
    )

    assert conversation == [{
        "role":
        "user",
        "content":
        "<|image_1|>\n<|image_2|>\nWhat's in these images?",
    }]
    _assert_mm_data_is_image_input(await mm_future, 2)
    _assert_mm_uuids(mm_uuids, 2, expected_uuids=[None, None])


def test_parse_chat_messages_placeholder_already_in_prompt(
    phi3v_model_config,
    phi3v_tokenizer,
    image_url,
):
    conversation, mm_data, mm_uuids = parse_chat_messages(
        [{
            "role":
            "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                },
                {
                    "type":
                    "text",
                    "text":
                    "What's in <|image_1|> and how does it compare to <|image_2|>?",  # noqa: E501
                },
            ],
        }],
        phi3v_model_config,
        phi3v_tokenizer,
        content_format="string",
    )
    assert conversation == [{
        "role":
        "user",
        "content":
        "What's in <|image_1|> and how does it compare to <|image_2|>?",
    }]
    _assert_mm_data_is_image_input(mm_data, 2)
    _assert_mm_uuids(mm_uuids, 2, expected_uuids=[None, None])


def test_parse_chat_messages_placeholder_one_already_in_prompt(
    phi3v_model_config,
    phi3v_tokenizer,
    image_url,
):
    conversation, mm_data, mm_uuids = parse_chat_messages(
        [{
            "role":
            "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                },
                {
                    "type":
                    "text",
                    "text":
                    "What's in <|image_1|> and how does it compare to the other one?",  # noqa: E501
                },
            ],
        }],
        phi3v_model_config,
        phi3v_tokenizer,
        content_format="string",
    )

    assert conversation == [{
        "role":
        "user",
        "content":
        "<|image_2|>\nWhat's in <|image_1|> and how does it compare to the "
        "other one?",
    }]
    _assert_mm_data_is_image_input(mm_data, 2)
    _assert_mm_uuids(mm_uuids, 2, expected_uuids=[None, None])


def test_parse_chat_messages_multiple_images_across_messages(
    phi3v_model_config,
    phi3v_tokenizer,
    image_url,
):
    conversation, mm_data, mm_uuids = parse_chat_messages(
        [
            {
                "role":
                "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    },
                    {
                        "type": "text",
                        "text": "What's in this image?"
                    },
                ],
            },
            {
                "role": "assistant",
                "content": "Some stuff."
            },
            {
                "role":
                "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    },
                    {
                        "type": "text",
                        "text": "What about this one?"
                    },
                ],
            },
        ],
        phi3v_model_config,
        phi3v_tokenizer,
        content_format="string",
    )

    assert conversation == [
        {
            "role": "user",
            "content": "<|image_1|>\nWhat's in this image?"
        },
        {
            "role": "assistant",
            "content": "Some stuff."
        },
        {
            "role": "user",
            "content": "<|image_2|>\nWhat about this one?"
        },
    ]
    _assert_mm_data_is_image_input(mm_data, 2)
    _assert_mm_uuids(mm_uuids, 2, expected_uuids=[None, None])


def test_parse_chat_messages_multiple_images_with_uuids_across_messages(
    phi3v_model_config,
    phi3v_tokenizer,
    image_url,
):
    image_uuid = str(hash(image_url))
    conversation, mm_data, mm_uuids = parse_chat_messages(
        [
            {
                "role":
                "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        },
                        "uuid": image_uuid,
                    },
                    {
                        "type": "text",
                        "text": "What's in this image?"
                    },
                ],
            },
            {
                "role": "assistant",
                "content": "Some stuff."
            },
            {
                "role":
                "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        },
                        "uuid": image_uuid,
                    },
                    {
                        "type": "text",
                        "text": "What about this one?"
                    },
                ],
            },
        ],
        phi3v_model_config,
        phi3v_tokenizer,
        content_format="string",
    )

    assert conversation == [
        {
            "role": "user",
            "content": "<|image_1|>\nWhat's in this image?"
        },
        {
            "role": "assistant",
            "content": "Some stuff."
        },
        {
            "role": "user",
            "content": "<|image_2|>\nWhat about this one?"
        },
    ]
    _assert_mm_data_is_image_input(mm_data, 2)
    _assert_mm_uuids(mm_uuids, 2, expected_uuids=[image_uuid, image_uuid])


def test_parse_chat_messages_context_text_format(
    phi3v_model_config,
    phi3v_tokenizer,
):
    conversation, mm_data, mm_uuids = parse_chat_messages(
        [
            {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "What's in this text?"
                }],
            },
            {
                "role": "assistant",
                "content": "Some stuff."
            },
            {
                "role": "user",
                "content": "What about this one?"
            },
        ],
        phi3v_model_config,
        phi3v_tokenizer,
        content_format="openai",
    )

    assert conversation == [
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": "What's in this text?"
            }],
        },
        {
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": "Some stuff."
            }],
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": "What about this one?"
            }],
        },
    ]
    assert mm_data is None
    assert mm_uuids is None


def test_parse_chat_messages_rejects_too_many_images_in_one_message(
    phi3v_model_config,
    phi3v_tokenizer,
    image_url,
):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="coroutine 'async_get_and_parse_image' was never awaited",
        )
        with pytest.raises(ValueError, match="At most"):
            parse_chat_messages(
                [{
                    "role":
                    "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            },
                        },
                        {
                            "type": "text",
                            "text": "What's in these images?"
                        },
                    ],
                }],
                phi3v_model_config,
                phi3v_tokenizer,
                content_format="string",
            )


def test_parse_chat_messages_rejects_too_many_images_across_messages(
    phi3v_model_config,
    phi3v_tokenizer,
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
                        "role":
                        "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url
                                },
                            },
                            {
                                "type": "text",
                                "text": "What's in this image?"
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": "Some stuff."
                    },
                    {
                        "role":
                        "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url
                                },
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url
                                },
                            },
                            {
                                "type": "text",
                                "text": "What about these two?"
                            },
                        ],
                    },
                ],
                phi3v_model_config,
                phi3v_tokenizer,
                content_format="string",
            )


def test_parse_chat_messages_multiple_images_uncommon_input(
    phi3v_model_config,
    phi3v_tokenizer,
    image_url,
):
    conversation, mm_data, mm_uuids = parse_chat_messages(
        [{
            "role":
            "user",
            "content": [
                "What's in these images?",
                {
                    "image_url": image_url
                },
                {
                    "image_url": image_url
                },
            ],
        }],
        phi3v_model_config,
        phi3v_tokenizer,
        content_format="string",
    )

    assert conversation == [{
        "role":
        "user",
        "content":
        "<|image_1|>\n<|image_2|>\nWhat's in these images?",
    }]
    _assert_mm_data_is_image_input(mm_data, 2)
    _assert_mm_uuids(mm_uuids, 2, expected_uuids=[None, None])


def test_parse_chat_messages_multiple_images_interleave(
    phi3v_model_config_mm_interleaved,
    phi3v_tokenizer,
    image_url,
):
    conversation, mm_data, mm_uuids = parse_chat_messages(
        [{
            "role":
            "user",
            "content": [
                {
                    "type": "text",
                    "text": "I need you to compare this image",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                },
                {
                    "type": "text",
                    "text": "and this one"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                },
                {
                    "type": "text",
                    "text": "Do they have differences?"
                },
            ],
        }],
        phi3v_model_config_mm_interleaved,
        phi3v_tokenizer,
        content_format="string",
    )

    assert conversation == [{
        "role":
        "user",
        "content":
        "I need you to compare this image\n<|image_1|>\nand this one\n<|image_2|>\n"  # noqa: E501
        "Do they have differences?",
    }]
    _assert_mm_data_is_image_input(mm_data, 2)
    _assert_mm_uuids(mm_uuids, 2, expected_uuids=[None, None])


@pytest.mark.asyncio
async def test_parse_chat_messages_multiple_images_interleave_async(
    phi3v_model_config_mm_interleaved,
    phi3v_tokenizer,
    image_url,
):
    conversation, mm_data, mm_uuids = parse_chat_messages_futures(
        [{
            "role":
            "user",
            "content": [
                {
                    "type": "text",
                    "text": "I need you to compare this image",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                },
                {
                    "type": "text",
                    "text": "and this one"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                },
                {
                    "type": "text",
                    "text": "Do they have differences?"
                },
            ],
        }],
        phi3v_model_config_mm_interleaved,
        phi3v_tokenizer,
        content_format="string",
    )

    assert conversation == [{
        "role":
        "user",
        "content":
        "I need you to compare this image\n<|image_1|>\nand this one\n<|image_2|>\n"  # noqa: E501
        "Do they have differences?",
    }]
    _assert_mm_data_is_image_input(await mm_data, 2)
    _assert_mm_uuids(mm_uuids, 2, expected_uuids=[None, None])


@pytest.mark.asyncio
async def test_parse_chat_messages_multiple_images_with_uuids_interleave_async(
    phi3v_model_config_mm_interleaved,
    phi3v_tokenizer,
    image_url,
):
    image_uuid = str(hash(image_url))
    conversation, mm_data, mm_uuids = parse_chat_messages_futures(
        [{
            "role":
            "user",
            "content": [
                {
                    "type": "text",
                    "text": "I need you to compare this image",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    },
                    "uuid": image_uuid,
                },
                {
                    "type": "text",
                    "text": "and this one"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    },
                    "uuid": image_uuid,
                },
                {
                    "type": "text",
                    "text": "Do they have differences?"
                },
            ],
        }],
        phi3v_model_config_mm_interleaved,
        phi3v_tokenizer,
        content_format="string",
    )

    assert conversation == [{
        "role":
        "user",
        "content":
        "I need you to compare this image\n<|image_1|>\nand this one\n<|image_2|>\n"  # noqa: E501
        "Do they have differences?",
    }]
    _assert_mm_data_is_image_input(await mm_data, 2)
    _assert_mm_uuids(mm_uuids, 2, expected_uuids=[image_uuid, image_uuid])


def test_parse_chat_messages_multiple_images_multiple_messages_interleave(
    phi3v_model_config_mm_interleaved,
    phi3v_tokenizer,
    image_url,
):
    conversation, mm_data, mm_uuids = parse_chat_messages(
        [
            {
                "role":
                "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's on this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    },
                    {
                        "type": "text",
                        "text": "Be accurate."
                    },
                ],
            },
            {
                "role": "assistant",
                "content": "Some stuff."
            },
            {
                "role":
                "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's on this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    },
                ],
            },
        ],
        phi3v_model_config_mm_interleaved,
        phi3v_tokenizer,
        content_format="string",
    )

    assert conversation == [
        {
            "role": "user",
            "content": "What's on this image?\n<|image_1|>\nBe accurate.",
        },
        {
            "role": "assistant",
            "content": "Some stuff."
        },
        {
            "role": "user",
            "content": "What's on this image?\n<|image_2|>"
        },
    ]
    _assert_mm_data_is_image_input(mm_data, 2)
    _assert_mm_uuids(mm_uuids, 2, expected_uuids=[None, None])


def test_parse_chat_messages_multiple_images_with_uuids_multiple_messages_interleave(  # noqa: E501
    phi3v_model_config_mm_interleaved,
    phi3v_tokenizer,
    image_url,
):
    image_uuid = str(hash(image_url))
    conversation, mm_data, mm_uuids = parse_chat_messages(
        [
            {
                "role":
                "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's on this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        },
                        "uuid": image_uuid,
                    },
                    {
                        "type": "text",
                        "text": "Be accurate."
                    },
                ],
            },
            {
                "role": "assistant",
                "content": "Some stuff."
            },
            {
                "role":
                "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's on this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        },
                        "uuid": image_uuid,
                    },
                ],
            },
        ],
        phi3v_model_config_mm_interleaved,
        phi3v_tokenizer,
        content_format="string",
    )

    assert conversation == [
        {
            "role": "user",
            "content": "What's on this image?\n<|image_1|>\nBe accurate.",
        },
        {
            "role": "assistant",
            "content": "Some stuff."
        },
        {
            "role": "user",
            "content": "What's on this image?\n<|image_2|>"
        },
    ]
    _assert_mm_data_is_image_input(mm_data, 2)
    _assert_mm_uuids(mm_uuids, 2, expected_uuids=[image_uuid, image_uuid])


def test_parse_chat_messages_multiple_modals_multiple_messages_interleave(
    qwen25omni_model_config_mm_interleaved,
    qwen25omni_tokenizer,
    image_url,
    video_url,
    audio_url,
):
    conversation, mm_data, mm_uuids = parse_chat_messages(
        [
            {
                "role":
                "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's on this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    },
                    {
                        "type": "text",
                        "text": "Now listen to this audio"
                    },
                    {
                        "type": "audio_url",
                        "audio_url": {
                            "url": audio_url
                        }
                    },
                ],
            },
            {
                "role": "assistant",
                "content": "Some stuff."
            },
            {
                "role":
                "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's on this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    },
                    {
                        "type": "text",
                        "text": "And what's in the video?"
                    },
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": video_url
                        }
                    },
                ],
            },
        ],
        qwen25omni_model_config_mm_interleaved,
        qwen25omni_tokenizer,
        content_format="string",
    )

    assert conversation == [
        {
            "role":
            "user",
            "content":
            "What's on this image?\n<|vision_start|><|IMAGE|><|vision_end|>\n"
            "Now listen to this audio\nAudio 1: <|audio_bos|><|AUDIO|><|audio_eos|>",  # noqa: E501
        },
        {
            "role": "assistant",
            "content": "Some stuff."
        },
        {
            "role":
            "user",
            "content":
            "What's on this image?\n<|vision_start|><|IMAGE|><|vision_end|>\n"
            "And what's in the video?\n<|vision_start|><|VIDEO|><|vision_end|>",
        },
    ]

    _assert_mm_data_inputs(mm_data, {"image": 2, "video": 1, "audio": 1})
    _assert_mm_uuids(mm_uuids,
                     2,
                     modality="image",
                     expected_uuids=[None, None])
    _assert_mm_uuids(mm_uuids, 1, modality="video", expected_uuids=[None])
    _assert_mm_uuids(mm_uuids, 1, modality="audio", expected_uuids=[None])


def test_parse_chat_messages_multiple_modals_with_uuids_multiple_messages_interleave(  # noqa: E501
    qwen25omni_model_config_mm_interleaved,
    qwen25omni_tokenizer,
    image_url,
    video_url,
    audio_url,
):
    conversation, mm_data, mm_uuids = parse_chat_messages(
        [
            {
                "role":
                "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's on this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        },
                        "uuid": "image_123",
                    },
                    {
                        "type": "text",
                        "text": "Now listen to this audio"
                    },
                    {
                        "type": "audio_url",
                        "audio_url": {
                            "url": audio_url
                        },
                        "uuid": "audio_123",
                    },
                ],
            },
            {
                "role": "assistant",
                "content": "Some stuff."
            },
            {
                "role":
                "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's on this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        },
                        "uuid": "image_123",
                    },
                    {
                        "type": "text",
                        "text": "And what's in the video?"
                    },
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": video_url
                        },
                        "uuid": "video_123",
                    },
                ],
            },
        ],
        qwen25omni_model_config_mm_interleaved,
        qwen25omni_tokenizer,
        content_format="string",
    )

    assert conversation == [
        {
            "role":
            "user",
            "content":
            "What's on this image?\n<|vision_start|><|IMAGE|><|vision_end|>\n"
            "Now listen to this audio\nAudio 1: <|audio_bos|><|AUDIO|><|audio_eos|>",  # noqa: E501
        },
        {
            "role": "assistant",
            "content": "Some stuff."
        },
        {
            "role":
            "user",
            "content":
            "What's on this image?\n<|vision_start|><|IMAGE|><|vision_end|>\n"
            "And what's in the video?\n<|vision_start|><|VIDEO|><|vision_end|>",
        },
    ]

    _assert_mm_data_inputs(mm_data, {"image": 2, "video": 1, "audio": 1})
    _assert_mm_uuids(mm_uuids,
                     2,
                     modality="image",
                     expected_uuids=["image_123", "image_123"])
    _assert_mm_uuids(mm_uuids,
                     1,
                     modality="video",
                     expected_uuids=["video_123"])
    _assert_mm_uuids(mm_uuids,
                     1,
                     modality="audio",
                     expected_uuids=["audio_123"])


def test_parse_chat_messages_multiple_modals_with_uuids_multiple_empty_media_messages_interleave(  # noqa: E501
    qwen25omni_model_config_mm_interleaved,
    qwen25omni_tokenizer,
    image_url,
    video_url,
    audio_url,
):
    conversation, mm_data, mm_uuids = parse_chat_messages(
        [
            {
                "role":
                "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's on this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": None,
                        "uuid": "image_123",
                    },
                    {
                        "type": "text",
                        "text": "Now listen to this audio"
                    },
                    {
                        "type": "audio_url",
                        "audio_url": None,
                        "uuid": "audio_123",
                    },
                ],
            },
            {
                "role": "assistant",
                "content": "Some stuff."
            },
            {
                "role":
                "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's on this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": None,
                        "uuid": "image_123",
                    },
                    {
                        "type": "text",
                        "text": "And what's in the video?"
                    },
                    {
                        "type": "video_url",
                        "video_url": None,
                        "uuid": "video_123",
                    },
                ],
            },
        ],
        qwen25omni_model_config_mm_interleaved,
        qwen25omni_tokenizer,
        content_format="string",
    )

    assert conversation == [
        {
            "role":
            "user",
            "content":
            "What's on this image?\n<|vision_start|><|IMAGE|><|vision_end|>\n"
            "Now listen to this audio\nAudio 1: <|audio_bos|><|AUDIO|><|audio_eos|>",  # noqa: E501
        },
        {
            "role": "assistant",
            "content": "Some stuff."
        },
        {
            "role":
            "user",
            "content":
            "What's on this image?\n<|vision_start|><|IMAGE|><|vision_end|>\n"
            "And what's in the video?\n<|vision_start|><|VIDEO|><|vision_end|>",
        },
    ]

    _assert_mm_data_inputs(mm_data, {
        "image": 2,
        "video": 1,
        "audio": 1
    },
                           skipped_media_indices={
                               "image": [0, 1],
                               "video": [0],
                               "audio": [0]
                           })
    _assert_mm_uuids(mm_uuids,
                     2,
                     modality="image",
                     expected_uuids=["image_123", "image_123"])
    _assert_mm_uuids(mm_uuids,
                     1,
                     modality="video",
                     expected_uuids=["video_123"])
    _assert_mm_uuids(mm_uuids,
                     1,
                     modality="audio",
                     expected_uuids=["audio_123"])


def test_parse_chat_messages_multiple_modals_with_partial_uuids_multiple_messages_interleave(  # noqa: E501
    qwen25omni_model_config_mm_interleaved,
    qwen25omni_tokenizer,
    image_url,
    video_url,
    audio_url,
):
    conversation, mm_data, mm_uuids = parse_chat_messages(
        [
            {
                "role":
                "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's on this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        },
                        "uuid": "image_123",
                    },
                    {
                        "type": "text",
                        "text": "Now listen to this audio"
                    },
                    {
                        "type": "audio_url",
                        "audio_url": {
                            "url": audio_url
                        }
                    },
                ],
            },
            {
                "role": "assistant",
                "content": "Some stuff."
            },
            {
                "role":
                "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's on this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    },
                    {
                        "type": "text",
                        "text": "And what's in the video?"
                    },
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": video_url
                        },
                        "uuid": "video_123",
                    },
                ],
            },
        ],
        qwen25omni_model_config_mm_interleaved,
        qwen25omni_tokenizer,
        content_format="string",
    )

    assert conversation == [
        {
            "role":
            "user",
            "content":
            "What's on this image?\n<|vision_start|><|IMAGE|><|vision_end|>\n"
            "Now listen to this audio\nAudio 1: <|audio_bos|><|AUDIO|><|audio_eos|>",  # noqa: E501
        },
        {
            "role": "assistant",
            "content": "Some stuff."
        },
        {
            "role":
            "user",
            "content":
            "What's on this image?\n<|vision_start|><|IMAGE|><|vision_end|>\n"
            "And what's in the video?\n<|vision_start|><|VIDEO|><|vision_end|>",
        },
    ]

    _assert_mm_data_inputs(mm_data, {"image": 2, "video": 1, "audio": 1})
    _assert_mm_uuids(mm_uuids,
                     2,
                     modality="image",
                     expected_uuids=["image_123", None])
    _assert_mm_uuids(mm_uuids,
                     1,
                     modality="video",
                     expected_uuids=["video_123"])
    _assert_mm_uuids(mm_uuids, 1, modality="audio", expected_uuids=[None])


def test_parse_chat_messages_multiple_images_interleave_with_placeholders(
    phi3v_model_config_mm_interleaved,
    phi3v_tokenizer,
    image_url,
):
    with pytest.raises(
            ValueError,
            match=r"Found more '<|image_1|>' placeholders in input prompt "
            "than actual multimodal data items.",
    ):
        parse_chat_messages(
            [{
                "role":
                "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    },
                    {
                        "type":
                        "text",
                        "text":
                        "I need you to compare this image\n<|image_1|>\nand this one\n<|image_2|>\n"  # noqa: E501
                        "Do they have differences?",
                    },
                ],
            }],
            phi3v_model_config_mm_interleaved,
            phi3v_tokenizer,
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
        skip_tokenizer_init=model_info.skip_tokenizer_init,
        enforce_eager=model_info.enforce_eager,
        dtype=model_info.dtype)

    # Build the tokenizer
    tokenizer = get_tokenizer(
        model,
        trust_remote_code=model_config.trust_remote_code,
    )

    tools = ([{
        "type": "function",
        "function": {
            "name": "dummy_function_name",
            "description": "This is a dummy function",
            "parameters": sample_json_schema,
        },
    }] if use_tools else None)

    # Test detecting the tokenizer's chat_template
    chat_template = resolve_hf_chat_template(
        tokenizer,
        chat_template=None,
        tools=tools,
        model_config=model_config,
    )
    assert isinstance(chat_template, str)


# NOTE: Qwen2-Audio default chat template is specially defined inside
# processor class instead of using `tokenizer_config.json`
# yapf: disable
@pytest.mark.parametrize(
    ("model", "expected_format"),
    [(PHI3V_MODEL_ID, "string"),
     (QWEN2VL_MODEL_ID, "openai"),
     (QWEN25VL_MODEL_ID, "openai"),
     (ULTRAVOX_MODEL_ID, "string"),
     (QWEN2AUDIO_MODEL_ID, "openai"),
     (LLAMA_GUARD_MODEL_ID, "openai")],
)
# yapf: enable
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
        skip_tokenizer_init=model_info.skip_tokenizer_init,
        enforce_eager=model_info.enforce_eager,
        dtype=model_info.dtype)

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


# yapf: disable
@pytest.mark.parametrize(
    ("model", "expected_format"),
    [("Salesforce/blip2-opt-2.7b", "string"),
     ("facebook/chameleon-7b", "string"),
     ("deepseek-ai/deepseek-vl2-tiny", "string"),
     ("adept/fuyu-8b", "string"),
     ("google/paligemma-3b-mix-224", "string"),
     ("Qwen/Qwen-VL", "string"),
     ("Qwen/Qwen-VL-Chat", "string")],
)
# yapf: enable
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
        skip_tokenizer_init=model_info.skip_tokenizer_init,
        enforce_eager=model_info.enforce_eager,
        dtype=model_info.dtype)

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


# yapf: disable
@pytest.mark.parametrize(
    ("template_path", "expected_format"),
    [("template_alpaca.jinja", "string"),
     ("template_baichuan.jinja", "string"),
     ("template_chatglm.jinja", "string"),
     ("template_chatglm2.jinja", "string"),
     ("template_chatml.jinja", "string"),
     ("template_dse_qwen2_vl.jinja", "openai"),
     ("template_falcon_180b.jinja", "string"),
     ("template_falcon.jinja", "string"),
     ("template_inkbot.jinja", "string"),
     ("template_teleflm.jinja", "string"),
     ("template_vlm2vec.jinja", "openai"),
     ("tool_chat_template_granite_20b_fc.jinja", "string"),
     ("tool_chat_template_hermes.jinja", "string"),
     ("tool_chat_template_internlm2_tool.jinja", "string"),
     ("tool_chat_template_llama3.1_json.jinja", "openai"),
     ("tool_chat_template_llama3.2_json.jinja", "openai"),
     ("tool_chat_template_mistral_parallel.jinja", "string"),
     ("tool_chat_template_mistral.jinja", "string")],
)
# yapf: enable
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


def test_parse_chat_messages_include_thinking_chunk(mistral_model_config,
                                                    mistral_tokenizer):
    messages = [{
        "role":
        "system",
        "content": [{
            "type": "text",
            "text": "You are a helpful assistant."
        }, {
            "type":
            "thinking",
            "closed":
            True,
            "thinking":
            "Only return the answer when you are confident."
        }]
    }, {
        "role": "user",
        "content": "What is 2+2?"
    }, {
        "role":
        "assistant",
        "content": [{
            "type": "text",
            "text": "Let me think about it."
        }, {
            "type": "thinking",
            "closed": True,
            "thinking": "2+2 = 4"
        }, {
            "type": "text",
            "text": "The answer is 4.",
        }],
    }]

    conversation_with_thinking, _, _ = parse_chat_messages(
        messages,
        mistral_model_config,
        mistral_tokenizer,
        content_format="openai",
    )

    expected_conversation = [{
        "role":
        "system",
        "content": [{
            "type": "text",
            "text": "You are a helpful assistant."
        }, {
            "type": "text",
            "text": "Only return the answer when you are confident."
        }],
    }, {
        "role":
        "user",
        "content": [{
            "type": "text",
            "text": "What is 2+2?"
        }],
    }, {
        "role":
        "assistant",
        "content": [
            {
                "type": "text",
                "text": "Let me think about it."
            },
            {
                "type": "text",
                "text": "2+2 = 4"
            },
            {
                "type": "text",
                "text": "The answer is 4."
            },
        ]
    }]

    assert conversation_with_thinking == expected_conversation


def test_apply_mistral_chat_template_thinking_chunk():
    # Moved import here to avoid yapf and isort conflicts
    from vllm.entrypoints.chat_utils import apply_mistral_chat_template
    messages = [{
        "role":
        "system",
        "content": [{
            "type": "text",
            "text": "You are a helpful assistant."
        }, {
            "type":
            "thinking",
            "closed":
            True,
            "thinking":
            "Only return the answer when you are confident."
        }]
    }, {
        "role": "user",
        "content": "What is 2+2?"
    }, {
        "role":
        "assistant",
        "content": [{
            "type": "text",
            "text": "Let me think about it."
        }, {
            "type": "thinking",
            "closed": True,
            "thinking": "2+2 = 4"
        }, {
            "type": "text",
            "text": "The answer is 4.",
        }],
    }, {
        "role": "user",
        "content": "Thanks, what is 3+3?"
    }]

    # TODO(Julien): upon model release change to a tokenizer already configured.
    # =================================================================
    mistral_tokenizer = MistralTokenizer.from_pretrained(
        "mistralai/Devstral-Small-2507")
    assert isinstance(mistral_tokenizer.tokenizer, Tekkenizer)
    # Add think special tokens to the tokenizer
    mistral_tokenizer.tokenizer._all_special_tokens[35] = SpecialTokenInfo(
        rank=35, is_control=True, token_str=SpecialTokens.begin_think.value)
    mistral_tokenizer.tokenizer._all_special_tokens[36] = SpecialTokenInfo(
        rank=36, is_control=True, token_str=SpecialTokens.end_think.value)
    mistral_tokenizer.tokenizer._special_tokens_reverse_vocab = {
        k: v
        for k, v in
        mistral_tokenizer.tokenizer._special_tokens_reverse_vocab.items()
        if v not in {35, 36}
    }
    mistral_tokenizer.tokenizer._special_tokens_reverse_vocab[
        SpecialTokens.begin_think.value] = 35
    mistral_tokenizer.tokenizer._special_tokens_reverse_vocab[
        SpecialTokens.end_think.value] = 36
    mistral_tokenizer.instruct.BEGIN_THINK = 35
    mistral_tokenizer.instruct.END_THINK = 36
    # =================================================================

    tokens_ids = apply_mistral_chat_template(mistral_tokenizer,
                                             messages,
                                             chat_template=None,
                                             tools=None)

    string_tokens = mistral_tokenizer.mistral.decode(
        tokens_ids, special_token_policy=SpecialTokenPolicy.KEEP)

    expected_tokens = (
        r"<s>[SYSTEM_PROMPT]You are a helpful assistant.[THINK]Only return the"
        r" answer when you are confident.[/THINK][/SYSTEM_PROMPT]"
        r"[INST]What is 2+2?[/INST]"
        r"Let me think about it.[THINK]2+2 = 4[/THINK]The answer is 4.</s>"
        r"[INST]Thanks, what is 3+3?[/INST]")

    assert string_tokens == expected_tokens


def test_parse_chat_messages_single_empty_audio_with_uuid(
    qwen2_audio_model_config,
    qwen2_audio_tokenizer,
):
    audio_uuid = "abcd"
    conversation, mm_data, mm_uuids = parse_chat_messages(
        [{
            "role":
            "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {},
                    "uuid": audio_uuid,
                },
                {
                    "type": "text",
                    "text": "What does the audio say?"
                },
            ],
        }],
        qwen2_audio_model_config,
        qwen2_audio_tokenizer,
        content_format="string",
    )

    assert conversation == [{
        "role":
        "user",
        "content":
        "Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\nWhat does the audio say?"
    }]
    _assert_mm_data_inputs(mm_data, {"audio": 1})
    _assert_mm_uuids(mm_uuids,
                     1,
                     modality="audio",
                     expected_uuids=[audio_uuid])


@pytest.mark.asyncio
async def test_parse_chat_messages_single_empty_audio_with_uuid_async(
    qwen2_audio_model_config,
    qwen2_audio_tokenizer,
):
    audio_uuid = "abcd"
    conversation, mm_future, mm_uuids = parse_chat_messages_futures(
        [{
            "role":
            "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {},
                    "uuid": audio_uuid,
                },
                {
                    "type": "text",
                    "text": "What does the audio say?"
                },
            ],
        }],
        qwen2_audio_model_config,
        qwen2_audio_tokenizer,
        content_format="string",
    )

    assert conversation == [{
        "role":
        "user",
        "content":
        "Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\nWhat does the audio say?"
    }]
    _assert_mm_data_inputs(await mm_future, {"audio": 1})
    _assert_mm_uuids(mm_uuids,
                     1,
                     modality="audio",
                     expected_uuids=[audio_uuid])
