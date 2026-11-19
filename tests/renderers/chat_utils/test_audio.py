# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.config import ModelConfig
from vllm.renderers.chat_utils import parse_chat_messages, parse_chat_messages_async
from vllm.utils.serial_utils import tensor2base64

from .utils import (
    assert_mm_data_inputs,
    assert_mm_uuids,
)

QWEN2AUDIO_MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"


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


@pytest.mark.parametrize("input_audio", [None, {}, {"data": "", "format": "wav"}])
def test_parse_chat_messages_single_empty_audio_with_uuid(
    qwen2_audio_model_config,
    input_audio,
):
    audio_uuid = "abcd"
    conversation, mm_data, mm_uuids = parse_chat_messages(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": input_audio,
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
    assert_mm_data_inputs(mm_data, {"audio": 1})
    assert_mm_uuids(mm_uuids, 1, modality="audio", expected_uuids=[audio_uuid])


@pytest.mark.asyncio
@pytest.mark.parametrize("input_audio", [None, {}, {"data": "", "format": "wav"}])
async def test_parse_chat_messages_single_empty_audio_with_uuid_async(
    qwen2_audio_model_config,
    input_audio,
):
    audio_uuid = "abcd"
    conversation, mm_data, mm_uuids = await parse_chat_messages_async(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": input_audio,
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
    assert_mm_data_inputs(mm_data, {"audio": 1})
    assert_mm_uuids(mm_uuids, 1, modality="audio", expected_uuids=[audio_uuid])


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
    assert_mm_uuids(mm_uuids, 1, modality="audio", expected_uuids=[uuid])


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
    assert_mm_uuids(mm_uuids, 1, modality="audio", expected_uuids=[None])


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
    assert_mm_uuids(mm_uuids, 1, modality="audio", expected_uuids=[None])
