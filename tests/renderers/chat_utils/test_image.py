# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import warnings

import pytest
import torch

from vllm.assets.image import ImageAsset
from vllm.config import ModelConfig
from vllm.exceptions import VLLMValidationError
from vllm.renderers.chat_utils import parse_chat_messages, parse_chat_messages_async
from vllm.utils.serial_utils import tensor2base64

from .utils import (
    assert_mm_data_is_image_input,
    assert_mm_uuids,
)

PHI3V_MODEL_ID = "microsoft/Phi-3.5-vision-instruct"


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
    assert_mm_data_is_image_input(mm_data, 1)
    assert_mm_uuids(mm_uuids, 1, expected_uuids=[None])


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
    assert_mm_data_is_image_input(mm_data, 1)
    assert_mm_uuids(mm_uuids, 1, expected_uuids=[image_uuid])


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
    assert_mm_data_is_image_input(mm_data, 1, skipped_image_indices=[0])
    assert_mm_uuids(mm_uuids, 1, expected_uuids=[image_uuid])


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
    assert_mm_data_is_image_input(mm_data, 1)
    assert_mm_uuids(mm_uuids, 1, expected_uuids=[None])


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
    assert_mm_data_is_image_input(mm_data, 2)
    assert_mm_uuids(mm_uuids, 2, expected_uuids=[image_uuid1, image_uuid2])


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
    assert_mm_data_is_image_input(mm_data, 2, skipped_image_indices=[0, 1])
    assert_mm_uuids(mm_uuids, 2, expected_uuids=[image_uuid1, image_uuid2])


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
    assert_mm_data_is_image_input(mm_data, 2, skipped_image_indices=[1])
    assert_mm_uuids(mm_uuids, 2, expected_uuids=[image_uuid1, image_uuid2])


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
    assert_mm_data_is_image_input(mm_data, 1)
    assert_mm_uuids(mm_uuids, 1, expected_uuids=[image_uuid])


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
    assert_mm_data_is_image_input(mm_data, 1, skipped_image_indices=[0])
    assert_mm_uuids(mm_uuids, 1, expected_uuids=[image_uuid])


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
    assert_mm_data_is_image_input(mm_data, 2)
    assert_mm_uuids(mm_uuids, 2, expected_uuids=[image_uuid1, image_uuid2])


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
    assert_mm_data_is_image_input(mm_data, 2, skipped_image_indices=[0, 1])
    assert_mm_uuids(mm_uuids, 2, expected_uuids=[image_uuid1, image_uuid2])


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
    assert_mm_data_is_image_input(mm_data, 2)
    assert_mm_uuids(mm_uuids, 2, expected_uuids=[None, image_uuid2])


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
    assert_mm_data_is_image_input(mm_data, 1)
    assert_mm_uuids(mm_uuids, 1, expected_uuids=[None])


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
    assert_mm_data_is_image_input(mm_data, 2)
    assert_mm_uuids(mm_uuids, 2, expected_uuids=[None, None])


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
    assert_mm_data_is_image_input(mm_data, 1, skipped_image_indices=[0])
    assert_mm_uuids(mm_uuids, 1, expected_uuids=[uuid])


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

    assert_mm_uuids(mm_uuids, 1, expected_uuids=[uuid])


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
    assert_mm_uuids(mm_uuids, 2, expected_uuids=[None, None])


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
    assert_mm_uuids(mm_uuids, 2, expected_uuids=[uuid1, uuid2])


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
    assert_mm_uuids(mm_uuids, 2, expected_uuids=[None, None])


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

    assert_mm_uuids(mm_uuids, 1, expected_uuids=[uuid])


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
    assert_mm_uuids(mm_uuids, 1, expected_uuids=[None])


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
    assert_mm_data_is_image_input(mm_data, 2)
    assert_mm_uuids(mm_uuids, 2, expected_uuids=[None, None])


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
    assert_mm_data_is_image_input(mm_data, 2)
    assert_mm_uuids(mm_uuids, 2, expected_uuids=[None, None])


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
    assert_mm_data_is_image_input(mm_data, 2)
    assert_mm_uuids(mm_uuids, 2, expected_uuids=[None, None])


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
    assert_mm_data_is_image_input(mm_data, 2)
    assert_mm_uuids(mm_uuids, 2, expected_uuids=[None, None])


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
    assert_mm_data_is_image_input(mm_data, 2)
    assert_mm_uuids(mm_uuids, 2, expected_uuids=[image_uuid, image_uuid])


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


def test_parse_chat_messages_openai_format_image_url(
    phi3v_model_config,
    image_url,
):
    content = [
        {"type": "image_url", "image_url": {"url": image_url}},
        {"type": "text", "text": "What's in the image?"},
    ]
    conversation, mm_data, mm_uuids = parse_chat_messages(
        [
            {
                "role": "user",
                "content": content,
            }
        ],
        phi3v_model_config,
        content_format="openai",
    )

    assert conversation == [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What's in the image?"},
            ],
        }
    ]
    assert_mm_data_is_image_input(mm_data, 1)
    assert_mm_uuids(mm_uuids, 1, expected_uuids=[None])


def test_parse_chat_messages_rejects_too_many_images_in_one_message(
    phi3v_model_config,
    image_url,
):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="coroutine 'async_get_and_parse_image' was never awaited",
        )
        with pytest.raises(VLLMValidationError, match="At most"):
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
        with pytest.raises(VLLMValidationError, match="At most"):
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
    assert_mm_data_is_image_input(mm_data, 2)
    assert_mm_uuids(mm_uuids, 2, expected_uuids=[None, None])


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
    assert_mm_data_is_image_input(mm_data, 2)
    assert_mm_uuids(mm_uuids, 2, expected_uuids=[None, None])


def test_parse_chat_messages_interleave_placeholder_overcount_raises(
    phi3v_model_config_mm_interleaved,
    image_url,
):
    # Only one image is supplied, but the user's text also contains the internal
    # image sentinel, so the interleaved prompt references more image
    # placeholders than actual images. This must surface as a clear validation
    # error rather than an IndexError from popping an empty list.
    with pytest.raises(VLLMValidationError, match="placeholders in input prompt"):
        parse_chat_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": "<##IMAGE##>"},
                    ],
                }
            ],
            phi3v_model_config_mm_interleaved,
            content_format="string",
        )


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
    assert_mm_data_is_image_input(mm_data, 2)
    assert_mm_uuids(mm_uuids, 2, expected_uuids=[None, None])


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
    assert_mm_data_is_image_input(mm_data, 2)
    assert_mm_uuids(mm_uuids, 2, expected_uuids=[image_uuid, image_uuid])


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
    assert_mm_data_is_image_input(mm_data, 2)
    assert_mm_uuids(mm_uuids, 2, expected_uuids=[None, None])


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
    assert_mm_data_is_image_input(mm_data, 2)
    assert_mm_uuids(mm_uuids, 2, expected_uuids=[image_uuid, image_uuid])


def test_parse_chat_messages_multiple_images_interleave_with_placeholders(
    phi3v_model_config_mm_interleaved,
    image_url,
):
    with pytest.raises(
        VLLMValidationError,
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
