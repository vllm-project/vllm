# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock

import pytest

from vllm.config import ModelConfig
from vllm.multimodal.media import MEDIA_CONNECTOR_REGISTRY
from vllm.renderers.chat_utils import (
    ChatCompletionMessageParam,
    parse_chat_messages,
    parse_chat_messages_async,
)

MISTRAL_MODEL_ID = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"


@pytest.fixture(scope="function")
def mistral_model_config():
    return ModelConfig(
        MISTRAL_MODEL_ID,
        runner="generate",
        limit_mm_per_prompt={
            "image": 2,
        },
    )


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
async def test_text_only_chat_does_not_initialize_media_connector(
    mistral_model_config,
    monkeypatch,
):
    load_connector = MagicMock()
    monkeypatch.setattr(MEDIA_CONNECTOR_REGISTRY, "load", load_connector)
    messages: list[ChatCompletionMessageParam] = [
        {"role": "user", "content": "Who are you?"}
    ]

    parse_chat_messages(
        messages,
        mistral_model_config,
        content_format="string",
    )
    await parse_chat_messages_async(
        messages,
        mistral_model_config,
        content_format="string",
    )

    load_connector.assert_not_called()


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
