# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from unittest.mock import MagicMock

import pybase64
import pytest

from vllm.config import ModelConfig
from vllm.exceptions import VLLMValidationError
from vllm.multimodal.media import MEDIA_CONNECTOR_REGISTRY
from vllm.renderers.chat_utils import (
    ChatCompletionMessageParam,
    has_non_text_content,
    parse_chat_messages,
    parse_chat_messages_async,
)
from vllm.renderers.chat_utils.content_parser import PROMPT_EMBEDS_PLACEHOLDER_TOKEN

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


def _b64(data: bytes) -> str:
    return pybase64.b64encode(data).decode()


@pytest.mark.parametrize("content_format", ["string", "openai"])
def test_parse_chat_messages_file_part(mistral_model_config, content_format):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Summarize:"},
                {
                    "type": "file",
                    "file": {
                        "file_data": f"data:text/plain;base64,{_b64(b'hello')}",
                    },
                },
            ],
        }
    ]
    assert not has_non_text_content(messages)

    conversation, mm_data, _ = parse_chat_messages(
        messages, mistral_model_config, content_format=content_format
    )

    assert mm_data is None
    if content_format == "string":
        assert conversation == [{"role": "user", "content": "Summarize:\nhello"}]
    else:
        assert conversation == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Summarize:"},
                    {"type": "text", "text": "hello"},
                ],
            }
        ]


def test_parse_chat_messages_input_file_part(mistral_model_config):
    # The filename is used to infer the media type of raw base64 data and is
    # shown to the model; other file fields must not leak into the template.
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_file",
                    "filename": "notes.txt",
                    "file_data": _b64("héllo".encode()),
                }
            ],
        }
    ]
    assert not has_non_text_content(messages)

    conversation, _, _ = parse_chat_messages(
        messages, mistral_model_config, content_format="openai"
    )

    assert conversation == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "File: notes.txt\nhéllo"}],
        }
    ]


@pytest.mark.parametrize(
    ("filename", "prefix"),
    [
        ("README.md", ""),
        ("config.yml", ""),
        ("notes.md", "data:application/octet-stream;base64,"),
    ],
)
def test_parse_chat_messages_file_part_text_extension(
    mistral_model_config, filename, prefix
):
    # Common text extensions are accepted for raw base64 or a generic data URL
    # type, even where `mimetypes` does not know them.
    part = {
        "type": "input_file",
        "filename": filename,
        "file_data": prefix + _b64(b"a: 1"),
    }

    conversation, _, _ = parse_chat_messages(
        [{"role": "user", "content": [part]}],
        mistral_model_config,
        content_format="openai",
    )

    assert conversation[0]["content"] == [
        {"type": "text", "text": f"File: {filename}\na: 1"}
    ]


@pytest.mark.parametrize(
    ("file", "match"),
    [
        ({"file_id": "file-abc"}, "'file_id' or 'file_url'"),
        ({}, "requires 'file_data'"),
        ({"filename": "a.pdf", "file_data": _b64(b"%PDF")}, "only text files"),
        ({"file_data": "data:text/plain;base64,not base64!"}, "Invalid base64"),
        ({"file_data": "data:text/plain;base64,/w=="}, "not valid UTF-8"),
    ],
)
@pytest.mark.parametrize("part_type", ["file", "input_file"])
def test_parse_chat_messages_rejects_unsupported_file(
    mistral_model_config, part_type, file, match
):
    part = (
        {"type": "file", "file": file}
        if part_type == "file"
        else {"type": "input_file", **file}
    )
    with pytest.raises(VLLMValidationError, match=match):
        parse_chat_messages(
            [{"role": "user", "content": [part]}],
            mistral_model_config,
            content_format="openai",
        )


def test_parse_chat_messages_rejects_placeholder_in_file():
    model_config = MagicMock()
    model_config.enable_prompt_embeds = True
    model_config.multimodal_config = None
    text = f"leading {PROMPT_EMBEDS_PLACEHOLDER_TOKEN}".encode()
    part = {
        "type": "file",
        "file": {"file_data": f"data:text/plain;base64,{_b64(text)}"},
    }

    with pytest.raises(VLLMValidationError, match="reserved placeholder"):
        parse_chat_messages(
            [{"role": "user", "content": [part]}],
            model_config,
            content_format="openai",
        )
