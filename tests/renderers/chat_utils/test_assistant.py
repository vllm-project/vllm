# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

import pytest
import requests

from tests.utils import RemoteOpenAIServer
from vllm.config import ModelConfig
from vllm.entrypoints.serve.tokenize.protocol import TokenizeChatRequest
from vllm.renderers.chat_utils import ConversationMessage, parse_chat_message_content
from vllm.renderers.chat_utils.item_tracker import MultiModalItemTracker

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"


@pytest.fixture(scope="module")
def refusal_model_config():
    return ModelConfig(MODEL_NAME)


@pytest.fixture(scope="module")
def server():
    args = ["--dtype", "float32", "--max-model-len", "256", "--enforce-eager"]
    with RemoteOpenAIServer(
        MODEL_NAME,
        args,
        env_dict={
            "VLLM_USE_RUST_FRONTEND": "0",
            "VLLM_CPU_KVCACHE_SPACE": "1",
        },
    ) as remote_server:
        yield remote_server


def _parse_one(message) -> ConversationMessage:
    # Plain-text content only reads these two tracker attributes.
    tracker = MagicMock()
    parser = tracker.create_parser.return_value
    parser.model_config.enable_prompt_embeds = False
    parser.mm_placeholder_storage.return_value = {}
    result = parse_chat_message_content(
        message, tracker, content_format="string", interleave_strings=False
    )
    assert len(result) == 1
    return result[0]


def test_assistant_reasoning_is_mirrored_to_reasoning_content():
    """`reasoning` is mirrored to `reasoning_content` for legacy templates."""
    conv_msg = _parse_one(
        {
            "role": "assistant",
            "content": "4",
            "reasoning": "2+2 equals 4",
        }
    )

    assert conv_msg.get("reasoning") == "2+2 equals 4"
    assert conv_msg.get("reasoning_content") == "2+2 equals 4"


def test_assistant_without_reasoning_sets_neither_field():
    """No reasoning in, no reasoning keys out: no empty thinking block in templates."""
    conv_msg = _parse_one({"role": "assistant", "content": "Hi"})

    assert "reasoning" not in conv_msg
    assert "reasoning_content" not in conv_msg


def test_non_assistant_reasoning_is_not_mirrored():
    """Only assistant turns carry reasoning; a user turn must not."""
    conv_msg = _parse_one(
        {"role": "user", "content": "Hi", "reasoning": "should be ignored"}
    )

    assert "reasoning" not in conv_msg
    assert "reasoning_content" not in conv_msg


def _parse_refusal_message(
    message, model_config, content_format
) -> ConversationMessage:
    result = parse_chat_message_content(
        message,
        MultiModalItemTracker(model_config),
        content_format=content_format,
        interleave_strings=False,
    )
    assert len(result) == 1
    return result[0]


@pytest.mark.parametrize("content", [None, "", []], ids=["null", "string", "list"])
@pytest.mark.parametrize("content_format", ["string", "openai"])
@pytest.mark.parametrize("validate_request", [False, True], ids=["dict", "request"])
def test_assistant_refusal_preserves_empty_content_history(
    refusal_model_config, content, content_format, validate_request
):
    """Replayed refusals retain the same text as explicit refusal content parts."""
    refusal = "I cannot provide private details, but can suggest public sources."
    message = {"role": "assistant", "content": content, "refusal": refusal}
    if validate_request:
        message = TokenizeChatRequest(messages=[message]).messages[0]
    expected = _parse_refusal_message(
        {
            "role": "assistant",
            "content": [{"type": "refusal", "refusal": refusal}],
        },
        refusal_model_config,
        content_format,
    )

    assert (
        _parse_refusal_message(message, refusal_model_config, content_format)
        == expected
    )


@pytest.mark.parametrize("content_kind", ["string", "list", "iterator"])
@pytest.mark.parametrize("content_format", ["string", "openai"])
def test_assistant_refusal_preserves_nonempty_content(
    refusal_model_config, content_kind, content_format
):
    """Existing content, including one-shot iterators, takes precedence."""
    text = "Here is the existing answer."
    parts = [{"type": "text", "text": text}]
    content = (
        text
        if content_kind == "string"
        else iter(parts)
        if content_kind == "iterator"
        else parts
    )
    expected = _parse_refusal_message(
        {"role": "assistant", "content": parts}, refusal_model_config, content_format
    )

    assert (
        _parse_refusal_message(
            {"role": "assistant", "content": content, "refusal": "Do not append me."},
            refusal_model_config,
            content_format,
        )
        == expected
    )


@pytest.mark.parametrize("content", [None, "", []], ids=["null", "string", "list"])
def test_tokenize_chat_preserves_assistant_refusal(server, content):
    """The Python renderer must replay refusals through the real HTTP entry point."""
    refusal = "I cannot provide private details, but can suggest public sources."
    results = []
    for assistant in (
        {"role": "assistant", "content": content, "refusal": refusal},
        {
            "role": "assistant",
            "content": [{"type": "refusal", "refusal": refusal}],
        },
    ):
        response = requests.post(
            server.url_for("tokenize"),
            json={
                "model": MODEL_NAME,
                "messages": [
                    {"role": "user", "content": "Can you share private details?"},
                    assistant,
                    {"role": "user", "content": "What alternative did you suggest?"},
                ],
            },
        )
        response.raise_for_status()
        results.append(response.json())

    assert results[0] == results[1]
