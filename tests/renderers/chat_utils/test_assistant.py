# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock

from vllm.renderers.chat_utils import ConversationMessage, parse_chat_message_content


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
    """``reasoning`` is mirrored to ``reasoning_content`` for legacy templates."""
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
