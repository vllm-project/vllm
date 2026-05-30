# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for hoisting ``role="system"`` messages into ``system``.

Anthropic's spec keeps the system prompt in the top-level ``system`` field and
only allows ``user``/``assistant`` roles inside ``messages[]``. Some clients
(notably Claude Code) instead emit the system prompt as a ``role="system"``
entry inside ``messages[]``. ``AnthropicMessagesRequest.hoist_system_messages``
is a lenient-compat ``model_validator(mode="before")`` that pulls such entries
out into the top-level ``system`` field before per-message validation runs.
"""

import pytest
from pydantic import ValidationError

from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages

_convert = AnthropicServingMessages._convert_anthropic_to_openai_request


def _make_request(messages: list[dict], **kwargs) -> AnthropicMessagesRequest:
    return AnthropicMessagesRequest(
        model="test-model",
        max_tokens=128,
        messages=messages,
        **kwargs,
    )


def _system_texts(request: AnthropicMessagesRequest) -> list[tuple[str, str]]:
    """Flatten the (parsed) ``system`` field to ``(type, text)`` pairs."""
    assert not isinstance(request.system, str)
    return [(block.type, block.text) for block in request.system]


class TestHoistSystemMessages:
    def test_string_system_message_is_hoisted(self):
        """A role="system" entry with string content moves to ``system``."""
        request = _make_request(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
            ]
        )
        assert _system_texts(request) == [("text", "You are helpful.")]
        assert [m.role for m in request.messages] == ["user"]

    def test_block_list_system_message_is_hoisted(self):
        """A role="system" entry with content blocks is preserved as blocks."""
        request = _make_request(
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "Be terse."}],
                },
                {"role": "user", "content": "Hi"},
            ]
        )
        assert _system_texts(request) == [("text", "Be terse.")]
        assert len(request.messages) == 1

    def test_existing_top_level_system_is_kept_first(self):
        """Pre-existing top-level system stays ahead of hoisted entries."""
        request = _make_request(
            [
                {"role": "system", "content": "From messages."},
                {"role": "user", "content": "Hi"},
            ],
            system="From top level.",
        )
        assert _system_texts(request) == [
            ("text", "From top level."),
            ("text", "From messages."),
        ]

    def test_multiple_system_messages_preserve_order(self):
        request = _make_request(
            [
                {"role": "system", "content": "First."},
                {"role": "user", "content": "Hi"},
                {"role": "system", "content": "Second."},
                {"role": "assistant", "content": "Hello"},
            ]
        )
        assert _system_texts(request) == [
            ("text", "First."),
            ("text", "Second."),
        ]
        assert [m.role for m in request.messages] == ["user", "assistant"]

    def test_no_system_message_leaves_request_unchanged(self):
        request = _make_request(
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ]
        )
        assert request.system is None
        assert [m.role for m in request.messages] == ["user", "assistant"]

    def test_spec_compliant_top_level_system_untouched(self):
        """Requests that already follow the spec are not modified."""
        request = _make_request(
            [{"role": "user", "content": "Hi"}],
            system="You are helpful.",
        )
        assert request.system == "You are helpful."
        assert len(request.messages) == 1

    def test_invalid_role_still_rejected(self):
        """Non-system invalid roles must still fail validation."""
        with pytest.raises(ValidationError):
            _make_request([{"role": "tool", "content": "x"}])

    def test_hoisted_system_flows_through_conversion(self):
        """End-to-end: a hoisted system message becomes an OpenAI system msg."""
        request = _make_request(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
            ]
        )
        result = _convert(request)
        assert result.messages[0] == {
            "role": "system",
            "content": "You are helpful.",
        }
        assert result.messages[1]["role"] == "user"
