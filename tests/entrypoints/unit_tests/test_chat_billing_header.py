# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Claude Code's attribution header in chat completions system messages."""

from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.chat_utils import _parse_chat_message_content

HEADER = (
    "x-anthropic-billing-header: cc_version=2.1.220.300; "
    "cc_entrypoint=claude-desktop; cch=7adab;"
)


def _parse(message, content_format="openai"):
    tracker = MagicMock()
    tracker.create_parser.return_value.mm_placeholder_storage.return_value = {}
    return _parse_chat_message_content(
        message, tracker, content_format, interleave_strings=False
    )


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize(
    "header",
    [
        HEADER,
        "x-anthropic-billing-header: cc_version=2.1.286.99b; cc_entrypoint=sdk-cli;",
    ],
)
def test_system_header_part_is_dropped(header):
    (msg,) = _parse(
        {
            "role": "system",
            "content": [
                {"type": "text", "text": header},
                {"type": "text", "text": "You are a Claude agent."},
            ],
        }
    )
    assert msg["content"] == [{"type": "text", "text": "You are a Claude agent."}]


@pytest.mark.skip_global_cleanup
def test_system_header_line_is_stripped_from_string():
    (msg,) = _parse(
        {"role": "system", "content": HEADER + "\nYou are a Claude agent."},
        content_format="string",
    )
    assert msg["content"] == "You are a Claude agent."


@pytest.mark.skip_global_cleanup
def test_requests_differing_only_in_cch_parse_identically():
    def system(cch):
        return {
            "role": "system",
            "content": HEADER.replace("7adab", cch) + "\nYou are a Claude agent.",
        }

    assert _parse(system("7adab")) == _parse(system("986dd"))


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize(
    "message",
    [
        {"role": "system", "content": "You are a Claude agent."},
        {"role": "system", "content": "Quote: " + HEADER},
        {"role": "user", "content": HEADER + "Hello"},
    ],
)
def test_other_content_is_unchanged(message):
    (msg,) = _parse(message, content_format="string")
    assert msg["content"] == message["content"]
