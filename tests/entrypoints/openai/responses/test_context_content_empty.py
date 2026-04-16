# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test that _extract_content_text handles empty content safely."""

import pytest

from vllm.entrypoints.openai.responses.context import _extract_content_text


class FakeContent:
    def __init__(self, text: str):
        self.text = text


class FakeMessage:
    def __init__(self, content=None):
        self.content = content if content is not None else []


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ([FakeContent("hello")], "hello"),
        ([FakeContent("tool output")], "tool output"),
        ([], ""),
        (None, ""),
    ],
    ids=[
        "single_text_item",
        "tool_result_text",
        "empty_list",
        "none_content",
    ],
)
def test_extract_content_text(content, expected):
    msg = FakeMessage(content=content)
    assert _extract_content_text(msg) == expected


def test_original_code_would_crash():
    """Demonstrate that direct content[0] access crashes on empty list."""
    msg = FakeMessage(content=[])
    with pytest.raises(IndexError):
        _ = msg.content[0].text
