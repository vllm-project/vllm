"""Test that _extract_content_text handles empty content safely."""
import pytest

from vllm.entrypoints.openai.responses.context import _extract_content_text


class FakeContent:
    def __init__(self, text: str):
        self.text = text


class FakeMessage:
    def __init__(self, content=None):
        self.content = content if content is not None else []


def test_extract_content_text_with_content():
    msg = FakeMessage(content=[FakeContent("hello")])
    assert _extract_content_text(msg) == "hello"


def test_extract_content_text_empty_content():
    msg = FakeMessage(content=[])
    assert _extract_content_text(msg) == ""


def test_extract_content_text_none_like():
    """Content that is falsy should return empty string."""
    msg = FakeMessage(content=None)
    # content=None → not msg.content is True → returns ""
    assert _extract_content_text(msg) == ""


def test_original_code_would_crash():
    """Demonstrate that direct content[0] access crashes on empty list."""
    msg = FakeMessage(content=[])
    with pytest.raises(IndexError):
        _ = msg.content[0].text
