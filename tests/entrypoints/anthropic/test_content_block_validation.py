# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for AnthropicContentBlock per-type field validation.

Guards that tool_use blocks missing required fields raise a 422-able
ValidationError, and that all other block types are unaffected.
"""

import pytest
from pydantic import ValidationError

from vllm.entrypoints.anthropic.protocol import AnthropicContentBlock

pytestmark = pytest.mark.skip_global_cleanup


# ======================================================================
# tool_use — required fields enforced
# ======================================================================


def test_tool_use_missing_id_raises():
    with pytest.raises(ValidationError, match="non-empty 'id'"):
        AnthropicContentBlock(
            type="tool_use",
            name="get_weather",
            input={"city": "NYC"},
        )


def test_tool_use_empty_id_raises():
    with pytest.raises(ValidationError, match="non-empty 'id'"):
        AnthropicContentBlock(
            type="tool_use",
            id="",
            name="get_weather",
            input={"city": "NYC"},
        )


def test_tool_use_missing_name_raises():
    with pytest.raises(ValidationError, match="non-empty 'name'"):
        AnthropicContentBlock(
            type="tool_use",
            id="toolu_01",
            input={"city": "NYC"},
        )


def test_tool_use_empty_name_raises():
    with pytest.raises(ValidationError, match="non-empty 'name'"):
        AnthropicContentBlock(
            type="tool_use",
            id="toolu_01",
            name="",
            input={"city": "NYC"},
        )


def test_tool_use_missing_input_raises():
    with pytest.raises(ValidationError, match="'input'"):
        AnthropicContentBlock(
            type="tool_use",
            id="toolu_01",
            name="get_weather",
        )


def test_tool_use_valid_passes():
    block = AnthropicContentBlock(
        type="tool_use",
        id="toolu_01",
        name="get_weather",
        input={"city": "NYC"},
    )
    assert block.name == "get_weather"
    assert block.id == "toolu_01"
    assert block.input == {"city": "NYC"}


def test_tool_use_empty_input_dict_passes():
    block = AnthropicContentBlock(
        type="tool_use",
        id="toolu_01",
        name="ping",
        input={},
    )
    assert block.input == {}


# ======================================================================
# text — text must not be None
# ======================================================================


def test_text_block_none_text_raises():
    with pytest.raises(ValidationError, match="text block requires 'text'"):
        AnthropicContentBlock(type="text")


def test_text_block_empty_string_passes():
    block = AnthropicContentBlock(type="text", text="")
    assert block.text == ""


def test_text_block_with_content_passes():
    block = AnthropicContentBlock(type="text", text="hello")
    assert block.text == "hello"


# ======================================================================
# image — source must not be None
# ======================================================================


def test_image_block_missing_source_raises():
    with pytest.raises(ValidationError, match="image block requires 'source'"):
        AnthropicContentBlock(type="image")


def test_image_block_with_source_passes():
    block = AnthropicContentBlock(
        type="image",
        source={"type": "url", "url": "https://example.com/img.png"},
    )
    assert block.source is not None


# ======================================================================
# Unvalidated types — no new constraints introduced
# ======================================================================


def test_tool_result_without_tool_use_id_passes():
    block = AnthropicContentBlock(type="tool_result")
    assert block.type == "tool_result"


def test_thinking_block_passes_with_no_extra_fields():
    block = AnthropicContentBlock(type="thinking")
    assert block.thinking is None


def test_redacted_thinking_block_passes():
    block = AnthropicContentBlock(type="redacted_thinking", data="opaque")
    assert block.data == "opaque"


def test_tool_reference_block_passes():
    block = AnthropicContentBlock(type="tool_reference", tool_name="search")
    assert block.tool_name == "search"
