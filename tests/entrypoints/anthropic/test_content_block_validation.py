# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
# tool_result — tool_use_id required
# ======================================================================


def test_tool_result_missing_tool_use_id_raises():
    with pytest.raises(ValidationError, match="non-empty 'tool_use_id'"):
        AnthropicContentBlock(type="tool_result")


def test_tool_result_empty_tool_use_id_raises():
    with pytest.raises(ValidationError, match="non-empty 'tool_use_id'"):
        AnthropicContentBlock(type="tool_result", tool_use_id="")


def test_tool_result_valid_passes():
    block = AnthropicContentBlock(
        type="tool_result",
        tool_use_id="toolu_01",
    )
    assert block.tool_use_id == "toolu_01"


def test_tool_result_with_content_passes():
    block = AnthropicContentBlock(
        type="tool_result",
        tool_use_id="toolu_01",
        content="result text",
    )
    assert block.content == "result text"


# ======================================================================
# text — text field required
# ======================================================================


def test_text_block_none_text_raises():
    with pytest.raises(ValidationError, match="requires 'text'"):
        AnthropicContentBlock(type="text")


def test_text_block_empty_string_passes():
    block = AnthropicContentBlock(type="text", text="")
    assert block.text == ""


def test_text_block_with_content_passes():
    block = AnthropicContentBlock(type="text", text="hello")
    assert block.text == "hello"


# ======================================================================
# image — source required
# ======================================================================


def test_image_block_missing_source_raises():
    with pytest.raises(ValidationError, match="requires 'source'"):
        AnthropicContentBlock(type="image")


def test_image_block_with_source_passes():
    block = AnthropicContentBlock(
        type="image",
        source={"type": "url", "url": "https://example.com/img.png"},
    )
    assert block.source is not None


# ======================================================================
# thinking — thinking field required (signature is internally generated)
# ======================================================================


def test_thinking_block_missing_thinking_raises():
    with pytest.raises(ValidationError, match="requires 'thinking'"):
        AnthropicContentBlock(type="thinking", signature="abc")


def test_thinking_block_valid_passes():
    block = AnthropicContentBlock(
        type="thinking",
        thinking="I need to think about this",
        signature="sig_123",
    )
    assert block.thinking == "I need to think about this"


def test_thinking_block_empty_string_passes():
    block = AnthropicContentBlock(type="thinking", thinking="")
    assert block.thinking == ""


def test_thinking_block_no_signature_passes():
    """Signature is internally generated by _ActiveBlockState in streaming."""
    block = AnthropicContentBlock(type="thinking", thinking="hmm")
    assert block.thinking == "hmm"
    assert block.signature is None


# ======================================================================
# redacted_thinking — data required
# ======================================================================


def test_redacted_thinking_missing_data_raises():
    with pytest.raises(ValidationError, match="non-empty 'data'"):
        AnthropicContentBlock(type="redacted_thinking")


def test_redacted_thinking_empty_data_raises():
    with pytest.raises(ValidationError, match="non-empty 'data'"):
        AnthropicContentBlock(type="redacted_thinking", data="")


def test_redacted_thinking_valid_passes():
    block = AnthropicContentBlock(type="redacted_thinking", data="opaque")
    assert block.data == "opaque"


# ======================================================================
# tool_reference — no per-type validation required
# ======================================================================


def test_tool_reference_block_passes():
    block = AnthropicContentBlock(
        type="tool_reference",
        tool_name="search",
    )
    assert block.tool_name == "search"
