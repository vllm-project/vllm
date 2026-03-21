# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ToolParser.adjust_request() with ResponsesRequest.

Verifies that thinking models use the logits processor path while
non-thinking models use guided generation (ResponseFormatTextJSONSchemaConfig).
"""

from unittest.mock import MagicMock

import pytest
from openai.types.responses import FunctionTool

from vllm.tool_parsers.abstract_tool_parser import ToolParser

pytestmark = pytest.mark.cpu_test


def _make_tokenizer(vocab: dict[str, int]):
    """Create a mock tokenizer with given vocab."""
    tok = MagicMock()
    tok.get_vocab.return_value = vocab
    return tok


def _make_responses_request(*, tools=None, tool_choice="required", reasoning=None,
                            vllm_xargs=None):
    """Create a mock ResponsesRequest."""
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

    request = MagicMock(spec=ResponsesRequest)
    request.tools = tools or [
        FunctionTool(
            type="function",
            name="get_weather",
            description="Get weather",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}},
            strict=False,
        )
    ]
    request.tool_choice = tool_choice
    request.reasoning = reasoning
    request.vllm_xargs = vllm_xargs
    request.text = None
    return request


class TestAdjustRequestResponses:

    def test_thinking_model_uses_logits_processor(self):
        """Model with </think> + <|im_end|> + reasoning enabled → sets vllm_xargs."""
        vocab = {
            "</think>": 100,
            "<|im_end|>": 101,
            "<|tool_call_end|>": 103,
        }
        parser = ToolParser(_make_tokenizer(vocab))
        request = _make_responses_request(
            reasoning=MagicMock(),  # reasoning is not None
            vllm_xargs=None,
        )

        result = parser.adjust_request(request)

        assert result.vllm_xargs is not None
        assert result.vllm_xargs["tool_choice_required_think_end"] == 100
        assert result.vllm_xargs["tool_choice_required_stop"] == 101
        assert result.vllm_xargs["tool_choice_required_section_end"] == 103
        # Should NOT have set guided generation
        assert result.text is None

    def test_non_thinking_model_uses_guided_generation(self):
        """Model without </think> → uses ResponseFormatTextJSONSchemaConfig."""
        vocab = {
            "<|im_end|>": 101,
            # No </think> → not a thinking model
        }
        parser = ToolParser(_make_tokenizer(vocab))
        request = _make_responses_request(
            reasoning=MagicMock(),
        )

        result = parser.adjust_request(request)

        # Should have set guided generation
        assert result.text is not None
        assert result.text.format is not None
        assert result.text.format.type == "json_schema"
        # Should NOT have set vllm_xargs
        assert result.vllm_xargs is None

    def test_reasoning_off_uses_guided_generation(self):
        """Even thinking model, reasoning=None → guided generation."""
        vocab = {
            "</think>": 100,
            "<|im_end|>": 101,
            "<|tool_call_end|>": 103,
        }
        parser = ToolParser(_make_tokenizer(vocab))
        request = _make_responses_request(
            reasoning=None,  # reasoning is off
        )

        result = parser.adjust_request(request)

        assert result.text is not None
        assert result.text.format is not None

    def test_tool_call_end_token_optional(self):
        """Model with </think> but no <|tool_call_end|> → section_end not set."""
        vocab = {
            "</think>": 100,
            "<|im_end|>": 101,
            # No <|tool_call_end|>
        }
        parser = ToolParser(_make_tokenizer(vocab))
        request = _make_responses_request(
            reasoning=MagicMock(),
            vllm_xargs=None,
        )

        result = parser.adjust_request(request)

        assert result.vllm_xargs is not None
        assert result.vllm_xargs["tool_choice_required_think_end"] == 100
        assert result.vllm_xargs["tool_choice_required_stop"] == 101
        assert "tool_choice_required_section_end" not in result.vllm_xargs

    def test_tool_choice_none_skips_adjustment(self):
        """tool_choice=none → no schema, no xargs."""
        vocab = {"</think>": 100, "<|im_end|>": 101}
        parser = ToolParser(_make_tokenizer(vocab))
        request = _make_responses_request(
            tool_choice="none",
            reasoning=MagicMock(),
        )

        result = parser.adjust_request(request)

        assert result.text is None
        assert result.vllm_xargs is None

    def test_tool_choice_auto_skips_adjustment(self):
        """tool_choice=auto → no schema, no xargs."""
        vocab = {"</think>": 100, "<|im_end|>": 101}
        parser = ToolParser(_make_tokenizer(vocab))
        request = _make_responses_request(
            tool_choice="auto",
            reasoning=MagicMock(),
        )

        result = parser.adjust_request(request)

        assert result.text is None
        assert result.vllm_xargs is None

    def test_preserves_existing_vllm_xargs(self):
        """Existing vllm_xargs should be preserved, not overwritten."""
        vocab = {
            "</think>": 100,
            "<|im_end|>": 101,
        }
        parser = ToolParser(_make_tokenizer(vocab))
        request = _make_responses_request(
            reasoning=MagicMock(),
            vllm_xargs={"existing_key": "existing_value"},
        )

        result = parser.adjust_request(request)

        assert result.vllm_xargs["existing_key"] == "existing_value"
        assert result.vllm_xargs["tool_choice_required_think_end"] == 100

    def test_no_tools_skips_adjustment(self):
        """No tools → early return, no changes."""
        vocab = {"</think>": 100, "<|im_end|>": 101}
        parser = ToolParser(_make_tokenizer(vocab))
        request = _make_responses_request(reasoning=MagicMock())
        request.tools = None

        result = parser.adjust_request(request)

        assert result.text is None
        assert result.vllm_xargs is None
