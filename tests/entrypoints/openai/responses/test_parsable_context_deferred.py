# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ParsableContext deferred parsing.

The ParsableContext accumulates output text across multiple append_output
calls and runs a single full parse when _ensure_final_parse() is called.
This avoids feeding partial deltas to ResponsesParser.process() which
expects complete output text.
"""

from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.responses.context import ParsableContext

pytestmark = pytest.mark.cpu_test


def _make_request():
    request = MagicMock()
    request.tools = []
    request.tool_choice = "auto"
    request.enable_response_messages = False
    return request


def _make_output(text: str, token_ids: tuple = (), finish_reason=None):
    """Create a mock RequestOutput."""
    inner = MagicMock()
    inner.text = text
    inner.token_ids = token_ids
    inner.finish_reason = finish_reason
    inner.logprobs = None

    output = MagicMock()
    output.outputs = [inner]
    output.prompt_token_ids = [1, 2, 3]
    output.num_cached_tokens = 0
    output.prompt = "test"
    output.kv_transfer_params = None
    return output


def _make_context():
    """Create a ParsableContext and replace its parser with a mock."""
    request = _make_request()
    reasoning_parser = MagicMock()
    reasoning_parser_cls = MagicMock(return_value=reasoning_parser)

    ctx = ParsableContext(
        response_messages=[],
        tokenizer=MagicMock(),
        reasoning_parser_cls=reasoning_parser_cls,
        request=request,
        available_tools=None,
        tool_parser_cls=None,
        chat_template=None,
        chat_template_content_format="auto",
    )
    # Replace the real ResponsesParser with a mock so we can track calls
    ctx.parser = MagicMock()
    ctx.parser.response_messages = []
    return ctx


class TestDeferredParsing:
    def test_append_output_accumulates_text(self):
        """Multiple append_output calls accumulate text without parsing."""
        ctx = _make_context()

        ctx.append_output(_make_output("Hello "))
        ctx.append_output(_make_output("world!"))

        assert ctx._accumulated_text == "Hello world!"
        assert not ctx._final_parsed
        # Parser.process should NOT have been called yet
        ctx.parser.process.assert_not_called()

    def test_ensure_final_parse_runs_once(self):
        """_ensure_final_parse triggers parse exactly once."""
        ctx = _make_context()

        ctx.append_output(
            _make_output("<think>reasoning</think>tool call", finish_reason="stop")
        )

        ctx._ensure_final_parse()
        assert ctx._final_parsed
        ctx.parser.process.assert_called_once()

        # Second call is a no-op
        ctx._ensure_final_parse()
        ctx.parser.process.assert_called_once()

    def test_ensure_final_parse_creates_synthetic_output(self):
        """The synthetic CompletionOutput has accumulated text and token_ids."""
        ctx = _make_context()

        ctx.append_output(_make_output("part1", (10, 11)))
        ctx.append_output(_make_output("part2", (12, 13), finish_reason="stop"))

        ctx._ensure_final_parse()

        call_args = ctx.parser.process.call_args[0][0]
        assert call_args.text == "part1part2"
        assert call_args.token_ids == (10, 11, 12, 13)
        assert call_args.finish_reason == "stop"

    def test_append_tool_output_resets_accumulation(self):
        """append_tool_output triggers parse, then resets for next turn."""
        ctx = _make_context()

        ctx.append_output(_make_output("first turn", finish_reason="stop"))
        ctx.append_tool_output([MagicMock()])

        # Should have parsed
        assert ctx.parser.process.call_count == 1
        # Accumulation reset
        assert ctx._accumulated_text == ""
        assert ctx._accumulated_token_ids == []
        assert not ctx._final_parsed

        # New turn accumulation works
        ctx.append_output(_make_output("second turn", finish_reason="stop"))
        assert ctx._accumulated_text == "second turn"

    def test_need_builtin_tool_call_triggers_parse(self):
        """need_builtin_tool_call calls _ensure_final_parse before checking."""
        ctx = _make_context()
        # Set up parser to return a non-tool message
        mock_msg = MagicMock()
        mock_msg.type = "message"
        ctx.parser.response_messages = [mock_msg]

        ctx.append_output(_make_output("some text", finish_reason="stop"))
        result = ctx.need_builtin_tool_call()

        assert ctx._final_parsed
        assert result is False

    def test_no_parse_when_no_text(self):
        """If no text accumulated, _ensure_final_parse does nothing."""
        ctx = _make_context()
        ctx._ensure_final_parse()

        assert ctx._final_parsed
        ctx.parser.process.assert_not_called()

    def test_last_output_stored(self):
        """append_output stores last_output for finish_reason access."""
        ctx = _make_context()
        out = _make_output("text", finish_reason="length")
        ctx.append_output(out)

        assert ctx.last_output is out

    def test_multi_turn_accumulation(self):
        """Two generation turns each accumulate and parse independently."""
        ctx = _make_context()

        # Turn 1
        ctx.append_output(_make_output("turn1_a", (1,)))
        ctx.append_output(_make_output("turn1_b", (2,), finish_reason="stop"))
        ctx.append_tool_output([MagicMock()])

        assert ctx.parser.process.call_count == 1
        first_call = ctx.parser.process.call_args_list[0][0][0]
        assert first_call.text == "turn1_aturn1_b"

        # Turn 2
        ctx.append_output(_make_output("turn2", (3,), finish_reason="stop"))
        ctx._ensure_final_parse()

        assert ctx.parser.process.call_count == 2
        second_call = ctx.parser.process.call_args_list[1][0][0]
        assert second_call.text == "turn2"

    def test_need_builtin_tool_call_empty_messages(self):
        """need_builtin_tool_call returns False when no messages parsed."""
        ctx = _make_context()
        ctx.append_output(_make_output("some text", finish_reason="stop"))
        # parser.response_messages is empty (mock default)
        ctx.parser.response_messages = []

        result = ctx.need_builtin_tool_call()

        assert ctx._final_parsed
        assert result is False
