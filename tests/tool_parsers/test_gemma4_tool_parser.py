# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers.gemma4_tool_parser import (
    TOOL_CALL_END,
    TOOL_CALL_START,
    Gemma4ToolParser,
    _parse_gemma4_args,
    _parse_gemma4_array,
)

# ---------------------------------------------------------------------------
# Real-tokenizer fixture (requires network access to download model weights)
# Used only by tests that need actual Gemma4 token IDs.
# ---------------------------------------------------------------------------
try:
    from vllm.tokenizers.registry import get_tokenizer as _get_tokenizer

    @pytest.fixture(scope="module")
    def gemma4_tokenizer():
        try:
            return _get_tokenizer("google/gemma-4-E2B-it")
        except Exception:
            pytest.skip("Gemma4 tokenizer unavailable (network or version issue)")

except Exception:

    @pytest.fixture(scope="module")
    def gemma4_tokenizer():
        pytest.skip("Gemma4 tokenizer unavailable")

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2, 3]
    # Include the tool call start token in the vocab for the parser
    tokenizer.get_vocab.return_value = {TOOL_CALL_START: 48, TOOL_CALL_END: 49}
    return tokenizer


@pytest.fixture
def parser(mock_tokenizer):
    return Gemma4ToolParser(mock_tokenizer)


@pytest.fixture
def mock_request():
    request = MagicMock(spec=ChatCompletionRequest)
    request.tools = []
    request.tool_choice = "auto"
    return request


# ---------------------------------------------------------------------------
# Unit tests for _parse_gemma4_args (shared parser logic)
# ---------------------------------------------------------------------------


class TestParseGemma4Args:
    def test_empty_string(self):
        assert _parse_gemma4_args("") == {}

    def test_whitespace_only(self):
        assert _parse_gemma4_args("   ") == {}

    def test_single_string_value(self):
        result = _parse_gemma4_args('location:<|"|>Paris<|"|>')
        assert result == {"location": "Paris"}

    def test_string_value_with_comma(self):
        result = _parse_gemma4_args('location:<|"|>Paris, France<|"|>')
        assert result == {"location": "Paris, France"}

    def test_multiple_string_values(self):
        result = _parse_gemma4_args(
            'location:<|"|>San Francisco<|"|>,unit:<|"|>celsius<|"|>'
        )
        assert result == {"location": "San Francisco", "unit": "celsius"}

    def test_integer_value(self):
        result = _parse_gemma4_args("count:42")
        assert result == {"count": 42}

    def test_float_value(self):
        result = _parse_gemma4_args("score:3.14")
        assert result == {"score": 3.14}

    def test_boolean_true(self):
        result = _parse_gemma4_args("flag:true")
        assert result == {"flag": True}

    def test_boolean_false(self):
        result = _parse_gemma4_args("flag:false")
        assert result == {"flag": False}

    def test_null_value(self):
        # Bare `null` must parse as None (Python), not the string "null".
        # Without this, tool_choice=auto would emit `{"param": "null"}`
        # instead of `{"param": null}` for nullable tool parameters.
        result = _parse_gemma4_args("param:null")
        assert result == {"param": None}
        assert json.dumps(result) == '{"param": null}'

    def test_mixed_types(self):
        result = _parse_gemma4_args(
            'name:<|"|>test<|"|>,count:42,active:true,score:3.14'
        )
        assert result == {
            "name": "test",
            "count": 42,
            "active": True,
            "score": 3.14,
        }

    def test_nested_object(self):
        result = _parse_gemma4_args('nested:{inner:<|"|>value<|"|>}')
        assert result == {"nested": {"inner": "value"}}

    def test_array_of_strings(self):
        result = _parse_gemma4_args('items:[<|"|>a<|"|>,<|"|>b<|"|>]')
        assert result == {"items": ["a", "b"]}

    def test_unterminated_string(self):
        """Unterminated strings should take everything after the delimiter."""
        result = _parse_gemma4_args('key:<|"|>unterminated')
        assert result == {"key": "unterminated"}

    def test_empty_value(self):
        """Key with no value after colon."""
        result = _parse_gemma4_args("key:")
        assert result == {"key": ""}

    def test_empty_value_partial_withheld(self):
        """Key with no value is withheld in partial mode to avoid premature emission."""
        result = _parse_gemma4_args("key:", partial=True)
        assert result == {}
        # also with a space after the colon
        result = _parse_gemma4_args("key: ", partial=True)
        assert result == {}

    def test_empty_value_after_other_keys_partial_withheld(self):
        """Trailing key with no value is withheld; earlier keys are kept."""
        result = _parse_gemma4_args('name:<|"|>test<|"|>,flag:', partial=True)
        assert result == {"name": "test"}


class TestParseGemma4Array:
    def test_string_array(self):
        result = _parse_gemma4_array('<|"|>a<|"|>,<|"|>b<|"|>')
        assert result == ["a", "b"]

    def test_empty_array(self):
        result = _parse_gemma4_array("")
        assert result == []

    def test_bare_values(self):
        result = _parse_gemma4_array("42,true,3.14")
        assert result == [42, True, 3.14]


# ---------------------------------------------------------------------------
# Non-streaming extraction tests
# ---------------------------------------------------------------------------


class TestExtractToolCalls:
    def test_no_tool_calls(self, parser, mock_request):
        model_output = "Hello, how can I help you today?"
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is False
        assert result.tool_calls == []
        assert result.content == model_output

    def test_single_tool_call(self, parser, mock_request):
        model_output = (
            '<|tool_call>call:get_weather{location:<|"|>London<|"|>}<tool_call|>'
        )
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"location": "London"}

    def test_multiple_arguments(self, parser, mock_request):
        model_output = (
            "<|tool_call>call:get_weather{"
            'location:<|"|>San Francisco<|"|>,'
            'unit:<|"|>celsius<|"|>}'
            "<tool_call|>"
        )
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"location": "San Francisco", "unit": "celsius"}

    def test_text_before_tool_call(self, parser, mock_request):
        model_output = (
            "Let me check the weather for you. "
            '<|tool_call>call:get_weather{location:<|"|>Paris<|"|>}'
            "<tool_call|>"
        )
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert result.content == "Let me check the weather for you."
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"

    def test_multiple_tool_calls(self, parser, mock_request):
        model_output = (
            '<|tool_call>call:get_weather{location:<|"|>London<|"|>}'
            "<tool_call|>"
            '<|tool_call>call:get_time{location:<|"|>London<|"|>}'
            "<tool_call|>"
        )
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "get_weather"
        assert result.tool_calls[1].function.name == "get_time"

    def test_nested_arguments(self, parser, mock_request):
        model_output = (
            "<|tool_call>call:complex_function{"
            'nested:{inner:<|"|>value<|"|>},'
            'list:[<|"|>a<|"|>,<|"|>b<|"|>]}'
            "<tool_call|>"
        )
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "complex_function"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"nested": {"inner": "value"}, "list": ["a", "b"]}

    def test_tool_call_with_number_and_boolean(self, parser, mock_request):
        model_output = (
            "<|tool_call>call:set_status{"
            "is_active:true,"
            "count:42,"
            "score:3.14}"
            "<tool_call|>"
        )
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "set_status"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {"is_active": True, "count": 42, "score": 3.14}

    def test_incomplete_tool_call(self, parser, mock_request):
        model_output = '<|tool_call>call:get_weather{location:<|"|>London'
        result = parser.extract_tool_calls(model_output, mock_request)

        # Incomplete — no <tool_call|> end marker, regex won't match
        assert result.tools_called is False
        assert result.content == model_output

    def test_hyphenated_function_name(self, parser, mock_request):
        """Ensure function names with hyphens are parsed correctly."""
        model_output = (
            '<|tool_call>call:get-weather{location:<|"|>London<|"|>}<tool_call|>'
        )
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert result.tool_calls[0].function.name == "get-weather"

    def test_dotted_function_name(self, parser, mock_request):
        """Ensure function names with dots are parsed correctly."""
        model_output = (
            '<|tool_call>call:weather.get{location:<|"|>London<|"|>}<tool_call|>'
        )
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert result.tool_calls[0].function.name == "weather.get"

    def test_no_arguments(self, parser, mock_request):
        """Tool calls with empty arguments."""
        model_output = "<|tool_call>call:get_status{}<tool_call|>"
        result = parser.extract_tool_calls(model_output, mock_request)

        assert result.tools_called is True
        assert result.tool_calls[0].function.name == "get_status"
        args = json.loads(result.tool_calls[0].function.arguments)
        assert args == {}


# ---------------------------------------------------------------------------
# Streaming extraction tests
# ---------------------------------------------------------------------------


class TestStreamingExtraction:
    """Tests for the streaming tool call extraction.

    These simulate the token-by-token streaming that vLLM performs,
    feeding incremental text to extract_tool_calls_streaming() and
    verifying that the accumulated argument deltas form valid JSON.
    """

    def _simulate_streaming(
        self, parser: Gemma4ToolParser, mock_request: Any, chunks: list[str]
    ) -> list[tuple[Any, str]]:
        """Feed chunks through the streaming parser and collect results.

        Returns a list of (delta_message, accumulated_text) tuples.
        """
        results: list[tuple[Any, str]] = []
        previous_text: str = ""
        previous_token_ids: list[int] = []

        for chunk in chunks:
            current_text = previous_text + chunk
            # Use token ID 48 for tool_call start, 49 for end, 0 otherwise
            delta_token_ids: list[int] = []
            if TOOL_CALL_START in chunk:
                delta_token_ids.append(48)
            elif TOOL_CALL_END in chunk:
                delta_token_ids.append(49)
            else:
                delta_token_ids.append(0)

            current_token_ids = previous_token_ids + delta_token_ids

            delta = parser.extract_tool_calls_streaming(
                previous_text=previous_text,
                current_text=current_text,
                delta_text=chunk,
                previous_token_ids=tuple(previous_token_ids),
                current_token_ids=tuple(current_token_ids),
                delta_token_ids=tuple(delta_token_ids),
                request=mock_request,
            )
            results.append((delta, current_text))
            previous_text = current_text
            previous_token_ids = list(current_token_ids)

        return results

    def _collect_arguments(self, results):
        """Collect all argument deltas from streaming results into one string."""
        args_text = ""
        for delta, _ in results:
            if delta and delta.tool_calls:
                for tc in delta.tool_calls:
                    func = tc.function if isinstance(tc.function, dict) else tc.function
                    if isinstance(func, dict):
                        arg = func.get("arguments", "")
                    else:
                        arg = getattr(func, "arguments", "") or ""
                    if arg:
                        args_text += arg
        return args_text

    def _collect_function_name(self, results):
        """Extract the function name from streaming results."""
        for delta, _ in results:
            if delta and delta.tool_calls:
                for tc in delta.tool_calls:
                    func = tc.function if isinstance(tc.function, dict) else tc.function
                    if isinstance(func, dict):
                        name = func.get("name")
                    else:
                        name = getattr(func, "name", None)
                    if name:
                        return name
        return None

    def test_basic_streaming_single_tool(self, parser, mock_request):
        """Simulate the exact streaming scenario from the bug report.

        Model generates:
        <|tool_call>call:get_weather{location:<|"|>Paris, France<|"|>}<tool_call|>

        Expected: arguments should be valid JSON {"location": "Paris, France"}
        """
        chunks = [
            "<|tool_call>",
            "call:get_weather{",
            'location:<|"|>Paris',
            ", France",
            '<|"|>}',
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)

        # Verify function name
        name = self._collect_function_name(results)
        assert name == "get_weather", f"Expected 'get_weather', got '{name}'"

        # Verify arguments form valid JSON
        args_text = self._collect_arguments(results)
        assert args_text, "No arguments were streamed"
        parsed_args = json.loads(args_text)
        assert parsed_args == {"location": "Paris, France"}

    def test_streaming_multi_arg(self, parser, mock_request):
        """Streaming with multiple arguments."""
        chunks = [
            "<|tool_call>",
            "call:get_weather{",
            'location:<|"|>Tokyo<|"|>,',
            'unit:<|"|>celsius<|"|>}',
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)

        name = self._collect_function_name(results)
        assert name == "get_weather"

        args_text = self._collect_arguments(results)
        assert args_text
        parsed_args = json.loads(args_text)
        assert parsed_args == {"location": "Tokyo", "unit": "celsius"}

    def test_streaming_no_extra_brace(self, parser, mock_request):
        """Verify the closing } is NOT leaked into arguments (Bug #2)."""
        chunks = [
            "<|tool_call>",
            "call:get_weather{",
            'location:<|"|>London<|"|>}',
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)
        assert args_text

        # The args text must be valid JSON (no extra })
        parsed = json.loads(args_text)
        assert parsed == {"location": "London"}

        # Specifically assert no double-brace
        assert args_text.count("}") <= 1, (
            f"Arguments contain extra closing brace: {args_text!r}"
        )

    def test_streaming_no_unquoted_keys(self, parser, mock_request):
        """Verify keys are properly quoted in JSON (Bug #1)."""
        chunks = [
            "<|tool_call>",
            "call:get_weather{",
            'location:<|"|>Paris<|"|>}',
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)

        # Must start with { and contain quoted key
        assert args_text.lstrip().startswith("{"), (
            f"Arguments don't start with '{{': {args_text!r}"
        )
        assert '"location"' in args_text, (
            f"Key 'location' not properly quoted: {args_text!r}"
        )

    def test_streaming_name_no_call_prefix(self, parser, mock_request):
        """Verify function name has no 'call:' prefix."""
        chunks = [
            "<|tool_call>",
            "call:get_weather{",
            'location:<|"|>Paris<|"|>}',
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        name = self._collect_function_name(results)
        assert name == "get_weather"
        assert not name.startswith("call:"), f"Name has 'call:' prefix: {name!r}"

    def test_streaming_text_before_tool_call(self, parser, mock_request):
        """Text before tool call should be emitted as content."""
        chunks = [
            "Let me check ",
            "the weather. ",
            "<|tool_call>",
            "call:get_weather{",
            'location:<|"|>London<|"|>}',
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)

        # First chunks should be content
        content_parts = []
        for delta, _ in results:
            if delta and delta.content:
                content_parts.append(delta.content)

        assert "".join(content_parts).strip().startswith("Let me check")

    def test_streaming_numeric_args(self, parser, mock_request):
        """Streaming with numeric and boolean argument values."""
        chunks = [
            "<|tool_call>",
            "call:set_config{",
            "count:42,",
            "active:true}",
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)
        if args_text:
            parsed_args = json.loads(args_text)
            assert parsed_args["count"] == 42
            assert parsed_args["active"] is True

    def test_streaming_boolean_split_across_chunks(self, parser, mock_request):
        """Boolean value split across token boundaries must not corrupt JSON."""
        chunks = [
            "<|tool_call>",
            "call:search{input:{all:" + "true"[:3],
            "e}}",
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)
        assert args_text, "No arguments were streamed"
        parsed_args = json.loads(args_text)
        assert parsed_args["input"]["all"] is True

    def test_streaming_false_split_across_chunks(self, parser, mock_request):
        """Boolean false split across chunks."""
        chunks = [
            "<|tool_call>",
            "call:set{flag:" + "false"[:4],
            "e}",
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)
        assert args_text, "No arguments were streamed"
        parsed_args = json.loads(args_text)
        assert parsed_args["flag"] is False

    def test_streaming_number_split_across_chunks(self, parser, mock_request):
        """Number split across chunks must not change type."""
        chunks = [
            "<|tool_call>",
            "call:set{count:4",
            "2}",
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)
        assert args_text, "No arguments were streamed"
        parsed_args = json.loads(args_text)
        assert parsed_args["count"] == 42

    def test_streaming_empty_args(self, parser, mock_request):
        """Tool call with no arguments."""
        chunks = [
            "<|tool_call>",
            "call:get_status{}",
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        name = self._collect_function_name(results)
        assert name == "get_status"

    def test_streaming_split_delimiter_no_invalid_json(self, parser, mock_request):
        """Partial <|"|> delimiter chars must not leak into streamed JSON.

        Reproduces the bug from https://github.com/vllm-project/vllm/issues/38946
        where a token boundary splits the string delimiter, leaving fragments
        like '<|' at the end of a parsed value which then corrupt the JSON.
        """
        chunks = [
            "<|tool_call>",
            "call:todowrite{",
            'content:<|"|>Buy milk<|',
            '"|>}',
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)

        args_text = self._collect_arguments(results)
        assert args_text, "No arguments were streamed"

        # Must be valid JSON — the original bug caused a JSON parse error
        parsed_args = json.loads(args_text)
        assert parsed_args["content"] == "Buy milk"

        # Ensure no raw delimiter fragments leaked into the JSON
        assert "<|" not in args_text, (
            f"Partial delimiter leaked into JSON: {args_text!r}"
        )

    def test_streaming_does_not_duplicate_plain_text_after_tool_call(
        self, parser, mock_request, monkeypatch
    ):
        """Buffered plain text after a tool call must not corrupt current_text."""
        captured_current_texts: list[str] = []
        original_extract_streaming = parser._extract_streaming

        def wrapped_extract_streaming(previous_text, current_text, delta_text):
            captured_current_texts.append(current_text)
            return original_extract_streaming(previous_text, current_text, delta_text)

        monkeypatch.setattr(parser, "_extract_streaming", wrapped_extract_streaming)

        chunks = [
            "<|tool_call>",
            "call:get_weather{",
            'location:<|"|>Paris<|"|>}',
            "<tool_call|><",
            "div>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        content_parts = [
            delta.content for delta, _ in results if delta is not None and delta.content
        ]
        assert "".join(content_parts) == "<div>"
        assert captured_current_texts[-1].endswith("<tool_call|><div>")
        assert not captured_current_texts[-1].endswith("<tool_call|><<div>")

    def test_streaming_html_argument_does_not_duplicate_tag_prefixes(
        self, parser, mock_request
    ):
        """HTML content inside tool arguments must not be duplicated."""
        chunks = [
            "<|tool_call>",
            "call:write_file{",
            'path:<|"|>index.html<|"|>,',
            'content:<|"|><!DOCTYPE html>\n<',
            'html lang="zh-CN">\n<',
            "head>\n    <",
            'meta charset="UTF-8">\n    <',
            'meta name="viewport" content="width=device-width">\n',
            '<|"|>}',
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)
        assert args_text

        parsed_args = json.loads(args_text)
        assert parsed_args["path"] == "index.html"
        assert (
            parsed_args["content"] == "<!DOCTYPE html>\n"
            '<html lang="zh-CN">\n'
            "<head>\n"
            '    <meta charset="UTF-8">\n'
            '    <meta name="viewport" content="width=device-width">\n'
        )

    def test_streaming_trailing_bare_bool_not_duplicated(self, parser, mock_request):
        """Trailing bare boolean must not be streamed twice."""
        chunks = [
            "<|tool_call>",
            "call:Edit{",
            'file_path:<|"|>src/env.py<|"|>,',
            'old_string:<|"|>old_val<|"|>,',
            'new_string:<|"|>new_val<|"|>,',
            "replace_all:",
            "false}",
            "<tool_call|>",
        ]

        results = self._simulate_streaming(parser, mock_request, chunks)
        args_text = self._collect_arguments(results)
        assert args_text, "No arguments were streamed"

        parsed_args = json.loads(args_text)
        assert parsed_args == {
            "file_path": "src/env.py",
            "old_string": "old_val",
            "new_string": "new_val",
            "replace_all": False,
        }

        assert args_text.count("replace_all") == 1


# ---------------------------------------------------------------------------
# Regression tests for issue #39885
# ---------------------------------------------------------------------------


class TestStreamingReasoningAutoToolIssue39885:
    """Regression tests for #39885.

    Verifies that reasoning tokens don't leak into delta.content when
    tool_choice='auto' is used in a multi-turn conversation that has a
    prior tool result. The auto-tool streaming branch (elif parser is
    not None in serving.py) must run reasoning extraction before passing
    text to the tool parser, same as the required and named tool branches.
    """

    def _run_fixed_streaming_path(
        self,
        reasoning_parser,
        tool_parser,
        prompt_token_ids: list[int],
        model_chunks: list[tuple[list[int], str]],
        mock_request,
    ) -> tuple[str, str]:
        """Simulate the auto-tool streaming branch from serving.py.

        Mirrors the logic in the elif parser is not None block:
        run reasoning extraction first, then hand off to the tool parser
        once reasoning has ended.

        Returns (accumulated_content, accumulated_reasoning).
        """
        reasoning_ended = reasoning_parser.is_reasoning_end(prompt_token_ids)

        content_acc = ""
        reasoning_acc = ""
        previous_text = ""
        previous_token_ids: list[int] = []

        for delta_token_ids, delta_text in model_chunks:
            current_text = previous_text + delta_text
            current_token_ids = previous_token_ids + delta_token_ids

            if not reasoning_ended:
                delta_message = reasoning_parser.extract_reasoning_streaming(
                    previous_text,
                    current_text,
                    delta_text,
                    previous_token_ids,
                    current_token_ids,
                    delta_token_ids,
                )
                if reasoning_parser.is_reasoning_end(delta_token_ids):
                    reasoning_ended = True
                    if delta_message and delta_message.content:
                        current_text = delta_message.content
                        delta_message.content = None
                    else:
                        current_text = ""
            else:
                delta_message = tool_parser.extract_tool_calls_streaming(
                    previous_text=previous_text,
                    current_text=current_text,
                    delta_text=delta_text,
                    previous_token_ids=previous_token_ids,
                    current_token_ids=current_token_ids,
                    delta_token_ids=delta_token_ids,
                    request=mock_request,
                )

            if delta_message:
                if delta_message.content:
                    content_acc += delta_message.content
                if delta_message.reasoning:
                    reasoning_acc += delta_message.reasoning

            previous_text = current_text
            previous_token_ids = current_token_ids

        return content_acc, reasoning_acc

    def test_reasoning_not_leaked_into_content_multiturn(self, gemma4_tokenizer):
        """Reasoning must not appear in delta.content in multi-turn tool streaming.

        When the prompt has a prior tool call token, is_reasoning_end could
        return True before the generation even starts, skipping reasoning
        extraction and leaking thought tokens into content. Regression for
        #39885.
        """
        from vllm.reasoning import ReasoningParserManager

        reasoning_parser_cls = ReasoningParserManager.get_reasoning_parser("gemma4")
        reasoning_parser = reasoning_parser_cls(gemma4_tokenizer)
        tool_parser = Gemma4ToolParser(gemma4_tokenizer)

        vocab = gemma4_tokenizer.get_vocab()
        channel_start_id = vocab["<|channel>"]
        channel_end_id = vocab["<channel|>"]
        tool_call_start_id = vocab["<|tool_call>"]
        tool_response_id = vocab["<|tool_response>"]
        new_turn_id = vocab["<|turn>"]

        def enc(text: str) -> list[int]:
            tok = getattr(gemma4_tokenizer, "tokenizer", gemma4_tokenizer)
            try:
                return tok.encode(text, add_special_tokens=False)
            except TypeError:
                return tok.encode(text)

        # prompt ends with a prior tool call + tool response + new turn,
        # which is the pattern that exposed the bug in is_reasoning_end
        prompt_token_ids = (
            enc("user\nSearch for something\n")
            + [new_turn_id]
            + enc("model\n")
            + [tool_call_start_id]  # prior tool call in prompt
            + enc("call:ToolA{x:<|")
            + [tool_response_id]
            + enc("response:ToolA{content:Success}")
            + [new_turn_id]  # generation prompt boundary
            + enc("model\n")
        )

        # model output: reasoning block followed by a tool call
        tool_call_content = 'call:ToolB{query:<|"|>find data protection laws<|"|>}'
        model_chunks: list[tuple[list[int], str]] = [
            # <|channel> arrives as a special token; with
            # skip_special_tokens=False the text render is "" here.
            ([channel_start_id], ""),
            (enc("thought\n"), "thought\n"),
            (enc("Actual reasoning"), "Actual reasoning"),
            ([channel_end_id], ""),  # <channel|>
            ([tool_call_start_id], TOOL_CALL_START),
            (enc(tool_call_content), tool_call_content),
            (enc(TOOL_CALL_END), TOOL_CALL_END),
        ]

        mock_req = MagicMock(spec=ChatCompletionRequest)
        mock_req.tools = []
        mock_req.tool_choice = "auto"

        content_acc, reasoning_acc = self._run_fixed_streaming_path(
            reasoning_parser,
            tool_parser,
            prompt_token_ids,
            model_chunks,
            mock_req,
        )

        assert "thought" not in content_acc, (
            f"reasoning leaked into content: {content_acc!r}"
        )
        assert "Actual reasoning" not in content_acc, (
            f"reasoning leaked into content: {content_acc!r}"
        )
        assert "Actual reasoning" in reasoning_acc, (
            f"expected reasoning in reasoning field, got: {reasoning_acc!r}"
        )

    def test_no_reasoning_parser_still_works(self):
        """Tool streaming without a reasoning parser still works correctly.

        When there is no reasoning parser, the auto-tool path falls straight
        through to extract_tool_calls_streaming with no reasoning checks.
        """
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.get_vocab.return_value = {
            TOOL_CALL_START: 48,
            TOOL_CALL_END: 49,
        }

        tool_parser = Gemma4ToolParser(mock_tokenizer)
        mock_req = MagicMock(spec=ChatCompletionRequest)
        mock_req.tools = []
        mock_req.tool_choice = "auto"

        # no reasoning prefix, tool call arrives immediately
        chunks = [
            TOOL_CALL_START,
            'call:MyTool{param:<|"|>hello<|"|>}',
            TOOL_CALL_END,
        ]
        previous_text = ""
        previous_token_ids: list[int] = []
        func_name: str | None = None
        args_acc = ""
        for chunk in chunks:
            current_text = previous_text + chunk
            delta_token_ids = (
                [48]
                if TOOL_CALL_START in chunk
                else [49]
                if TOOL_CALL_END in chunk
                else [0]
            )
            current_token_ids = previous_token_ids + delta_token_ids
            delta = tool_parser.extract_tool_calls_streaming(
                previous_text=previous_text,
                current_text=current_text,
                delta_text=chunk,
                previous_token_ids=previous_token_ids,
                current_token_ids=current_token_ids,
                delta_token_ids=delta_token_ids,
                request=mock_req,
            )
            if delta and delta.tool_calls:
                for tc in delta.tool_calls:
                    fn = tc.function
                    name = fn.name if hasattr(fn, "name") else (fn or {}).get("name")
                    if name:
                        func_name = name
                    arg = (
                        fn.arguments
                        if hasattr(fn, "arguments")
                        else (fn or {}).get("arguments", "")
                    )
                    if arg:
                        args_acc += arg
            previous_text = current_text
            previous_token_ids = list(current_token_ids)

        assert func_name == "MyTool"
        parsed = json.loads(args_acc)
        assert parsed == {"param": "hello"}
