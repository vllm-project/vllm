# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the engine-based Inkling parser.

Inkling output is a sequence of typed content blocks delimited by dedicated
special tokens; the tool-call payload is ``{"name":...,"args":{...}}``
between ``<|content_invoke_tool_json|>`` and ``<|end_message|>``. The
cases mirror the Rust unified parser's tests
(``rust/src/parser/src/unified/inkling.rs``) where applicable.
"""

import json

import pytest

from tests.parser.engine.conftest import make_mock_tokenizer
from tests.parser.engine.streaming_helpers import (
    collect_content,
    collect_function_name,
    collect_tool_arguments,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine_config import ParserState
from vllm.parser.engine.streaming_parser_engine import StreamingParserEngine
from vllm.parser.inkling import InklingParser, _inkling_arg_converter, inkling_config
from vllm.parser.parser_manager import ParserManager

MSG_MODEL = "<|message_model|>"
TEXT_START = "<|content_text|>"
THINK_START = "<|content_thinking|>"
TOOL_JSON = "<|content_invoke_tool_json|>"
TOOL_TEXT = "<|content_invoke_tool_text|>"
TOOL_ERROR = "<|content_tool_error|>"
END_MESSAGE = "<|end_message|>"
END_SAMPLING = "<|content_model_end_sampling|>"

_TML_VOCAB = {
    MSG_MODEL: 200001,
    TEXT_START: 200004,
    END_SAMPLING: 200006,
    THINK_START: 200008,
    END_MESSAGE: 200010,
    TOOL_ERROR: 200022,
    TOOL_JSON: 200049,
    TOOL_TEXT: 200057,
}


@pytest.fixture
def mock_tokenizer():
    return make_mock_tokenizer(_TML_VOCAB)


@pytest.fixture
def parser(mock_tokenizer):
    return InklingParser(mock_tokenizer)


def _tool_block(name: str, args: str) -> str:
    return f'{TOOL_JSON}{{"name":"{name}","args":{args}}}{END_MESSAGE}'


_MARKERS = sorted(_TML_VOCAB, key=len, reverse=True)


def _tokenize(text: str) -> list[tuple[int, str]]:
    """Tokenize like the real stream: markers are atomic special tokens,
    plain text becomes one token per character (matching the mock
    tokenizer's ``chr``-based decode)."""
    tokens: list[tuple[int, str]] = []
    i = 0
    while i < len(text):
        for marker in _MARKERS:
            if text.startswith(marker, i):
                tokens.append((_TML_VOCAB[marker], marker))
                i += len(marker)
                break
        else:
            tokens.append((ord(text[i]), text[i]))
            i += 1
    return tokens


def _stream(parser, request, text: str, chunk_size: int):
    """Stream production-shaped deltas: ``chunk_size`` tokens per delta,
    with delta_token_ids covering every token (specials and text)."""
    tokens = _tokenize(text)
    results = []
    previous_text = ""
    previous_token_ids: list[int] = []
    for start in range(0, len(tokens), chunk_size):
        batch = tokens[start : start + chunk_size]
        delta_text = "".join(t for _, t in batch)
        delta_token_ids = [tid for tid, _ in batch]
        current_text = previous_text + delta_text
        current_token_ids = previous_token_ids + delta_token_ids
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=tuple(previous_token_ids),
            current_token_ids=tuple(current_token_ids),
            delta_token_ids=tuple(delta_token_ids),
            request=request,
        )
        results.append((delta, current_text))
        previous_text = current_text
        previous_token_ids = current_token_ids
    finish = parser.finish_streaming()
    if finish is not None:
        results.append((finish, text))
    return results


def _stream_text_only(parser, request, text: str, chunk_size: int):
    """Stream text-only deltas (no token ids), chunked at arbitrary
    character boundaries — exercises the text-lexing fallback path,
    including markers split across chunks."""
    results = []
    previous_text = ""
    for start in range(0, len(text), chunk_size):
        delta_text = text[start : start + chunk_size]
        current_text = previous_text + delta_text
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=(),
            current_token_ids=(),
            delta_token_ids=(),
            request=request,
        )
        results.append((delta, current_text))
        previous_text = current_text
    finish = parser.finish_streaming()
    if finish is not None:
        results.append((finish, text))
    return results


def _collect_reasoning(results) -> str:
    return "".join(d.reasoning for d, _ in results if d and d.reasoning)


def _function_tool(name: str = "get_weather") -> ChatCompletionToolsParam:
    """A real function-tool definition, as a request would carry it."""
    return ChatCompletionToolsParam(
        function=FunctionDefinition(
            name=name,
            parameters={"type": "object", "properties": {}},
        ),
    )


def _delegating(mock_tokenizer, tools=None):
    """Build the served reasoning+tool DelegatingParser for Inkling."""
    parser_cls = ParserManager.get_parser(
        tool_parser_name="inkling",
        reasoning_parser_name="inkling",
        enable_auto_tools=True,
    )
    return parser_cls(mock_tokenizer, tools or [])


def _stream_delegating(parser, request, text, chunk_size, prompt_token_ids):
    """Stream ``text`` through ``DelegatingParser.parse_delta``, ``chunk_size``
    tokens per delta; return ``(content, reasoning, ordered tool names,
    ordered tool arguments)``. Arguments arrive in fragments, so they are
    concatenated per tool index."""
    tokens = _tokenize(text)
    content, reasoning = "", ""
    tools: dict[int, str] = {}
    args: dict[int, str] = {}
    for start in range(0, len(tokens), chunk_size):
        batch = tokens[start : start + chunk_size]
        delta = parser.parse_delta(
            "".join(t for _, t in batch),
            [tid for tid, _ in batch],
            request,
            prompt_token_ids=prompt_token_ids if start == 0 else None,
            finished=(start + chunk_size >= len(tokens)),
        )
        if delta and delta.content:
            content += delta.content
        if delta and delta.reasoning:
            reasoning += delta.reasoning
        if delta and delta.tool_calls:
            for tc in delta.tool_calls:
                if tc.function and tc.function.name:
                    tools[tc.index] = tc.function.name
                if tc.function and tc.function.arguments:
                    args[tc.index] = args.get(tc.index, "") + tc.function.arguments
    return (
        content,
        reasoning,
        [tools[k] for k in sorted(tools)],
        [args.get(k, "") for k in sorted(tools)],
    )


class TestArgConverter:
    def test_complete_wrapper(self):
        raw = '{"name":"get_weather","args":{"city":"SF"}}'
        assert _inkling_arg_converter(raw, False) == '{"city":"SF"}'

    def test_partial_before_args(self):
        assert _inkling_arg_converter('{"name":"get_w', True) == ""

    def test_partial_inside_args(self):
        raw = '{"name":"x","args":{"a":1'
        assert _inkling_arg_converter(raw, True) == '{"a":1'

    def test_prefix_stability(self):
        full = '{"name":"x","args":{"a":{"b":[1,2]},"c":"d"}}'
        prev = ""
        for end in range(len(full)):
            out = _inkling_arg_converter(full[:end], True)
            assert out.startswith(prev) or prev.startswith(out) or not prev
            if out.startswith(prev):
                prev = out

    def test_args_value_appearing_in_name(self):
        raw = '{"name":"args","args":{"k":1}}'
        assert _inkling_arg_converter(raw, False) == '{"k":1}'

    def test_whitespace_tolerated(self):
        raw = '{ "name" : "x" , "args" : {"a": 1} }'
        assert _inkling_arg_converter(raw, False) == '{"a": 1}'

    def test_missing_args_defaults_empty(self):
        assert _inkling_arg_converter('{"name":"x"}', False) == "{}"

    def test_non_object_args_rejected(self):
        with pytest.raises(ValueError, match="JSON object"):
            _inkling_arg_converter('{"name":"x","args":[1]}', False)


class TestNonStreaming:
    @pytest.mark.parametrize("suffix", ["", END_MESSAGE, END_SAMPLING])
    def test_bare_text_after_model_opener(self, parser, mock_request, suffix):
        reasoning, content, tools = parser.parse(f"hello world{suffix}", mock_request)
        assert reasoning is None
        assert content == "hello world"
        assert tools is None

    def test_plain_text(self, parser, mock_request):
        reasoning, content, tools = parser.parse(
            f"{TEXT_START}hello world{END_MESSAGE}", mock_request
        )
        assert reasoning is None
        assert content == "hello world"
        assert tools is None

    def test_reasoning_text_tool(self, parser, mock_request):
        text = (
            f"{THINK_START}I should check the weather.{END_MESSAGE}"
            f"{MSG_MODEL}{TEXT_START}Let me check.{END_MESSAGE}"
            f"{MSG_MODEL}" + _tool_block("get_weather", '{"city":"SF"}')
        )
        reasoning, content, tools = parser.parse(text, mock_request)
        assert reasoning == "I should check the weather."
        assert content == "Let me check."
        assert [t.name for t in tools] == ["get_weather"]
        assert json.loads(tools[0].arguments) == {"city": "SF"}

    def test_tool_header_name_is_not_visible_content(self, parser, mock_request):
        text = "get_weather" + _tool_block("get_weather", '{"city":"SF"}')
        _, content, tools = parser.parse(text, mock_request)
        assert content is None
        assert [tool.name for tool in tools] == ["get_weather"]

    def test_parallel_tool_calls(self, parser, mock_request):
        text = _tool_block("a", "{}") + MSG_MODEL + _tool_block("b", '{"x":[1,2]}')
        _, _, tools = parser.parse(text, mock_request)
        assert [t.name for t in tools] == ["a", "b"]
        assert json.loads(tools[0].arguments) == {}
        assert json.loads(tools[1].arguments) == {"x": [1, 2]}

    def test_nested_args(self, parser, mock_request):
        args = '{"q":{"deep":{"list":[{"k":"v"}]}},"s":"a}b"}'
        _, _, tools = parser.parse(_tool_block("f", args), mock_request)
        assert json.loads(tools[0].arguments) == json.loads(args)

    def test_invoke_tool_text_is_visible_text(self, parser, mock_request):
        reasoning, content, tools = parser.parse(
            f"{TOOL_TEXT}do something{END_MESSAGE}", mock_request
        )
        assert content == "do something"
        assert tools is None

    def test_tool_error_is_visible_text(self, parser, mock_request):
        _, content, tools = parser.parse(f"{TOOL_ERROR}boom{END_MESSAGE}", mock_request)
        assert content == "boom"
        assert tools is None

    def test_end_sampling_closes_blocks(self, parser, mock_request):
        reasoning, content, _ = parser.parse(
            f"{THINK_START}hm{END_MESSAGE}{MSG_MODEL}{TEXT_START}hi{END_SAMPLING}",
            mock_request,
        )
        assert reasoning == "hm"
        assert content == "hi"

    def test_multiple_reasoning_blocks_concatenate(self, parser, mock_request):
        text = (
            f"{THINK_START}one{END_MESSAGE}"
            f"{MSG_MODEL}{TEXT_START}mid{END_MESSAGE}"
            f"{MSG_MODEL}{THINK_START}two{END_MESSAGE}"
        )
        reasoning, content, _ = parser.parse(text, mock_request)
        assert reasoning == "onetwo"
        assert content == "mid"

    def test_text_after_tool_call(self, parser, mock_request):
        text = _tool_block("f", "{}") + f"{MSG_MODEL}{TEXT_START}done{END_MESSAGE}"
        _, content, tools = parser.parse(text, mock_request)
        assert [t.name for t in tools] == ["f"]
        assert content == "done"

    def test_incomplete_tool_call_at_eos(self, parser, mock_request):
        # Engine convention: best-effort with what arrived. (The Rust
        # parser instead errors with "incomplete Inkling tool call".)
        _, _, tools = parser.parse(
            f'{TOOL_JSON}{{"name":"d","args":{{"k":"v"', mock_request
        )
        assert [t.name for t in tools] == ["d"]

    def test_prose_marker_without_token_ids_is_structural(self, parser, mock_request):
        # Inkling opts into text-lexer terminal recognition so held-back
        # structural marker text from the detokenizer is still parsed.
        _, content, _ = parser.parse(
            f"{TEXT_START}see {TEXT_START} token{END_MESSAGE}", mock_request
        )
        assert content == "see  token"


class TestStreaming:
    @pytest.mark.parametrize("chunk_size", [1, 3, 7, 64, 4096])
    def test_chunk_invariance_tool_call(self, mock_tokenizer, mock_request, chunk_size):
        parser = InklingParser(mock_tokenizer)
        text = f"{TEXT_START}Check this.{END_MESSAGE}{MSG_MODEL}" + _tool_block(
            "get_weather", '{"city":"San Francisco"}'
        )
        results = _stream(parser, mock_request, text, chunk_size)
        assert collect_content(results) == "Check this."
        assert collect_function_name(results) == "get_weather"
        assert json.loads(collect_tool_arguments(results)) == {"city": "San Francisco"}

    @pytest.mark.parametrize("chunk_size", [1, 3, 7, 64])
    def test_chunk_invariance_tool_call_text_only(
        self, mock_tokenizer, mock_request, chunk_size
    ):
        # Same case through the text-lexing fallback (no token ids),
        # with markers split at arbitrary character boundaries.
        parser = InklingParser(mock_tokenizer)
        text = f"{TEXT_START}Check this.{END_MESSAGE}{MSG_MODEL}" + _tool_block(
            "get_weather", '{"city":"San Francisco"}'
        )
        results = _stream_text_only(parser, mock_request, text, chunk_size)
        assert collect_content(results) == "Check this."
        assert collect_function_name(results) == "get_weather"
        assert json.loads(collect_tool_arguments(results)) == {"city": "San Francisco"}

    @pytest.mark.parametrize("chunk_size", [1, 5, 11])
    def test_chunk_invariance_reasoning(self, mock_tokenizer, mock_request, chunk_size):
        parser = InklingParser(mock_tokenizer)
        text = (
            f"{THINK_START}thinking...{END_MESSAGE}"
            f"{MSG_MODEL}{TEXT_START}answer{END_MESSAGE}"
        )
        results = _stream(parser, mock_request, text, chunk_size)
        assert _collect_reasoning(results) == "thinking..."
        assert collect_content(results) == "answer"

    def test_split_marker_held_across_chunks(self, parser, mock_request):
        # Mirrors Rust `inkling_streaming_holds_split_markers`.
        text = f"{TEXT_START}hello{END_MESSAGE}"
        results = _stream_text_only(parser, mock_request, text, 9)
        assert collect_content(results) == "hello"

    def test_name_streams_before_args_complete(self, parser, mock_request):
        # Feed only up to the name's closing quote — the name delta must
        # already be emitted before any args arrive.
        prefix = f'{TOOL_JSON}{{"name":"get_weather",'
        results = _stream(parser, mock_request, prefix, 4096)
        assert collect_function_name(results) == "get_weather"

    def test_combined_parser_reasoning_to_tool_handoff_uses_text_markers(
        self, mock_tokenizer, mock_request
    ):
        parser_cls = ParserManager.get_parser(
            tool_parser_name="inkling",
            reasoning_parser_name="inkling",
            enable_auto_tools=True,
        )
        parser = parser_cls(mock_tokenizer, [])

        first = parser.parse_delta(
            THINK_START,
            [_TML_VOCAB[THINK_START]],
            mock_request,
            prompt_token_ids=[_TML_VOCAB[MSG_MODEL]],
            finished=False,
        )
        assert first is None

        second = parser.parse_delta(
            "thinking",
            [ord(c) for c in "thinking"],
            mock_request,
            finished=False,
        )
        assert second is not None
        assert second.reasoning == "thinking"

        # Mirrors the DelegatingParser handoff after reasoning closes: the
        # tool pass receives reconstructed text that starts at the Inkling
        # tool marker, while the token-id slice has already moved past it.
        body = (
            "get_weather"
            f'{TOOL_JSON}{{"name":"get_weather","args":{{"city":"Seattle"}}}}'
            f"{END_MESSAGE}"
        )
        third = parser.parse_delta(
            body,
            [_TML_VOCAB[END_MESSAGE], _TML_VOCAB[END_SAMPLING]],
            mock_request,
            finished=True,
        )
        assert third is not None
        assert third.tool_calls
        assert third.tool_calls[0].function.name == "get_weather"
        assert third.tool_calls[0].function.arguments == '{"city":"Seattle"}'
        assert TOOL_JSON not in ((third.content or "") + (third.reasoning or ""))

    def test_streamed_args_are_object_only(self, parser, mock_request):
        # The streamed `arguments` must be the bare args object, never
        # the `{"name":...}` wrapper.
        text = _tool_block("f", '{"a":1}')
        results = _stream(parser, mock_request, text, 3)
        args = collect_tool_arguments(results)
        assert json.loads(args) == {"a": 1}
        assert "name" not in args

    @pytest.mark.parametrize("chunk_size", [1, 9])
    def test_parallel_calls_streaming(self, mock_tokenizer, mock_request, chunk_size):
        parser = InklingParser(mock_tokenizer)
        text = _tool_block("a", '{"i":1}') + MSG_MODEL + _tool_block("b", '{"i":2}')
        results = _stream(parser, mock_request, text, chunk_size)
        indexed: dict[int, dict[str, str]] = {}
        for delta, _ in results:
            if not (delta and delta.tool_calls):
                continue
            for tc in delta.tool_calls:
                slot = indexed.setdefault(tc.index, {"name": "", "args": ""})
                if tc.function and tc.function.name:
                    slot["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    slot["args"] += tc.function.arguments
        assert indexed[0]["name"] == "a"
        assert indexed[1]["name"] == "b"
        assert json.loads(indexed[0]["args"]) == {"i": 1}
        assert json.loads(indexed[1]["args"]) == {"i": 2}


class TestPromptSeededState:
    def test_prompt_ending_in_thinking_starts_reasoning(self, parser, mock_request):
        parser.adjust_initial_state_from_prompt([200001, _TML_VOCAB[THINK_START]])
        assert parser._engine.state == ParserState.REASONING

    def test_prompt_ending_in_text_starts_content(self, parser):
        parser.adjust_initial_state_from_prompt([200001, _TML_VOCAB[TEXT_START]])
        assert parser._engine.state == ParserState.CONTENT

    def test_generation_prompt_tail_starts_message_header(self, parser):
        parser.adjust_initial_state_from_prompt(
            [_TML_VOCAB[END_MESSAGE], _TML_VOCAB[MSG_MODEL]]
        )
        assert parser._engine.state == ParserState.MESSAGE_HEADER

    def test_generation_prompt_header_hides_tool_name(self, parser, mock_request):
        text = "get_weather" + _tool_block("get_weather", '{"city":"SF"}')
        delta = parser.parse_delta(
            text,
            [token_id for token_id, _ in _tokenize(text)],
            mock_request,
            prompt_token_ids=[_TML_VOCAB[END_MESSAGE], _TML_VOCAB[MSG_MODEL]],
            finished=True,
        )
        assert delta is not None
        assert delta.content is None
        assert delta.tool_calls[0].function.name == "get_weather"

    def test_generation_prompt_header_flushes_bare_text_at_finish(
        self, parser, mock_request
    ):
        prompt_token_ids = [_TML_VOCAB[END_MESSAGE], _TML_VOCAB[MSG_MODEL]]
        first = parser.parse_delta(
            "plain ",
            [ord(char) for char in "plain "],
            mock_request,
            prompt_token_ids=prompt_token_ids,
            finished=False,
        )
        assert first is None

        second = parser.parse_delta(
            "answer",
            [ord(char) for char in "answer"],
            mock_request,
            finished=True,
        )
        assert second is not None
        assert second.content == "plain answer"


class TestToolCallFiltering:
    """Inkling equivalents of the generic tool-call-filtering replay tests
    (Inkling is excluded from those in test_replay.py: its structural
    role/kind tokens and shared block-end token don't fit the generic
    reasoning/tool split model)."""

    def test_skip_tool_parsing_round_trip(self, mock_tokenizer, mock_request):
        # First pass (reasoning adapter, skip_tool_parsing): reasoning is
        # classified as reasoning while tool markup survives in content;
        # second pass (tool adapter) re-extracts the calls from it.
        text = (
            f"{THINK_START}plan{END_MESSAGE}{MSG_MODEL}"
            + _tool_block("f", '{"a":1}')
            + MSG_MODEL
            + _tool_block("g", '{"b":[2]}')
        )
        first = InklingParser(mock_tokenizer)
        first.skip_tool_parsing = True
        reasoning, content = first.extract_reasoning(text, mock_request)
        assert reasoning == "plan"
        assert content.count(TOOL_JSON) == 2

        second = InklingParser(mock_tokenizer)
        result = second.extract_tool_calls_from_content(content, mock_request)
        assert result.tools_called
        assert [tc.function.name for tc in result.tool_calls] == ["f", "g"]
        assert json.loads(result.tool_calls[0].function.arguments) == {"a": 1}
        assert json.loads(result.tool_calls[1].function.arguments) == {"b": [2]}

    @pytest.fixture
    def none_request(self, mock_request):
        mock_request.tools = [{"type": "function", "function": {"name": "f"}}]
        mock_request.tool_choice = "none"
        return mock_request

    def test_tool_choice_none_non_streaming(self, mock_tokenizer, none_request):
        parser = InklingParser(mock_tokenizer)
        text = (
            f"{THINK_START}plan{END_MESSAGE}"
            f"{MSG_MODEL}{TEXT_START}visible{END_MESSAGE}"
            f"{MSG_MODEL}" + _tool_block("f", '{"a":1}')
        )
        reasoning, content, tools = parser.parse(text, none_request)
        assert reasoning == "plan"
        assert content == "visible"
        assert not tools

    def test_tool_choice_none_streaming(self, mock_tokenizer, none_request):
        parser = InklingParser(mock_tokenizer)
        text = f"{TEXT_START}visible{END_MESSAGE}{MSG_MODEL}" + _tool_block(
            "f", '{"a":1}'
        )
        results = _stream(parser, none_request, text, 3)
        assert collect_content(results) == "visible"
        assert all(not (d and d.tool_calls) for d, _ in results)


class TestRegisteredAdapters:
    def test_adapters_resolve(self):
        from vllm.reasoning import ReasoningParserManager
        from vllm.tool_parsers import ToolParserManager

        reasoning_cls = ReasoningParserManager.get_reasoning_parser("inkling")
        tool_cls = ToolParserManager.get_tool_parser("inkling")
        assert reasoning_cls._parser_engine_cls is InklingParser
        assert tool_cls._parser_engine_cls is InklingParser
        assert tool_cls.supports_required_and_named is False

    def test_adapter_round_trip(self, mock_tokenizer, mock_request):
        from vllm.tool_parsers import ToolParserManager

        tool_cls = ToolParserManager.get_tool_parser("inkling")
        adapter = tool_cls(mock_tokenizer)
        result = adapter.extract_tool_calls(_tool_block("f", '{"a":1}'), mock_request)
        assert result.tools_called
        assert result.tool_calls[0].function.name == "f"
        assert json.loads(result.tool_calls[0].function.arguments) == {"a": 1}


class TestDelegatingTwoPass:
    """Served reasoning+tool ``DelegatingParser`` path (issue #51387).

    Unlike ``TestStreaming`` above (single engine, ``skip_tool_parsing=False``),
    these drive the real two-pass parser, where the reasoning pass runs with
    ``skip_tool_parsing=True`` — the path where the trailing-marker leak lived.

    The bug is specific to the *tools-enabled* plain-text/reasoning path, so the
    content-only cases (which emit no tool call and would still pass in a
    no-tools mode) offer a real function tool on both the request and the
    parser, keeping ``tool_choice="auto"``.
    """

    GEN_PROMPT = [_TML_VOCAB[MSG_MODEL]]

    def test_plain_text_non_streaming(self, mock_tokenizer, mock_request):
        tools = [_function_tool()]
        mock_request.tools = tools
        _, content, calls = _delegating(mock_tokenizer, tools).parse(
            f"{TEXT_START}The answer is 42.{END_MESSAGE}",
            mock_request,
            enable_auto_tools=True,
        )
        assert content == "The answer is 42."
        assert END_MESSAGE not in content
        assert not calls

    @pytest.mark.parametrize("chunk_size", [1, 3, 7, 64])
    def test_plain_text_streaming(self, mock_tokenizer, mock_request, chunk_size):
        tools = [_function_tool()]
        mock_request.tools = tools
        content, _, calls, _ = _stream_delegating(
            _delegating(mock_tokenizer, tools),
            mock_request,
            f"{TEXT_START}The answer is 42.{END_MESSAGE}",
            chunk_size,
            self.GEN_PROMPT,
        )
        assert content == "The answer is 42."
        assert END_MESSAGE not in content
        assert not calls

    def test_reasoning_then_text(self, mock_tokenizer, mock_request):
        tools = [_function_tool()]
        mock_request.tools = tools
        reasoning, content, _ = _delegating(mock_tokenizer, tools).parse(
            f"{THINK_START}plan{END_MESSAGE}"
            f"{MSG_MODEL}{TEXT_START}Let me look.{END_MESSAGE}",
            mock_request,
            enable_auto_tools=True,
        )
        assert reasoning == "plan"
        assert content == "Let me look."
        assert END_MESSAGE not in content

    def test_multi_tool_round_trip(self, mock_tokenizer, mock_request):
        _, _, tools = _delegating(mock_tokenizer).parse(
            _tool_block("f", '{"a":1}') + MSG_MODEL + _tool_block("g", '{"b":2}'),
            mock_request,
            enable_auto_tools=True,
        )
        assert [t.name for t in tools] == ["f", "g"]

    def test_text_then_tool(self, mock_tokenizer, mock_request):
        _, content, tools = _delegating(mock_tokenizer).parse(
            f"{TEXT_START}intro{END_MESSAGE}{MSG_MODEL}" + _tool_block("f", "{}"),
            mock_request,
            enable_auto_tools=True,
        )
        assert content == "intro"
        assert [t.name for t in tools] == ["f"]

    def test_tool_then_text(self, mock_tokenizer, mock_request):
        _, content, tools = _delegating(mock_tokenizer).parse(
            _tool_block("f", "{}") + MSG_MODEL + f"{TEXT_START}done{END_MESSAGE}",
            mock_request,
            enable_auto_tools=True,
        )
        assert content == "done"
        assert [t.name for t in tools] == ["f"]

    def test_two_tools_then_text(self, mock_tokenizer, mock_request):
        _, content, tools = _delegating(mock_tokenizer).parse(
            _tool_block("f", "{}")
            + MSG_MODEL
            + _tool_block("g", "{}")
            + MSG_MODEL
            + f"{TEXT_START}after{END_MESSAGE}",
            mock_request,
            enable_auto_tools=True,
        )
        assert content == "after"
        assert [t.name for t in tools] == ["f", "g"]

    def test_reasoning_then_tool_non_streaming(self, mock_tokenizer, mock_request):
        reasoning, _, tools = _delegating(mock_tokenizer).parse(
            f"{THINK_START}think{END_MESSAGE}{MSG_MODEL}" + _tool_block("f", '{"x":1}'),
            mock_request,
            enable_auto_tools=True,
        )
        assert reasoning == "think"
        assert [t.name for t in tools] == ["f"]

    def test_end_sampling_text_closer_consumed(self, mock_tokenizer, mock_request):
        tools = [_function_tool()]
        mock_request.tools = tools
        _, content, calls = _delegating(mock_tokenizer, tools).parse(
            f"{TEXT_START}hi{END_SAMPLING}",
            mock_request,
            enable_auto_tools=True,
        )
        assert content == "hi"
        assert END_SAMPLING not in content
        assert not calls

    def test_end_sampling_tool_closer_round_trips(self, mock_tokenizer, mock_request):
        _, _, tools = _delegating(mock_tokenizer).parse(
            f'{TOOL_JSON}{{"name":"f","args":{{}}}}{END_SAMPLING}',
            mock_request,
            enable_auto_tools=True,
        )
        assert [t.name for t in tools] == ["f"]

    def test_incomplete_tool_at_eos(self, mock_tokenizer, mock_request):
        _, _, tools = _delegating(mock_tokenizer).parse(
            f'{TOOL_JSON}{{"name":"d","args":{{"k":"v"',
            mock_request,
            enable_auto_tools=True,
        )
        assert [t.name for t in tools] == ["d"]

    def test_reset_reuse_after_incomplete_span(self, mock_tokenizer, mock_request):
        parser = _delegating(mock_tokenizer)
        parser.parse(
            f'{TOOL_JSON}{{"name":"d","args":{{"k":"v"',
            mock_request,
            enable_auto_tools=True,
        )
        _, content, _ = parser.parse(
            f"{TEXT_START}fresh{END_MESSAGE}",
            mock_request,
            enable_auto_tools=True,
        )
        assert content == "fresh"
        assert END_MESSAGE not in content

    @pytest.mark.parametrize("chunk_size", [1, 3, 64])
    def test_reasoning_then_tool_streaming(
        self, mock_tokenizer, mock_request, chunk_size
    ):
        content, _, tools, _ = _stream_delegating(
            _delegating(mock_tokenizer),
            mock_request,
            f"{THINK_START}plan{END_MESSAGE}{MSG_MODEL}"
            + _tool_block("get_weather", '{"city":"SF"}'),
            chunk_size,
            self.GEN_PROMPT,
        )
        assert tools == ["get_weather"]
        assert TOOL_JSON not in content
        assert END_MESSAGE not in content

    @pytest.mark.parametrize("opener", [TEXT_START, TOOL_TEXT, TOOL_ERROR])
    @pytest.mark.parametrize("chunk_size", [1, 3, 64])
    def test_visible_text_then_tool_streaming(
        self, mock_tokenizer, mock_request, chunk_size, opener
    ):
        """Visible content before a tool call, with no thinking block.

        The reasoning pass leaves its reasoning phase only on an explicit
        reasoning-end event. A response that opens with visible content never
        emitted one, so the tool pass never ran: the entire tool block came
        back as assistant content and no tool call was parsed. Every opener
        that starts a visible block has to confirm the boundary, which is why
        ``TOOL_TEXT`` and ``TOOL_ERROR`` are covered alongside ``TEXT_START``.
        """
        tools = [_function_tool()]
        mock_request.tools = tools
        content, _, names, args = _stream_delegating(
            _delegating(mock_tokenizer, tools),
            mock_request,
            f"{opener}let me check{END_MESSAGE}{MSG_MODEL}"
            + _tool_block("get_weather", '{"city":"SF"}'),
            chunk_size,
            self.GEN_PROMPT,
        )
        assert content == "let me check"
        assert names == ["get_weather"]
        assert json.loads(args[0]) == {"city": "SF"}
        assert TOOL_JSON not in content
        assert END_MESSAGE not in content

    @pytest.mark.parametrize("chunk_size", [1, 3, 7, 64])
    def test_tool_start_from_message_header_streaming(
        self, mock_tokenizer, mock_request, chunk_size
    ):
        """A tool call with no block of any kind ahead of it.

        The generation prompt ends in ``<|message_model|>``, so this fires
        TOOL_START straight from MESSAGE_HEADER. A tool block is the one
        opener that confirms no reasoning is open without rendering visible
        content, so it is the case the visible-block openers cannot cover.
        """
        tools = [_function_tool()]
        mock_request.tools = tools
        content, _, names, args = _stream_delegating(
            _delegating(mock_tokenizer, tools),
            mock_request,
            _tool_block("get_weather", '{"city":"Seattle"}'),
            chunk_size,
            self.GEN_PROMPT,
        )
        assert names == ["get_weather"]
        assert json.loads(args[0]) == {"city": "Seattle"}
        assert content == ""
        assert TOOL_JSON not in content
        assert END_MESSAGE not in content

    def test_function_name_header_before_tool_start_streaming(
        self, mock_tokenizer, mock_request
    ):
        """The optional function name between ``<|message_model|>`` and the
        content-kind marker is metadata: the buffered header must be
        discarded on the way out, not flushed into content."""
        tools = [_function_tool()]
        mock_request.tools = tools
        content, _, names, args = _stream_delegating(
            _delegating(mock_tokenizer, tools),
            mock_request,
            "someFn" + _tool_block("get_weather", '{"city":"Seattle"}'),
            1,
            self.GEN_PROMPT,
        )
        assert names == ["get_weather"]
        assert json.loads(args[0]) == {"city": "Seattle"}
        assert content == ""

    def test_content_state_tool_start_streaming(self, mock_tokenizer, mock_request):
        """Same opener reached from CONTENT rather than MESSAGE_HEADER: a
        text block closed with no ``<|message_model|>`` before the tool
        block. Preceding text must reach content exactly once, unmarked."""
        tools = [_function_tool()]
        mock_request.tools = tools
        content, _, names, args = _stream_delegating(
            _delegating(mock_tokenizer, tools),
            mock_request,
            f"{TEXT_START}intro{END_MESSAGE}"
            + _tool_block("get_weather", '{"city":"Seattle"}'),
            1,
            self.GEN_PROMPT,
        )
        assert names == ["get_weather"]
        assert json.loads(args[0]) == {"city": "Seattle"}
        assert content == "intro"


def test_content_tool_start_emits_reasoning_end_in_reasoning_pass():
    """(CONTENT, TOOL_START) must carry REASONING_END through the
    reasoning pass. Every visible-block opener already confirms the
    boundary (#49876), but the header-flush path
    (MESSAGE_HEADER --END_MESSAGE--> CONTENT) reaches CONTENT without
    one, so a tool block opening from there relies on this transition
    alone to hand off to the tool pass."""
    engine = StreamingParserEngine(inkling_config(), tokenizer=None)
    engine.skip_tool_parsing = True
    engine.reset(initial_state=ParserState.CONTENT)
    events = engine.parse_complete(f'{TOOL_JSON}{{"name":"f","args":{{}}}}')
    assert [e.type for e in events[:2]] == [
        EventType.REASONING_END,
        EventType.TEXT_CHUNK,
    ]
    assert events[1].value == TOOL_JSON


def test_skip_reasoning_parsing_inert_for_shared_markers():
    """``skip_reasoning_parsing`` may only bypass reasoning-exclusive
    markers. Inkling has none — ``<|end_message|>`` is labelled THINK_END
    yet also closes text, header, and tool blocks — so the flag must be
    inert: a tool block still closes through its transition and following
    text returns to CONTENT instead of leaking into the argument stream."""
    engine = StreamingParserEngine(inkling_config(), tokenizer=None)
    engine.skip_reasoning_parsing = True
    engine.reset(initial_state=ParserState.CONTENT)
    events = engine.parse_complete(
        f'{TOOL_JSON}{{"name":"f","args":{{}}}}{END_MESSAGE}after'
    )
    types = [e.type for e in events]
    end_idx = types.index(EventType.TOOL_CALL_END)
    text_after = [e.value for e in events[end_idx:] if e.type == EventType.TEXT_CHUNK]
    assert text_after == ["after"]


class TestToolParserWithoutReasoningParser:
    """Inkling served with only the tool parser, no ``--reasoning-parser``.

    That configuration turns on ``skip_reasoning_parsing``, and Inkling
    labels ``<|end_message|>`` as THINK_END; a bypass keyed on the label
    would neutralize the closer of every block kind, leaking markers into
    content and leaving tool calls unterminated. Structure must keep
    parsing exactly as with the reasoning parser attached."""

    GEN_PROMPT = [_TML_VOCAB[MSG_MODEL]]

    def _tool_only(self, mock_tokenizer, tools=None):
        parser_cls = ParserManager.get_parser(
            tool_parser_name="inkling",
            enable_auto_tools=True,
        )
        return parser_cls(mock_tokenizer, tools or [])

    def test_plain_text_non_streaming(self, mock_tokenizer, mock_request):
        tools = [_function_tool()]
        mock_request.tools = tools
        _, content, calls = self._tool_only(mock_tokenizer, tools).parse(
            f"{TEXT_START}The answer is 42.{END_MESSAGE}",
            mock_request,
            enable_auto_tools=True,
        )
        assert content == "The answer is 42."
        assert not calls

    @pytest.mark.parametrize("chunk_size", [1, 3, 64])
    def test_plain_text_streaming(self, mock_tokenizer, mock_request, chunk_size):
        tools = [_function_tool()]
        mock_request.tools = tools
        content, _, names, _ = _stream_delegating(
            self._tool_only(mock_tokenizer, tools),
            mock_request,
            f"{TEXT_START}The answer is 42.{END_MESSAGE}",
            chunk_size,
            self.GEN_PROMPT,
        )
        assert content == "The answer is 42."
        assert not names

    def test_tool_block_non_streaming(self, mock_tokenizer, mock_request):
        _, content, calls = self._tool_only(mock_tokenizer).parse(
            _tool_block("get_weather", '{"city":"SF"}'),
            mock_request,
            enable_auto_tools=True,
        )
        assert [c.name for c in calls] == ["get_weather"]
        assert not content

    @pytest.mark.parametrize("chunk_size", [1, 3, 64])
    def test_tool_block_streaming(self, mock_tokenizer, mock_request, chunk_size):
        tools = [_function_tool()]
        mock_request.tools = tools
        content, _, names, args = _stream_delegating(
            self._tool_only(mock_tokenizer, tools),
            mock_request,
            _tool_block("get_weather", '{"city":"SF"}'),
            chunk_size,
            self.GEN_PROMPT,
        )
        assert names == ["get_weather"]
        assert json.loads(args[0]) == {"city": "SF"}
        assert content == ""

    def test_thinking_then_text_keeps_structure(self, mock_tokenizer, mock_request):
        tools = [_function_tool()]
        mock_request.tools = tools
        _, content, calls = self._tool_only(mock_tokenizer, tools).parse(
            f"{THINK_START}plan{END_MESSAGE}"
            f"{MSG_MODEL}{TEXT_START}Let me look.{END_MESSAGE}",
            mock_request,
            enable_auto_tools=True,
        )
        assert content == "Let me look."
        assert END_MESSAGE not in content
        assert THINK_START not in content
        assert not calls

    def test_thinking_then_tool_still_promotes(self, mock_tokenizer, mock_request):
        _, content, calls = self._tool_only(mock_tokenizer).parse(
            f"{THINK_START}think{END_MESSAGE}{MSG_MODEL}"
            + _tool_block("get_weather", '{"x":1}'),
            mock_request,
            enable_auto_tools=True,
        )
        assert [c.name for c in calls] == ["get_weather"]
        assert not content
