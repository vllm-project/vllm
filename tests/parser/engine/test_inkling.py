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


# ── Issue #50512: a turn that never opens a thinking block ────────────
# `<|message_model|>` is prefilled by the generation prompt, so generation
# starts in MESSAGE_HEADER. Which state TOOL_START then fires from depends
# only on what the turn emits first; it is verified per test name below.

_TOOL_TURN = _tool_block("get_weather", '{"city":"Seattle"}')

# TOOL_START from MESSAGE_HEADER — the reported trigger.
BARE_TOOL_TURN = _TOOL_TURN
# Same, plus the optional function-name header the renderer emits between
# `<|message_model|>` and the content-kind marker; exercises the
# MESSAGE_HEADER buffer on the way out.
NAMED_TOOL_TURN = "get_weather" + _TOOL_TURN
# Text block, a fresh message header, then the tool call: still
# MESSAGE_HEADER, but with visible text that must survive intact.
TEXT_HEADER_TOOL_TURN = (
    f"{TEXT_START}Let me check.{END_MESSAGE}{MSG_MODEL}" + _TOOL_TURN
)
# Text block, tool block, no intervening `<|message_model|>`: TOOL_START
# fires from CONTENT instead — the same hole, one state over.
TEXT_TOOL_TURN = f"{TEXT_START}Let me check.{END_MESSAGE}" + _TOOL_TURN
# Control: opens with reasoning, so REASONING_END already arrives from
# `(REASONING, THINK_END)`. This path always worked.
REASONING_TOOL_TURN = (
    f"{THINK_START}I should check.{END_MESSAGE}{MSG_MODEL}" + _TOOL_TURN
)

# One token per delta and the whole turn in one delta — the two
# boundaries. Intermediate sizes add cost without adding a code path.
CHUNK_BOUNDARIES = [1, 4096]

SINGLE_TURN_PROMPT = f"{TEXT_START}hi{END_MESSAGE}{MSG_MODEL}"
# A prior assistant tool call plus its result, i.e. the shape that exists
# from turn 2 onwards. Both prompts share the
# `<|end_message|><|message_model|>` tail the renderer always emits.
MULTI_TURN_PROMPT = (
    f"{TEXT_START}hi{END_MESSAGE}"
    f"{MSG_MODEL}" + _tool_block("get_weather", '{"city":"SF"}') + f"{TEXT_START}sunny"
    f"{END_MESSAGE}{MSG_MODEL}"
)


def _delegating_parser(mock_tokenizer, *, reasoning: bool = True):
    parser_cls = ParserManager.get_parser(
        tool_parser_name="inkling",
        reasoning_parser_name="inkling" if reasoning else None,
        enable_auto_tools=True,
    )
    return parser_cls(mock_tokenizer, [])


def _stream_delta(parser, request, text: str, chunk_size: int, prompt: str):
    """Drive ``parse_delta`` the way the serving layer does — one call per
    delta, ``prompt_token_ids`` passed every time, ``finished`` on the last.

    This is the entry point where the reasoning-pass → tool-pass handoff
    lives, which is why these cases cannot go through ``_stream``.
    Results are shaped like ``_stream``'s so the shared collectors apply.
    """
    prompt_token_ids = [token_id for token_id, _ in _tokenize(prompt)]
    tokens = _tokenize(text)
    results = []
    for start in range(0, len(tokens), chunk_size):
        batch = tokens[start : start + chunk_size]
        delta = parser.parse_delta(
            delta_text="".join(t for _, t in batch),
            delta_token_ids=[token_id for token_id, _ in batch],
            request=request,
            prompt_token_ids=prompt_token_ids,
            finished=start + chunk_size >= len(tokens),
        )
        results.append((delta, text))
    return results


def _assert_get_weather_call(results, expected_content: str) -> None:
    """The tool block became a ``get_weather`` call and content is exactly
    ``expected_content``.

    When a text block precedes the tool call, a trailing bare
    ``<|end_message|>`` is tolerated: that is the separate end-of-block
    leakage of #49865, which travels the ``(CONTENT, THINK_END)`` path and
    is out of scope here. A turn with no text block cannot trigger #49865,
    so it gets no such amnesty and content must be exactly empty.
    """
    content = collect_content(results)
    assert collect_function_name(results) == "get_weather", (
        f"no get_weather call; content={content!r}"
    )
    assert json.loads(collect_tool_arguments(results)) == {"city": "Seattle"}
    if expected_content:
        content = content.removesuffix(END_MESSAGE)
    assert content == expected_content


class TestToolStartWithoutThinking:
    """Issue #50512: a turn reaching its tool block without first closing a
    thinking block leaked the whole tool markup into ``delta.content`` and
    emitted no tool calls.

    The discriminant is the missing leading thinking block, not turn count.
    The reasoning pass runs with ``skip_tool_parsing`` and hands off to the
    tool pass only once it has seen a REASONING_END; a turn opening with a
    thinking block gets one from ``(REASONING, THINK_END)``, and a turn that
    does not used to get none at all.
    """

    @pytest.mark.parametrize("chunk_size", CHUNK_BOUNDARIES)
    def test_tool_start_from_message_header(
        self, mock_tokenizer, mock_request, chunk_size
    ):
        results = _stream_delta(
            _delegating_parser(mock_tokenizer),
            mock_request,
            BARE_TOOL_TURN,
            chunk_size,
            SINGLE_TURN_PROMPT,
        )
        _assert_get_weather_call(results, "")

    @pytest.mark.parametrize("chunk_size", CHUNK_BOUNDARIES)
    def test_tool_start_from_content(self, mock_tokenizer, mock_request, chunk_size):
        results = _stream_delta(
            _delegating_parser(mock_tokenizer),
            mock_request,
            TEXT_TOOL_TURN,
            chunk_size,
            SINGLE_TURN_PROMPT,
        )
        _assert_get_weather_call(results, "Let me check.")

    @pytest.mark.parametrize("chunk_size", CHUNK_BOUNDARIES)
    def test_visible_text_survives_exactly_once(
        self, mock_tokenizer, mock_request, chunk_size
    ):
        """Text block, fresh header, tool call: the text must reach content
        once — neither dropped nor replayed by the handoff."""
        results = _stream_delta(
            _delegating_parser(mock_tokenizer),
            mock_request,
            TEXT_HEADER_TOOL_TURN,
            chunk_size,
            SINGLE_TURN_PROMPT,
        )
        _assert_get_weather_call(results, "Let me check.")

    def test_function_name_header_before_tool_block(self, mock_tokenizer, mock_request):
        """The buffered header name must be discarded, not emitted."""
        results = _stream_delta(
            _delegating_parser(mock_tokenizer),
            mock_request,
            NAMED_TOOL_TURN,
            1,
            SINGLE_TURN_PROMPT,
        )
        _assert_get_weather_call(results, "")

    @pytest.mark.parametrize(
        "turn, expected_content",
        [
            pytest.param(BARE_TOOL_TURN, "", id="from_message_header"),
            pytest.param(TEXT_TOOL_TURN, "Let me check.", id="from_content"),
        ],
    )
    def test_multi_turn_prompt_is_no_different(
        self, mock_tokenizer, mock_request, turn, expected_content
    ):
        """Turn depth is not the discriminant: the same turns behave the
        same after a prompt that already contains a tool call and result."""
        results = _stream_delta(
            _delegating_parser(mock_tokenizer),
            mock_request,
            turn,
            1,
            MULTI_TURN_PROMPT,
        )
        _assert_get_weather_call(results, expected_content)

    @pytest.mark.parametrize(
        "turn, expected_content",
        [
            pytest.param(BARE_TOOL_TURN, None, id="from_message_header"),
            pytest.param(TEXT_TOOL_TURN, "Let me check.", id="from_content"),
        ],
    )
    def test_non_streaming_agrees_with_streaming(
        self, parser, mock_request, turn, expected_content
    ):
        """Streaming and non-streaming must reach the same answer. The
        single-pass path never sets ``skip_tool_parsing``, so it was always
        correct; this is the invariant the issue is really about."""
        _, content, tools = parser.parse(turn, mock_request)
        assert [t.name for t in tools] == ["get_weather"]
        assert json.loads(tools[0].arguments) == {"city": "Seattle"}
        assert content == expected_content

    def test_tool_only_parser_is_unaffected(self, mock_tokenizer, mock_request):
        """With no reasoning parser there is no handoff to get wrong, so
        this cell must stay green independently of the reasoning pass."""
        results = _stream_delta(
            _delegating_parser(mock_tokenizer, reasoning=False),
            mock_request,
            BARE_TOOL_TURN,
            1,
            SINGLE_TURN_PROMPT,
        )
        _assert_get_weather_call(results, "")

    def test_reasoning_first_turn_still_hands_off(self, mock_tokenizer, mock_request):
        """Control: the path that already worked must keep working, and the
        second REASONING_END on the tool terminal must stay a no-op."""
        results = _stream_delta(
            _delegating_parser(mock_tokenizer),
            mock_request,
            REASONING_TOOL_TURN,
            1,
            SINGLE_TURN_PROMPT,
        )
        assert _collect_reasoning(results) == "I should check."
        _assert_get_weather_call(results, "")


class TestMessageHeaderExitsWithoutReasoningEnd:
    """``<|end_message|>`` and ``<|content_model_end_sampling|>`` also leave
    MESSAGE_HEADER through the ``skip_tool_parsing`` gate, but carry no
    REASONING_END. They pin the reordered branch on Inkling's real table:
    the generic config in ``test_engine.py`` cannot show that.
    """

    @pytest.mark.parametrize("terminator", [END_MESSAGE, END_SAMPLING])
    def test_no_reasoning_end_and_header_discarded(self, mock_tokenizer, terminator):
        engine = StreamingParserEngine(inkling_config(), mock_tokenizer)
        engine.reset(initial_state=ParserState.MESSAGE_HEADER)
        engine.skip_tool_parsing = True
        events = engine.parse_complete(f"get_weather{terminator}")
        assert EventType.REASONING_END not in [event.type for event in events]
        assert engine._message_header_buffer == ""


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
