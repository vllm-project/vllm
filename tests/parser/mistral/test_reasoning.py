# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
from mistral_common.protocol.instruct.messages import (
    AssistantMessage,
    ThinkChunk,
    ToolMessage,
    UserMessage,
)
from mistral_common.protocol.instruct.request import InstructRequest
from mistral_common.protocol.instruct.tool_calls import FunctionCall, ToolCall

from tests.reasoning.utils import run_reasoning_extraction_mistral
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.parser_manager import ParserManager
from vllm.reasoning import ReasoningParser, ReasoningParserManager
from vllm.tokenizers.detokenizer_utils import detokenize_incrementally
from vllm.tokenizers.mistral import MistralTokenizer

_PARSER_NAME = "mistral"
_MODEL_V13 = "mistralai/Magistral-Small-2509"
_MODEL_V11 = "mistralai/Magistral-Small-2506"

_SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]


@pytest.fixture(scope="module")
def mistral_tokenizer() -> MistralTokenizer:
    """v13 Magistral tokenizer with `[THINK]`/`[/THINK]` special tokens."""
    return MistralTokenizer.from_pretrained(_MODEL_V13)


@pytest.fixture(scope="module")
def mistral_v11_tokenizer() -> MistralTokenizer:
    """v11 Magistral tokenizer using plain-text `<think>`/`</think>` tags."""
    return MistralTokenizer.from_pretrained(_MODEL_V11)


def _encode_v13(tokenizer: MistralTokenizer, output: str) -> list[int]:
    """Encode output string with `[THINK]`/`[/THINK]` markers into token IDs.

    `[THINK]` and `[/THINK]` placeholders in `output` are replaced by their
    special-token IDs; all surrounding text is encoded with the base tokenizer.
    Mirrors the encoding helper from the deleted
    `tests/reasoning/test_mistral_reasoning_parser.py`.
    """
    think_start = "[THINK]"
    think_end = "[/THINK]"
    len_start = len(think_start)
    len_end = len(think_end)

    index_start = output.find(think_start)
    index_end = output.find(think_end)

    out: list[int] = []
    if index_start != -1:
        out += tokenizer.tokenizer.encode(output[:index_start], False, False)
        out += [tokenizer.instruct.BEGIN_THINK]

        if index_end != -1:
            middle = output[index_start + len_start : index_end]
            suffix = output[index_end + len_end :]
            out += tokenizer.tokenizer.encode(middle, False, False)
            out += [tokenizer.instruct.END_THINK]
            out += tokenizer.tokenizer.encode(suffix, False, False)
        else:
            out += tokenizer.tokenizer.encode(
                output[index_start + len_start :], False, False
            )
    elif index_end != -1:
        out += tokenizer.tokenizer.encode(output[:index_end], False, False)
        out += [tokenizer.instruct.END_THINK]
        out += tokenizer.tokenizer.encode(output[index_end + len_end :], False, False)
    else:
        out += tokenizer.tokenizer.encode(output, False, False)
    return out


_TEST_CASES = [
    # v13 special-token encoding: [THINK]…[/THINK] with trailing content.
    pytest.param(
        False,
        "[THINK]r[/THINK]c",
        "r",
        "c",
        id="v13_basic",
    ),
    pytest.param(
        True,
        "[THINK]r[/THINK]c",
        "r",
        "c",
        id="v13_basic_streaming",
    ),
    # No trailing content after reasoning ends.
    pytest.param(
        False,
        "[THINK]r[/THINK]",
        "r",
        None,
        id="v13_no_trailing_content",
    ),
    pytest.param(
        True,
        "[THINK]r[/THINK]",
        "r",
        None,
        id="v13_no_trailing_content_streaming",
    ),
    # Multi-line reasoning and multi-line content.
    pytest.param(
        False,
        "[THINK]a\nb[/THINK]c\nd",
        "a\nb",
        "c\nd",
        id="v13_multiline",
    ),
    pytest.param(
        True,
        "[THINK]a\nb[/THINK]c\nd",
        "a\nb",
        "c\nd",
        id="v13_multiline_streaming",
    ),
    # No reasoning at all: output is pure content.
    pytest.param(
        False,
        "hello world",
        None,
        "hello world",
        id="v13_no_reasoning",
    ),
    pytest.param(
        True,
        "hello world",
        None,
        "hello world",
        id="v13_no_reasoning_streaming",
    ),
    # [THINK] opened but never closed; reasoning accumulates, content is None.
    pytest.param(
        False,
        "[THINK]r",
        "r",
        None,
        id="v13_think_no_end",
    ),
    pytest.param(
        True,
        "[THINK]r",
        "r",
        None,
        id="v13_think_no_end_streaming",
    ),
    # Stray [/THINK] with no matching [THINK]: no reasoning, content follows.
    pytest.param(
        False,
        "[/THINK]This is the rest",
        None,
        "This is the rest",
        id="v13_stray_think_end",
    ),
    pytest.param(
        True,
        "[/THINK]This is the rest",
        None,
        "This is the rest",
        id="v13_stray_think_end_streaming",
    ),
    # Content before [THINK]: leading segment joins with trailing segment.
    pytest.param(
        False,
        "Before\n[THINK]r[/THINK]\nAfter",
        "r",
        "Before\n\nAfter",
        id="v13_leading_content",
    ),
    pytest.param(
        True,
        "Before\n[THINK]r[/THINK]\nAfter",
        "r",
        "Before\n\nAfter",
        id="v13_leading_content_streaming",
    ),
    # Empty output: no reasoning, no content.
    pytest.param(
        False,
        "",
        None,
        None,
        id="v13_empty",
    ),
    pytest.param(
        True,
        "",
        None,
        None,
        id="v13_empty_streaming",
    ),
]


@pytest.mark.parametrize(
    "streaming, output, expected_reasoning, expected_content",
    _TEST_CASES,
)
def test_mistral_reasoning_v13(
    streaming: bool,
    output: str,
    expected_reasoning: str | None,
    expected_content: str | None,
    mistral_tokenizer: MistralTokenizer,
) -> None:
    output_tokens = _encode_v13(mistral_tokenizer, output)
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(_PARSER_NAME)(
        mistral_tokenizer
    )
    reasoning, content = run_reasoning_extraction_mistral(
        parser, output_tokens, streaming=streaming
    )
    assert reasoning == expected_reasoning
    assert content == expected_content


def test_system_prompt_think_ignored(
    mistral_tokenizer: MistralTokenizer,
) -> None:
    """Streaming content extraction must not
    include `[THINK]`/`[/THINK]` examples from the system prompt.
    """
    output = "[THINK]real[/THINK]answer"
    output_tokens = _encode_v13(mistral_tokenizer, output)
    for streaming in (False, True):
        parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(
            _PARSER_NAME
        )(mistral_tokenizer)
        reasoning, content = run_reasoning_extraction_mistral(
            parser, output_tokens, streaming=streaming
        )
        assert reasoning == "real"
        assert content == "answer"


def test_reasoning_with_tool_calls(
    mistral_tokenizer: MistralTokenizer,
) -> None:
    """Reasoning extraction with a tool call in the generated output."""
    tool_calls_token_id = mistral_tokenizer.get_vocab().get("[TOOL_CALLS]")
    if tool_calls_token_id is None:
        pytest.skip("Tokenizer does not have [TOOL_CALLS] in vocab")

    # Token sequence representing: [THINK]r[/THINK][TOOL_CALLS]f{"a":1}
    output_tokens = (
        [mistral_tokenizer.instruct.BEGIN_THINK]
        + mistral_tokenizer.tokenizer.encode("r", False, False)
        + [mistral_tokenizer.instruct.END_THINK]
        + [tool_calls_token_id]
        + mistral_tokenizer.tokenizer.encode('f{"a":1}', False, False)
    )
    for streaming in (False, True):
        parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(
            _PARSER_NAME
        )(mistral_tokenizer)
        reasoning, _content = run_reasoning_extraction_mistral(
            parser, output_tokens, streaming=streaming
        )
        assert reasoning == "r"
        assert "[TOOL_CALLS]" not in (reasoning or "")


@pytest.mark.parametrize("streaming", [False, True], ids=["nonstream", "stream"])
def test_reasoning_v11_plain_text_think(
    mistral_v11_tokenizer: MistralTokenizer, streaming: bool
) -> None:
    """v11 tokenizers use plain-text `<think>`/`</think>` (no special tokens)."""
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(_PARSER_NAME)(
        mistral_v11_tokenizer
    )

    tokens = mistral_v11_tokenizer.tokenizer.encode("<think>r</think>c", False, False)
    reasoning, content = run_reasoning_extraction_mistral(
        parser, tokens, streaming=streaming
    )
    assert reasoning == "r"
    assert content == "c"


def test_is_reasoning_end(
    mistral_tokenizer: MistralTokenizer,
) -> None:
    """is_reasoning_end returns True iff reasoning has finished."""
    parser: ReasoningParser = ReasoningParserManager.get_reasoning_parser(_PARSER_NAME)(
        mistral_tokenizer
    )

    # Complete reasoning: END_THINK present → True.
    complete_ids = _encode_v13(mistral_tokenizer, "[THINK]r[/THINK]c")
    assert parser.is_reasoning_end(complete_ids)

    # Open-only: no END_THINK, BEGIN_THINK present → False.
    open_ids = _encode_v13(mistral_tokenizer, "[THINK]r")
    assert not parser.is_reasoning_end(open_ids)

    # No reasoning markers at all → False.
    no_reasoning_ids = _encode_v13(mistral_tokenizer, "hello")
    assert not parser.is_reasoning_end(no_reasoning_ids)

    # Explicit END_THINK closes reasoning → True.
    explicit_end_ids = (
        [mistral_tokenizer.instruct.BEGIN_THINK]
        + mistral_tokenizer.tokenizer.encode("r", False, False)
        + [mistral_tokenizer.instruct.END_THINK]
    )
    assert parser.is_reasoning_end(explicit_end_ids)

    # [TOOL_CALLS] acts as implicit reasoning-end marker.
    tool_calls_token_id = mistral_tokenizer.get_vocab().get("[TOOL_CALLS]")
    if tool_calls_token_id is None:
        pytest.skip("Tokenizer does not have [TOOL_CALLS] in vocab")

    implicit_end_ids = (
        [mistral_tokenizer.instruct.BEGIN_THINK]
        + mistral_tokenizer.tokenizer.encode("r", False, False)
        + [tool_calls_token_id]
    )
    assert parser.is_reasoning_end(implicit_end_ids)


def _stream_parse_delta(parser, tokenizer, gen_ids, request, prompt_ids):
    """Stream `gen_ids` through `parse_delta` and reconstruct reasoning/content."""
    reasoning = ""
    content = ""
    previous_tokens: list[str] | None = None
    prefix_offset = 0
    read_offset = 0
    for i, tok in enumerate(gen_ids):
        new_tokens, delta_text, prefix_offset, read_offset = detokenize_incrementally(
            tokenizer=tokenizer,
            all_input_ids=gen_ids[: i + 1],
            prev_tokens=previous_tokens,
            prefix_offset=prefix_offset,
            read_offset=read_offset,
            skip_special_tokens=True,
            spaces_between_special_tokens=True,
        )
        previous_tokens = (
            previous_tokens + new_tokens if previous_tokens else new_tokens
        )
        delta_message = parser.parse_delta(
            delta_text=delta_text,
            delta_token_ids=[tok],
            request=request,
            prompt_token_ids=prompt_ids,
            finished=(i == len(gen_ids) - 1),
        )
        if delta_message is not None:
            reasoning += delta_message.reasoning or ""
            content += delta_message.content or ""
    return reasoning, content


def test_parse_delta_keeps_reasoning_open_after_closed_think_prompt(
    mistral_tokenizer: MistralTokenizer,
) -> None:
    """A closed `[THINK]` block in the prompt must not finalize reasoning.

    Regression: reasoning system prompts end the prompt with a closed
    `[THINK]...[/THINK]` block. With a parser-injected grammar the generation's
    reasoning must still be captured as ``reasoning_content`` rather than
    leaking into ``content`` (the prompt-based reasoning-end heuristic must be
    skipped). This exercises ``DelegatingParser.parse_delta``'s prompt check,
    which the isolated reasoning-parser tests do not cover.
    """
    parser_cls = ParserManager.get_parser(
        tool_parser_name="mistral",
        reasoning_parser_name="mistral",
        enable_auto_tools=True,
    )
    parser = parser_cls(mistral_tokenizer, None)
    request = parser.adjust_request(
        ChatCompletionRequest(
            messages=[],
            model="test",
            tools=_SAMPLE_TOOLS,
            tool_choice="auto",
        )
    )
    assert getattr(request, "_grammar_from_parser", False)

    prompt_ids = _encode_v13(mistral_tokenizer, "[THINK]system reasoning[/THINK]")
    gen_ids = _encode_v13(
        mistral_tokenizer,
        "[THINK]because two plus two is four[/THINK]The answer is 4.",
    )

    reasoning, content = _stream_parse_delta(
        parser, mistral_tokenizer, gen_ids, request, prompt_ids
    )

    assert "because two plus two is four" in reasoning
    assert "because two plus two is four" not in content
    assert "The answer is 4." in content


def _encode_gen(tokenizer: MistralTokenizer, text: str) -> list[int]:
    """Encode `text` mapping `[THINK]`, `[/THINK]`, `[TOOL_CALLS]`, `[ARGS]`
    to their special-token IDs; all other text uses the base tokenizer.
    """
    vocab = tokenizer.get_vocab()
    markers: dict[str, int] = {
        "[THINK]": tokenizer.instruct.BEGIN_THINK,
        "[/THINK]": tokenizer.instruct.END_THINK,
        "[TOOL_CALLS]": vocab["[TOOL_CALLS]"],
        "[ARGS]": vocab["[ARGS]"],
    }
    out: list[int] = []
    remaining = text
    while remaining:
        earliest: int | None = None
        earliest_marker: str | None = None
        for marker in markers:
            idx = remaining.find(marker)
            if idx != -1 and (earliest is None or idx < earliest):
                earliest = idx
                earliest_marker = marker
        if earliest_marker is None:
            out += tokenizer.tokenizer.encode(remaining, False, False)
            break
        assert earliest is not None  # set together with earliest_marker
        if earliest > 0:
            out += tokenizer.tokenizer.encode(remaining[:earliest], False, False)
        out.append(markers[earliest_marker])
        remaining = remaining[earliest + len(earliest_marker) :]
    return out


def _stream_turn(
    parser,
    tokenizer: MistralTokenizer,
    gen_ids: list[int],
    request,
    prompt_ids: list[int],
) -> tuple[str, str, list[tuple[str, str]]]:
    """Stream `gen_ids` through `parse_delta`, returning `(reasoning, content,
    tool_calls)`.

    `tool_calls` is a list of `(name, arguments)` tuples ordered by
    `DeltaToolCall.index`.  Arguments are concatenated across deltas.
    """
    reasoning = ""
    content = ""
    tool_call_acc: dict[int, dict[str, str]] = {}
    previous_tokens: list[str] | None = None
    prefix_offset = 0
    read_offset = 0
    for i, tok in enumerate(gen_ids):
        new_tokens, delta_text, prefix_offset, read_offset = detokenize_incrementally(
            tokenizer=tokenizer,
            all_input_ids=gen_ids[: i + 1],
            prev_tokens=previous_tokens,
            prefix_offset=prefix_offset,
            read_offset=read_offset,
            skip_special_tokens=True,
            spaces_between_special_tokens=True,
        )
        previous_tokens = (
            previous_tokens + new_tokens if previous_tokens else new_tokens
        )
        delta_message = parser.parse_delta(
            delta_text=delta_text,
            delta_token_ids=[tok],
            request=request,
            prompt_token_ids=prompt_ids,
            finished=(i == len(gen_ids) - 1),
        )
        if delta_message is not None:
            reasoning += delta_message.reasoning or ""
            content += delta_message.content or ""
            for tc in delta_message.tool_calls:
                if tc.index not in tool_call_acc:
                    tool_call_acc[tc.index] = {"name": "", "args": ""}
                if tc.function:
                    if tc.function.name:
                        tool_call_acc[tc.index]["name"] = tc.function.name
                    if tc.function.arguments:
                        tool_call_acc[tc.index]["args"] += tc.function.arguments
    tool_calls = [
        (tool_call_acc[k]["name"], tool_call_acc[k]["args"])
        for k in sorted(tool_call_acc)
    ]
    return reasoning, content, tool_calls


def test_parse_delta_multi_turn_reason_and_tool(
    mistral_tokenizer: MistralTokenizer,
) -> None:
    """Two-turn loop: each assistant turn carries reasoning + a tool call.

    For each turn, the mistral-common conversation is encoded into `prompt_ids`
    before streaming the next generation.  Turn 2's prompt therefore contains
    a `[THINK]`/`[/THINK]` block and `[TOOL_CALLS]` from turn 1 — the
    accumulated tool-call/result history that single-turn tests never exercise.

    Asserts that across both turns the parser correctly classifies think tokens
    as `reasoning_content` (never leaking into `content`), and that the tool
    call name/arguments are reconstructed correctly from the streaming deltas.
    """
    parser_cls = ParserManager.get_parser(
        tool_parser_name="mistral",
        reasoning_parser_name="mistral",
        enable_auto_tools=True,
    )
    request = parser_cls(mistral_tokenizer, None).adjust_request(
        ChatCompletionRequest(
            messages=[],
            model="test",
            tools=_SAMPLE_TOOLS,
            tool_choice="auto",
        )
    )
    assert getattr(request, "_grammar_from_parser", False)

    cities = ["Dallas", "Paris"]
    conversation: list = [UserMessage(content="What is the weather in Dallas?")]

    for i, city in enumerate(cities):
        # Fresh parser per turn: streaming state must not bleed across turns.
        parser = parser_cls(mistral_tokenizer, None)

        prompt_ids = mistral_tokenizer.instruct.encode_instruct(
            InstructRequest(messages=conversation)
        ).tokens

        gen_str = (
            f"[THINK]I should check {city} weather.[/THINK]"
            f'[TOOL_CALLS]get_weather[ARGS]{{"city": "{city}"}}'
        )
        gen_ids = _encode_gen(mistral_tokenizer, gen_str)

        reasoning, content, tool_calls = _stream_turn(
            parser, mistral_tokenizer, gen_ids, request, prompt_ids
        )

        assert f"check {city} weather" in reasoning, f"turn {i}: reasoning not captured"
        assert f"check {city} weather" not in content, (
            f"turn {i}: reasoning leaked into content"
        )
        assert "[THINK]" not in content, f"turn {i}: [THINK] leaked"
        assert "[TOOL_CALLS]" not in content, f"turn {i}: [TOOL_CALLS] leaked"
        assert len(tool_calls) == 1, f"turn {i}: expected 1 tool call"
        name, args = tool_calls[0]
        assert name == "get_weather", f"turn {i}: wrong tool name {name!r}"
        assert city in args, f"turn {i}: city not in args {args!r}"

        # Extend conversation with this turn's history for the next turn.
        tc_id = f"tc{i:07d}"  # 9-char alphanumeric id: "tc0000000", "tc0000001"
        conversation.extend(
            [
                AssistantMessage(
                    content=[
                        ThinkChunk(
                            thinking=f"I should check {city} weather.",
                            closed=True,
                        )
                    ],
                    tool_calls=[
                        ToolCall(
                            id=tc_id,
                            function=FunctionCall(
                                name="get_weather",
                                arguments=json.dumps({"city": city}),
                            ),
                        )
                    ],
                ),
                ToolMessage(
                    content=f"{city}: 70F sunny",
                    tool_call_id=tc_id,
                ),
            ]
        )
        if i < len(cities) - 1:
            conversation.append(UserMessage(content=f"And in {cities[i + 1]}?"))
