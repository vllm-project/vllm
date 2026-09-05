# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""parse_delta-level regressions for MuseGlimmer's unified channel parser."""

import json
from types import SimpleNamespace

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.exceptions import VLLMValidationError
from vllm.parser import ParserManager
from vllm.reasoning.muse_glimmer_reasoning_parser import MuseGlimmerReasoningParser
from vllm.reasoning.muse_glimmer_utils import advance_emitted
from vllm.sampling_params import StructuredOutputsParams

REASONING_MODEL_NAME = "meta-models/Muse-Glimmer-30B"
PROMPT = "<|start|>user<|message|>hi<|eom|><|start|>assistant"
TOOL_XML = (
    "<atem:function_calls>\n"
    '<atem:invoke name="weather.get">\n'
    '<atem:parameter name="city">Paris</atem:parameter>\n'
    "</atem:invoke>\n"
    "</atem:function_calls>"
)
EMPTY_TOOL_XML = (
    "<atem:function_calls>\n"
    '<atem:invoke name="weather.get">\n'
    "</atem:invoke>\n"
    "</atem:function_calls>"
)
FRAMING = (
    "<|start|>",
    "<|message|>",
    "<|eom|>",
    "<|eot|>",
    "to=self",
    "to=user",
    "<atem:",
)


class CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(map(ord, text))

    def decode(self, ids):
        return "".join(chr(token_id) for token_id in ids)

    def get_vocab(self):
        return {}


@pytest.fixture(scope="module", params=("char", "real"))
def tokenizer(request):
    if request.param == "char":
        return CharTokenizer()

    from transformers import AutoTokenizer

    # meta-models/Muse-Glimmer-30B is public, but CI/dev without network (or an
    # empty HF cache) cannot fetch it. Skip rather than hard-error so the
    # checkpoint-free ``char`` variants still exercise every case.
    try:
        return AutoTokenizer.from_pretrained(
            REASONING_MODEL_NAME, trust_remote_code=True
        )
    except Exception as exc:  # noqa: BLE001 - any load failure -> skip
        pytest.skip(f"MuseGlimmer tokenizer unavailable: {exc}")


def encode(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def tool_names(tools):
    return [name for _index, name, _arguments in tools]


def drive(
    tokenizer,
    chunks,
    *,
    prompt=PROMPT,
    with_tool_parser=True,
    request=None,
):
    parser_kwargs = {"reasoning_parser_name": "muse_glimmer"}
    if with_tool_parser:
        parser_kwargs.update(
            tool_parser_name="muse_glimmer",
            enable_auto_tools=True,
        )
    parser = ParserManager.get_parser(**parser_kwargs)(tokenizer)
    if request is None:
        request = SimpleNamespace(
            tools=None,
            tool_choice="auto",
            include_reasoning=True,
        )

    prompt_ids = encode(tokenizer, prompt)
    messages = []
    for index, chunk in enumerate(chunks):
        message = parser.parse_delta(
            chunk,
            encode(tokenizer, chunk),
            request,
            prompt_token_ids=prompt_ids if index == 0 else None,
            finished=index == len(chunks) - 1,
        )
        if message is not None:
            messages.append(message)

    reasoning = "".join(message.reasoning or "" for message in messages)
    content = "".join(message.content or "" for message in messages)
    tools = []
    for message in messages:
        for tool in message.tool_calls or []:
            function = tool.function
            name = function.get("name") if isinstance(function, dict) else function.name
            arguments = (
                function.get("arguments")
                if isinstance(function, dict)
                else function.arguments
            )
            tools.append((tool.index, name, arguments))
    return reasoning, content, tools


def drive_tokenwise(tokenizer, text, **kwargs):
    ids = encode(tokenizer, text)
    chunks = [tokenizer.decode([token_id]) for token_id in ids]
    return drive(tokenizer, chunks, **kwargs)


def assert_no_framing(text):
    for marker in FRAMING:
        assert marker not in text, f"framing {marker!r} leaked: {text!r}"


def test_reasoning_then_answer(tokenizer):
    reasoning, content, tools = drive_tokenwise(
        tokenizer,
        " to=self<|message|>Let me think step by step.<|eom|>"
        "<|start|>assistant to=user<|message|>The answer is 42.<|eot|>",
    )
    assert reasoning == "Let me think step by step."
    assert content == "The answer is 42."
    assert tools == []
    assert_no_framing(reasoning + content)


def test_nonstreaming_multiple_reasoning_blocks(tokenizer):
    parser = ParserManager.get_parser(
        reasoning_parser_name="muse_glimmer",
        tool_parser_name="muse_glimmer",
        enable_auto_tools=True,
    )(tokenizer)
    request = SimpleNamespace(
        tools=None,
        tool_choice="auto",
        include_reasoning=True,
    )
    reasoning, content, tools = parser.parse(
        "<|start|>assistant to=self<|message|>step one<|eom|>"
        "<|start|>assistant to=self<|message|>step two<|eom|>"
        "<|start|>assistant to=user<|message|>done.<|eot|>",
        request,
        enable_auto_tools=True,
    )
    assert reasoning == "step one\nstep two"
    assert content == "done."
    assert not tools


def test_streaming_multiple_reasoning_blocks(tokenizer):
    reasoning, content, tools = drive_tokenwise(
        tokenizer,
        "<|start|>assistant to=self<|message|>step one<|eom|>"
        "<|start|>assistant to=self<|message|>step two<|eom|>"
        "<|start|>assistant to=user<|message|>done.<|eot|>",
    )
    assert reasoning == "step one\nstep two"
    assert content == "done."
    assert tools == []


def test_tool_choice_none_streams_clean_answer(tokenizer):
    request = SimpleNamespace(
        tools=None,
        tool_choice="none",
        include_reasoning=True,
    )
    reasoning, content, tools = drive_tokenwise(
        tokenizer,
        " to=self<|message|>Check the result.<|eom|>"
        "<|start|>assistant to=user<|message|>The answer is 42.<|eot|>",
        request=request,
    )
    assert reasoning == "Check the result."
    assert content == "The answer is 42."
    assert tools == []
    assert_no_framing(reasoning + content)


def test_content_only(tokenizer):
    reasoning, content, tools = drive_tokenwise(
        tokenizer, " to=user<|message|>Just a direct answer.<|eot|>"
    )
    assert reasoning == ""
    assert content == "Just a direct answer."
    assert tools == []
    assert_no_framing(content)


def test_tool_call(tokenizer):
    reasoning, content, tools = drive_tokenwise(
        tokenizer,
        " to=self<|message|>I should read the hostname.<|eom|>"
        "<|start|>assistant to=read.read<|message|>"
        '<atem:function_calls>\n<atem:invoke name="read.read">\n'
        '<atem:parameter name="path">/etc/hostname</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls>",
    )
    assert reasoning == "I should read the hostname."
    assert content == ""
    assert len(tools) == 1
    index, name, arguments = tools[0]
    assert index == 0
    assert name == "read.read"
    assert json.loads(arguments) == {"path": "/etc/hostname"}


def test_streaming_tool_call_carries_id_and_type(tokenizer):
    """The streamed tool call must expose a nonempty id and ``type="function"``
    alongside its index, name, and arguments. ``drive`` flattens tool calls to
    ``(index, name, arguments)``, so assert the full metadata on the raw
    ``DeltaMessage`` objects here.
    """
    parser = ParserManager.get_parser(
        reasoning_parser_name="muse_glimmer",
        tool_parser_name="muse_glimmer",
        enable_auto_tools=True,
    )(tokenizer)
    request = SimpleNamespace(tools=None, tool_choice="auto", include_reasoning=True)
    text = (
        " to=self<|message|>I should read the hostname.<|eom|>"
        "<|start|>assistant to=read.read<|message|>"
        '<atem:function_calls>\n<atem:invoke name="read.read">\n'
        '<atem:parameter name="path">/etc/hostname</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls>"
    )
    prompt_ids = encode(tokenizer, PROMPT)
    token_ids = encode(tokenizer, text)
    collected = []
    for position, token_id in enumerate(token_ids):
        message = parser.parse_delta(
            tokenizer.decode([token_id]),
            [token_id],
            request,
            prompt_token_ids=prompt_ids if position == 0 else None,
            finished=position == len(token_ids) - 1,
        )
        if message is not None:
            collected.extend(message.tool_calls or [])
    assert len(collected) == 1
    call = collected[0]
    assert call.index == 0
    assert call.type == "function"
    assert call.id
    function = call.function
    name = function.get("name") if isinstance(function, dict) else function.name
    assert name == "read.read"


def test_truncated_cot_does_not_parse_contemplated_call(tokenizer):
    reasoning, content, tools = drive_tokenwise(
        tokenizer,
        " to=self<|message|>Maybe I should call "
        '<atem:function_calls>\n<atem:invoke name="read.read">\n'
        '<atem:parameter name="path">/etc/hostname</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls> but wait",
    )
    assert tools == []
    assert content == ""
    assert "Maybe I should call" in reasoning


def test_reasoning_can_be_suppressed(tokenizer):
    request = SimpleNamespace(
        tools=None,
        tool_choice="auto",
        include_reasoning=False,
    )
    reasoning, content, tools = drive_tokenwise(
        tokenizer,
        " to=self<|message|>secret thoughts<|eom|>"
        "<|start|>assistant to=user<|message|>Public answer.<|eot|>",
        request=request,
    )
    assert reasoning == ""
    assert content == "Public answer."
    assert tools == []


def test_continued_user_channel_surfaces_clean_content(tokenizer):
    prompt = PROMPT + " to=user<|message|>"
    reasoning, content, tools = drive(
        tokenizer,
        ['{"value":"x"}<|eot|>'],
        prompt=prompt,
    )
    bare_reasoner = MuseGlimmerReasoningParser(tokenizer)
    assert bare_reasoner.is_reasoning_end(encode(tokenizer, prompt))
    assert reasoning == ""
    assert content == '{"value":"x"}'
    assert tools == []


def test_continued_user_channel_surfaces_clean_content_reasoning_only(tokenizer):
    # Same as above but with no tool parser: the reasoning-only composite must
    # still stream the continued to=user body as clean content.
    prompt = PROMPT + " to=user<|message|>"
    reasoning, content, tools = drive(
        tokenizer,
        ['{"value":"x"}<|eot|>'],
        prompt=prompt,
        with_tool_parser=False,
    )
    assert reasoning == ""
    assert content == '{"value":"x"}'
    assert tools == []


def test_nonstreaming_answer_with_tools_auto_is_clean(tokenizer):
    # Non-streaming parse(): tools registered + tool_choice="auto", but the model
    # answers without calling a tool. The final content must be the clean answer,
    # with no channel framing leaked (regression guard: extract_reasoning returns
    # the raw framed turn, and the composite must strip it when no call fires).
    parser = ParserManager.get_parser(
        reasoning_parser_name="muse_glimmer",
        tool_parser_name="muse_glimmer",
        enable_auto_tools=True,
    )(tokenizer)
    request = SimpleNamespace(
        tools=[
            {
                "type": "function",
                "function": {"name": "weather.get", "parameters": {"type": "object"}},
            }
        ],
        tool_choice="auto",
        include_reasoning=True,
    )
    model_output = (
        " to=self<|message|>Simple. 3+3 = 6. No tool needed.<|eom|>"
        "<|start|>assistant to=user<|message|>3 + 3 = 6<|eot|>"
    )
    reasoning, content, tools = parser.parse(
        model_output, request, enable_auto_tools=True
    )
    assert reasoning == "Simple. 3+3 = 6. No tool needed."
    assert content == "3 + 3 = 6"
    for marker in FRAMING:
        assert marker not in (content or ""), (marker, content)
    assert not tools


def test_quoted_bare_tool_header_stays_in_reasoning(tokenizer):
    quoted = "I am quoting: to=weather.get<|message|> as plain text."
    reasoning, content, tools = drive(tokenizer, [" to=self<|message|>" + quoted])
    assert reasoning == quoted
    assert content == ""
    assert tools == []


def test_bare_tool_switch_with_atem_after_open_reasoning_parses(tokenizer):
    reasoning, content, tools = drive(
        tokenizer,
        [
            " to=self<|message|>think then to=weather.get<|message|>"
            + EMPTY_TOOL_XML
            + "<|eot|>"
        ],
    )
    assert reasoning == "think then "
    assert content == ""
    assert tool_names(tools) == ["weather.get"]


def test_framed_answer_closes_open_reasoning(tokenizer):
    reasoning, content, tools = drive(
        tokenizer,
        [
            " to=self<|message|>think"
            "<|start|>assistant to=user<|message|>The answer is 42.<|eot|>"
        ],
    )
    assert reasoning == "think"
    assert content == "The answer is 42."
    assert tools == []


def test_closed_prompt_reasoning_does_not_seed_generation(tokenizer):
    prompt = PROMPT + " to=self<|message|>Prior.<|eom|>"
    reasoner = MuseGlimmerReasoningParser(tokenizer)
    reasoner.adjust_initial_state_from_prompt(encode(tokenizer, prompt))
    reasoning, content, tools = drive(
        tokenizer,
        [" <|start|>assistant to=user<|message|>The answer is 42.<|eot|>"],
        prompt=prompt,
    )
    assert reasoner._initial_recipient is None
    assert reasoning == ""
    assert content == "The answer is 42."
    assert tools == []


def test_fully_framed_tool_switch_after_open_reasoning_still_parses(tokenizer):
    reasoning, content, tools = drive(
        tokenizer,
        [
            " to=self<|message|>Need weather.",
            "<|start|>assistant to=weather.get<|message|>" + TOOL_XML,
        ],
    )
    assert reasoning == "Need weather."
    assert content == ""
    assert tool_names(tools) == ["weather.get"]


def test_answer_tail_is_emitted_before_straddled_tool_call(tokenizer):
    reasoning, content, tools = drive(
        tokenizer,
        [
            " to=user<|message|>The answer",
            " tail.<|start|>assistant to=weather.get<|message|>",
            TOOL_XML,
        ],
    )
    assert reasoning == ""
    assert content == "The answer tail."
    assert tool_names(tools) == ["weather.get"]


def test_sequential_tool_channels_in_one_delta_are_preserved(tokenizer):
    second = TOOL_XML.replace("weather.get", "weather.forecast")
    reasoning, content, tools = drive(
        tokenizer,
        [
            " to=self<|message|>Need both.<|eom|>"
            "<|start|>assistant to=weather.get<|message|>"
            + TOOL_XML
            + "<|eom|><|start|>assistant to=weather.forecast<|message|>"
            + second
            + "<|eot|>"
        ],
    )
    assert reasoning == "Need both."
    assert content == ""
    assert tool_names(tools) == ["weather.get", "weather.forecast"]


def test_truncated_user_body_flushes_held_tail(tokenizer):
    reasoning, content, tools = drive(
        tokenizer,
        [" to=user<|message|>Comparison: 2 <"],
    )
    assert reasoning == ""
    assert content == "Comparison: 2 <"
    assert tools == []


def test_reasoning_only_keeps_straddled_answer_tail(tokenizer):
    reasoning, content, tools = drive(
        tokenizer,
        [
            " to=self<|message|>think<|eom|>"
            "<|start|>assistant to=user<|message|>answer before ",
            "tool<|eom|><|start|>assistant to=weather.get<|message|>"
            + TOOL_XML
            + "<|eot|>",
        ],
        with_tool_parser=False,
    )
    assert reasoning == "think"
    assert content == "answer before tool"
    assert tools == []
    assert_no_framing(content)


def test_reasoning_only_tool_channel_yields_no_content(tokenizer):
    reasoning, content, tools = drive(
        tokenizer,
        [
            " to=self<|message|>think<|eom|>"
            "<|start|>assistant to=weather.get<|message|>" + TOOL_XML + "<|eot|>"
        ],
        with_tool_parser=False,
    )
    assert reasoning == "think"
    assert content == ""
    assert tools == []


def test_streaming_never_leaks_partial_tool_header(tokenizer):
    generation = (
        " to=self<|message|>maybe to=weather.get<|message|>"
        + EMPTY_TOOL_XML
        + "<|eot|>"
    )
    reasoning, content, tools = drive_tokenwise(tokenizer, generation)
    assert reasoning == "maybe "
    assert content == ""
    assert tool_names(tools) == ["weather.get"]
    assert_no_framing(reasoning)


def test_streaming_preserves_words_ending_in_t(tokenizer):
    reasoning, content, tools = drive_tokenwise(
        tokenizer, " to=self<|message|>the most important point"
    )
    assert reasoning == "the most important point"
    assert content == ""
    assert tools == []


def test_answer_may_quote_atem_markup(tokenizer):
    """An answer addressed to the user may legitimately quote ATEM markup, e.g.
    when the question is about tool-call syntax. Withholding such a body loses
    the whole answer, and tool-call parsing is scoped to recipient-tagged
    bodies, so the quoted markup can never become a call.
    """
    answer = 'Call it like <atem:invoke name="weather.get"> and close it.'
    reasoning, content, tools = drive(
        tokenizer,
        [
            " to=self<|message|>explain the syntax<|eom|>"
            "<|start|>assistant to=user<|message|>" + answer + "<|eot|>"
        ],
    )
    assert reasoning == "explain the syntax"
    assert content == answer
    assert tools == []


def test_framed_header_without_space_still_bounds_a_body(tokenizer):
    """``MSG_HEADER_RE`` accepts a framed header with no space before
    ``<|message|>``, so the body-boundary pattern must too. If it did not, the
    preceding body would be cut at the stray ``<|start|>`` and this message
    would be skipped entirely, dropping its body.
    """
    reasoning, content, tools = drive(
        tokenizer,
        [
            " to=self<|message|>think"
            "<|start|>assistant<|message|>The answer is 42.<|eot|>"
        ],
    )
    assert reasoning == "think"
    assert content == "The answer is 42."
    assert tools == []
    assert_no_framing(content)


@pytest.mark.parametrize(
    ("emitted", "current", "expected"),
    [
        ("abc", "abcdef", ("def", "abcdef")),
        ("", "new", ("new", "new")),
        ("abc", "abc", ("", "abc")),
        # A reclassified body legitimately shrinks -- a partial header becomes
        # recognisable and is trimmed, or a body stops qualifying as content.
        # The cursor must hold so already-streamed text is not re-emitted.
        ("abcdef", "abc", ("", "abcdef")),
        ("abc", "", ("", "abc")),
        # A body that shrank and regrew with a different prefix must not have
        # the old cursor applied to the new text.
        ("abc", "xyz", ("", "abc")),
    ],
)
def test_advance_emitted_never_moves_cursor_backwards(emitted, current, expected):
    assert advance_emitted(emitted, current) == expected


@pytest.mark.parametrize("constraint", ["response_format", "structured_outputs"])
def test_active_tools_reject_caller_output_constraint(constraint):
    request_kwargs = {}
    if constraint == "response_format":
        request_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": {"type": "object"}},
        }
    else:
        request_kwargs["structured_outputs"] = StructuredOutputsParams(
            json={"type": "object"}
        )
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}],
        tools=[
            {
                "type": "function",
                "function": {"name": "weather.get", "parameters": {"type": "object"}},
            }
        ],
        tool_choice="auto",
        **request_kwargs,
    )
    parser = ParserManager.get_parser(
        reasoning_parser_name="muse_glimmer",
        tool_parser_name="muse_glimmer",
        enable_auto_tools=True,
    )(CharTokenizer())
    with pytest.raises(VLLMValidationError, match="cannot be combined"):
        parser.adjust_request(request)


def test_responses_text_format_rejected_with_tools():
    request = ResponsesRequest.model_validate(
        {
            "model": "test-model",
            "input": "hi",
            "tools": [
                {
                    "type": "function",
                    "name": "weather.get",
                    "parameters": {"type": "object"},
                    "strict": True,
                }
            ],
            "tool_choice": "auto",
            "text": {"format": {"type": "json_object"}, "verbosity": "high"},
        }
    )
    parser = ParserManager.get_parser(
        reasoning_parser_name="muse_glimmer",
        tool_parser_name="muse_glimmer",
        enable_auto_tools=True,
    )(CharTokenizer())
    with pytest.raises(VLLMValidationError, match="cannot be combined"):
        parser.adjust_request(request)


def test_active_tools_allow_plain_text_response_format():
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}],
        tools=[
            {
                "type": "function",
                "function": {"name": "weather.get", "parameters": {"type": "object"}},
            }
        ],
        tool_choice="auto",
        response_format={"type": "text"},
    )
    parser = ParserManager.get_parser(
        reasoning_parser_name="muse_glimmer",
        tool_parser_name="muse_glimmer",
        enable_auto_tools=True,
    )(CharTokenizer())
    adjusted = parser.adjust_request(request)
    assert adjusted.response_format is not None
    assert adjusted.response_format.type == "text"


def test_responses_tools_allow_plain_text_format_and_preserve_text_fields():
    request = ResponsesRequest.model_validate(
        {
            "model": "test-model",
            "input": "hi",
            "tools": [
                {
                    "type": "function",
                    "name": "weather.get",
                    "parameters": {"type": "object"},
                    "strict": True,
                }
            ],
            "tool_choice": "auto",
            "text": {"format": {"type": "text"}, "verbosity": "high"},
        }
    )
    parser = ParserManager.get_parser(
        reasoning_parser_name="muse_glimmer",
        tool_parser_name="muse_glimmer",
        enable_auto_tools=True,
    )(CharTokenizer())
    adjusted = parser.adjust_request(request)
    assert adjusted.text is not None
    assert adjusted.text.format is not None
    assert adjusted.text.format.type == "text"
    assert adjusted.text.verbosity == "high"


@pytest.mark.parametrize("constraint", ["response_format", "structured_outputs"])
def test_reasoning_only_parser_preserves_output_constraint(constraint):
    request_kwargs = {}
    if constraint == "response_format":
        request_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": {"type": "object"}},
        }
    else:
        request_kwargs["structured_outputs"] = StructuredOutputsParams(
            json={"type": "object"}
        )
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}],
        tools=[
            {
                "type": "function",
                "function": {"name": "weather.get", "parameters": {"type": "object"}},
            }
        ],
        tool_choice="auto",
        **request_kwargs,
    )
    parser = ParserManager.get_parser(reasoning_parser_name="muse_glimmer")(
        CharTokenizer()
    )
    parser.adjust_request(request)
    if constraint == "response_format":
        assert request.response_format is not None
    else:
        assert request.structured_outputs is not None
