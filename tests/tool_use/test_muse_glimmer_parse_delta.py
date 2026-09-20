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
    tool_parser_name="muse_glimmer",
    request=None,
):
    parser_kwargs = {"reasoning_parser_name": "muse_glimmer"}
    if with_tool_parser:
        parser_kwargs.update(
            tool_parser_name=tool_parser_name,
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
    # Quoted ATEM stays reasoning text; channel framing must never leak.
    for marker in ("<|start|>", "<|message|>", "<|eom|>", "<|eot|>"):
        assert marker not in reasoning


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


def test_engine_seed_boundary_is_the_bare_reasoners(tokenizer):
    # serving.py seeds the engine-side grammar from the BARE reasoner: its
    # boundary (any non-self channel, incl. to=user) is wider than the
    # composite's stream-ownership rule (tool-parser handoff only).
    bare = MuseGlimmerReasoningParser(tokenizer)
    composite = ParserManager.get_parser(
        reasoning_parser_name="muse_glimmer",
        tool_parser_name="muse_glimmer",
        enable_auto_tools=True,
    )(tokenizer)
    mid_reasoning = encode(tokenizer, " to=self<|message|>thinking")
    mid_answer = encode(tokenizer, " to=user<|message|>the answer")
    assert not bare.is_reasoning_end(mid_reasoning)
    assert bare.is_reasoning_end(mid_answer)
    assert composite.is_reasoning_end(mid_reasoning)
    assert composite.is_reasoning_end(mid_answer)


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
            (
                " to=self<|message|>think"
                "<|start|>assistant to=user<|message|>The answer is 42.<|eot|>"
            )
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


def test_truncated_user_body_drops_partial_marker(tokenizer):
    reasoning, content, tools = drive(
        tokenizer,
        [" to=user<|message|>answer<|eo"],
    )
    assert reasoning == ""
    assert content == "answer"
    assert tools == []


def test_truncated_reasoning_flushes_held_tail(tokenizer):
    reasoning, content, tools = drive_tokenwise(
        tokenizer,
        " to=self<|message|>thinking hard about to",
        with_tool_parser=False,
    )
    assert reasoning == "thinking hard about to"
    assert content == ""
    assert tools == []


@pytest.mark.parametrize(
    "preamble",
    [
        "Some preamble ",
        "This deliberately long preamble extends beyond the final answer ",
    ],
    ids=("short", "long"),
)
def test_untagged_atem_does_not_hide_following_answer(tokenizer, preamble):
    reasoning, content, tools = drive_tokenwise(
        tokenizer,
        "<|message|>"
        + preamble
        + TOOL_XML
        + "<|eot|><|start|>assistant to=user<|message|>real answer<|eot|>",
    )
    assert reasoning == ""
    # The prefix before the markup is kept; the markup itself never surfaces.
    assert content == preamble + "real answer"
    assert tools == []
    assert_no_framing(content)


def test_mixed_hermes_tool_parser_rejected(tokenizer):
    # A foreign tool parser cannot read ATEM channels: the mixed configuration
    # is rejected at construction instead of leaking raw framing.
    with pytest.raises(VLLMValidationError, match="tool-call-parser"):
        ParserManager.get_parser(
            reasoning_parser_name="muse_glimmer",
            tool_parser_name="hermes",
            enable_auto_tools=True,
        )(tokenizer)


def test_closed_body_preserves_quoted_start_marker(tokenizer):
    reasoning, content, tools = drive_tokenwise(
        tokenizer,
        " to=self<|message|>quote <|start|>garbage here<|eom|>"
        "<|start|>assistant to=user<|message|>done<|eot|>",
    )
    assert reasoning == "quote <|start|>garbage here"
    assert content == "done"
    assert tools == []


def test_reasoning_only_keeps_straddled_answer_tail(tokenizer):
    reasoning, content, tools = drive(
        tokenizer,
        [
            (
                " to=self<|message|>think<|eom|>"
                "<|start|>assistant to=user<|message|>answer before "
            ),
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


def test_closed_body_preserves_literal_partial_header(tokenizer):
    # A literal `<|start|>assistant` (no <|message|>) inside a CLOSED body is
    # user text, not a partial header: kept, and the body stays closed.
    reasoning, content, tools = drive_tokenwise(
        tokenizer, " to=user<|message|>note <|start|>assistant<|eom|>"
    )
    assert reasoning == ""
    assert content == "note <|start|>assistant"
    assert tools == []


def test_word_glued_bare_tool_header_stays_text(tokenizer):
    # A `to=` glued to a word is not a channel boundary (bare switches
    # require whitespace anchoring AND immediate ATEM markup).
    reasoning, content, tools = drive_tokenwise(
        tokenizer,
        " to=self<|message|>think xto=calc<|message|>" + EMPTY_TOOL_XML + "<|eot|>",
    )
    assert tools == []
    assert "xto=calc" in reasoning


def test_bare_tool_header_with_whitespace_before_atem_stays_reasoning(tokenizer):
    # The defect switch requires ATEM *immediately* after the header.
    reasoning, content, tools = drive_tokenwise(
        tokenizer,
        (
            " to=self<|message|>think then to=calc<|message|>\n"
            + EMPTY_TOOL_XML
            + "<|eot|>"
        ),
    )
    assert tools == []
    assert "to=calc" in reasoning


def test_finish_drops_complete_bare_header(tokenizer):
    # A trailing COMPLETE bare header (a channel that never got a body) is
    # framing and drops.
    reasoning, content, tools = drive(
        tokenizer,
        [" to=self<|message|>thinking to=calc<|message|>"],
        with_tool_parser=False,
    )
    assert reasoning == "thinking "
    assert content == ""
    assert tools == []


def test_untagged_prefill_continuation_streams_content(tokenizer):
    prompt = PROMPT + "<|message|>"
    reasoning, content, tools = drive(
        tokenizer, ['{"value":"x"}<|eot|>'], prompt=prompt, with_tool_parser=False
    )
    assert reasoning == ""
    assert content == '{"value":"x"}'
    assert tools == []


def test_tool_only_prefill_continuation_streams_content(tokenizer):
    # Tool-only config (no reasoning parser): the composite must seed from the
    # prompt's open channel itself, or the continued body is silently dropped.
    parser = ParserManager.get_parser(
        tool_parser_name="muse_glimmer", enable_auto_tools=True
    )(tokenizer)
    prompt_ids = encode(tokenizer, PROMPT + " to=user<|message|>")
    request = SimpleNamespace(tools=None, tool_choice="auto", include_reasoning=True)
    text = '{"value":"x"}<|eot|>'
    ids = encode(tokenizer, text)
    chunks = [tokenizer.decode([token_id]) for token_id in ids]
    content = ""
    for index, chunk in enumerate(chunks):
        message = parser.parse_delta(
            chunk,
            encode(tokenizer, chunk),
            request,
            prompt_token_ids=prompt_ids if index == 0 else None,
            finished=index == len(chunks) - 1,
        )
        if message is not None and message.content:
            content += message.content
    assert content == '{"value":"x"}'


def test_tool_only_nonstreaming_strips_channel_framing(tokenizer):
    # Tool-only config, non-streaming: no reasoner runs, so the composite must
    # strip the channel framing itself.
    parser = ParserManager.get_parser(
        tool_parser_name="muse_glimmer", enable_auto_tools=True
    )(tokenizer)
    request = SimpleNamespace(tools=None, tool_choice="auto", include_reasoning=True)
    _reasoning, content, tools = parser.parse(
        " to=user<|message|>The answer is 42.<|eot|>", request, enable_auto_tools=True
    )
    assert content == "The answer is 42."
    assert not tools


def test_bare_self_user_header_with_atem_stays_text(tokenizer):
    # Even with ATEM right after it, a bare self/user header never switches
    # channels: it is quoted text.
    reasoning, content, tools = drive_tokenwise(
        tokenizer,
        " to=self<|message|>quoting: to=user<|message|>" + EMPTY_TOOL_XML + "<|eot|>",
    )
    assert tools == []
    assert "to=user" in reasoning


def test_open_body_preserves_quoted_start_marker(tokenizer):
    # A quoted `<|start|>` inside an OPEN body is literal text too, and
    # truncation must not eat it.
    reasoning, content, tools = drive_tokenwise(
        tokenizer, " to=self<|message|>quote <|start|>garbage here"
    )
    assert reasoning == "quote <|start|>garbage here"
    assert tools == []


def test_newline_anchored_bare_tool_header_switches_cleanly(tokenizer):
    # The defect switch also fires after a newline anchor, with no `to`
    # fragment ever leaking into the stream first.
    reasoning, content, tools = drive_tokenwise(
        tokenizer,
        " to=self<|message|>think\nto=calc<|message|>" + EMPTY_TOOL_XML + "<|eot|>",
    )
    # The invoke's name attribute is the call name (the header's is not
    # consulted), so a mismatched header recipient passes through verbatim.
    assert tool_names(tools) == ["weather.get"]
    assert reasoning == "think\n"


def test_frozen_untagged_body_prefix_streams(tokenizer):
    # An untagged body cut by a later framed header is frozen: its prefix was
    # validatable then and must not be withheld forever.
    reasoning, content, tools = drive_tokenwise(
        tokenizer, "<|message|>pre <|start|>assistant to=user<|message|>done<|eot|>"
    )
    assert reasoning == ""
    assert content == "pre done"
    assert tools == []


def test_finish_keeps_trailing_self_user_bare_header(tokenizer):
    # A trailing bare self/user header was already streamed as text
    # mid-stream, so finish must not retract it (unlike a tool header, which
    # is held back the whole way).
    reasoning, content, tools = drive(
        tokenizer,
        [" to=self<|message|>thinking to=user<|message|>"],
        with_tool_parser=False,
    )
    assert reasoning == "thinking to=user<|message|>"
    assert tools == []


def test_overlong_recipient_name_never_forms_a_header(tokenizer):
    # A >1KB recipient can never be a real channel header: the cap keeps the
    # body instead of losing it (the over-long name itself is pre-header junk
    # here and drops).
    name = "a" * 2048
    parser = ParserManager.get_parser(reasoning_parser_name="muse_glimmer")(tokenizer)
    request = SimpleNamespace(tools=None, tool_choice="auto", include_reasoning=True)
    _reasoning, content, tools = parser.parse(
        f" to={name}<|message|>the answer<|eot|>", request
    )
    assert tools == []
    assert content == "the answer"


def test_finish_keeps_bare_header_followed_by_newline(tokenizer):
    # A bare header followed by a newline is stream-validated text; the finish
    # flush must not strip it (absolute-end anchoring, not `$`-before-\n).
    reasoning, content, tools = drive(
        tokenizer,
        [" to=user<|message|>ans to=x<|message|\n"],
        with_tool_parser=False,
    )
    assert reasoning == ""
    assert content == "ans to=x<|message|\n"
    assert tools == []


def test_flush_never_strips_frozen_body_tails(tokenizer):
    # The frozen body's tail is validated text; a partial-marker lookalike
    # spanning the join must not strip it, in either path.
    generation = "<|message|><|<|eot|><|message|>e"
    reasoning, content, tools = drive_tokenwise(tokenizer, generation)
    assert content == "<|e"
    assert tools == []

    parser = ParserManager.get_parser(
        reasoning_parser_name="muse_glimmer",
        tool_parser_name="muse_glimmer",
        enable_auto_tools=True,
    )(tokenizer)
    request = SimpleNamespace(tools=None, tool_choice="auto", include_reasoning=True)
    _reasoning, content, _tools = parser.parse(
        generation, request, enable_auto_tools=True
    )
    assert content == "<|e"


def test_truncated_lone_angle_bracket_is_text(tokenizer):
    # A lone `<` is model text, not framing; only an actual marker-in-progress
    # (`<|…`) is stripped at finish.
    reasoning, _content, tools = drive_tokenwise(tokenizer, " to=self<|message|>cut <")
    assert reasoning == "cut <"
    assert tools == []


def test_unframed_grammar_output_streams_as_content(tokenizer):
    # A grammar-constrained answer that never opened a channel streams as
    # plain content instead of vanishing.
    for with_tool_parser in (True, False):
        reasoning, content, tools = drive_tokenwise(
            tokenizer,
            '{"answer": 42}',
            with_tool_parser=with_tool_parser,
        )
        assert reasoning == ""
        assert content == '{"answer": 42}'
        assert tools == []


def test_muse_kimi_mixed_pairing_rejected(tokenizer):
    # The kimi_k3/cohere special-case blocks must not bypass the muse pairing
    # validation: any muse-involving mix routes to the muse composite first.
    with pytest.raises(VLLMValidationError, match="tool-call-parser"):
        ParserManager.get_parser(
            reasoning_parser_name="muse_glimmer",
            tool_parser_name="kimi_k3",
            enable_auto_tools=True,
        )(tokenizer)


def test_muse_foreign_reasoning_pairing_rejected(tokenizer):
    # The reverse mix is rejected too: a foreign reasoning parser reports a
    # boundary the composite's stream ownership does not expect.
    with pytest.raises(VLLMValidationError, match="reasoning-parser"):
        ParserManager.get_parser(
            reasoning_parser_name="qwen3",
            tool_parser_name="muse_glimmer",
            enable_auto_tools=True,
        )(tokenizer)


def test_unframed_to_framed_transition_loses_nothing(tokenizer):
    # A header arriving in pieces (no leading space, so the first delta is a
    # bare `to`) must not wedge the content cursor: once framing completes,
    # the answer streams normally.
    reasoning, content, tools = drive_tokenwise(
        tokenizer,
        "to=self<|message|>think<|eom|>to=user<|message|>the answer<|eot|>",
    )
    assert reasoning == "think"
    assert content == "the answer"
    assert tools == []


def test_unframed_prefix_text_survives_late_framing(tokenizer):
    # Real text streamed before the first header: once framing arrives the
    # segmenter drops the pre-header text, so the streaming cursor must
    # re-anchor (emit the framed body as a fresh delta) instead of wedging
    # on the unframed prefix and losing the answer.
    text = "Hello there<|start|>assistant to=user<|message|>Answer<|eot|>"
    for with_tool_parser in (True, False):
        reasoning, content, tools = drive_tokenwise(
            tokenizer, text, with_tool_parser=with_tool_parser
        )
        assert reasoning == ""
        assert content == "Hello thereAnswer"
        assert tools == []


def test_unframed_stream_strips_trailing_end_marker_runs(tokenizer):
    # A partial marker can hide a complete one at the tail ("ok<|eom|><"):
    # the holdback must strip the whole run, not just one layer.
    for text, expected in (
        ("The answer is 42.<|eom|> ", "The answer is 42."),
        ("ok<|eom|><|eom|>", "ok"),
    ):
        for with_tool_parser in (True, False):
            reasoning, content, tools = drive_tokenwise(
                tokenizer, text, with_tool_parser=with_tool_parser
            )
            assert reasoning == ""
            assert content == expected
            assert tools == []


def test_headerless_complete_atem_salvaged_at_finish(tokenizer):
    # A derailed ATEM block with no channel framing streams out as text (the
    # markup is indistinguishable from quoted text until it completes), but a
    # complete call inside it is still salvaged at finish -- matching the
    # non-streaming fallback, which extracts the same call.
    text = '<atem:invoke name="weather.get"></atem:invoke>'
    reasoning, content, tools = drive_tokenwise(tokenizer, text)
    assert reasoning == ""
    assert content == text
    assert tool_names(tools) == ["weather.get"]


def test_headerless_atem_with_stray_start_marker_salvaged_once(tokenizer):
    # A stray <|start|> activates the framed path before any channel
    # completes: the headerless call must not fire mid-stream (a later header
    # could retroactively reclassify the markup as quoted text) and must
    # salvage exactly once at finish, matching non-streaming.
    text = '<|start|><atem:invoke name="weather.get"></atem:invoke>'
    _reasoning, _content, tools = drive_tokenwise(tokenizer, text)
    assert tool_names(tools) == ["weather.get"]


def test_headerless_atem_before_a_completed_header_is_quoted_text(tokenizer):
    # Once a channel header completes, markup before it is pre-header text:
    # no call fires mid-stream or at finish, matching non-streaming.
    text = '<|start|><atem:invoke name="weather.get"></atem:invoke><|message|>x<|eot|>'
    _reasoning, content, tools = drive_tokenwise(tokenizer, text)
    assert content == "x"
    assert tool_names(tools) == []


def test_salvage_respects_tool_choice_none(tokenizer):
    # tool_choice="none" suppresses the finish-time salvage too, matching the
    # streaming and non-streaming paths.
    request = SimpleNamespace(tools=None, tool_choice="none", include_reasoning=True)
    _reasoning, _content, tools = drive_tokenwise(
        tokenizer, '<atem:invoke name="weather.get"></atem:invoke>', request=request
    )
    assert tool_names(tools) == []


def test_unframed_prefix_then_open_untagged_channel_flushes(tokenizer):
    # The untagged body is withheld the whole stream (it may still grow ATEM
    # markup); it must still flush at finish even though the stream began
    # unframed. The held-back whitespace before the header drops with the
    # pre-header text.
    text = "prose <|start|>assistant<|message|>the answer"
    for with_tool_parser in (True, False):
        reasoning, content, tools = drive_tokenwise(
            tokenizer, text, with_tool_parser=with_tool_parser
        )
        assert reasoning == ""
        assert content == "prosethe answer"
        assert tools == []
        # Same when the prose and the framing arrive in a single delta.
        _r, one_shot, _t = drive(tokenizer, [text], with_tool_parser=with_tool_parser)
        assert one_shot == "prosethe answer"


def test_unframed_framing_flip_is_chunking_independent(tokenizer):
    # A framed body whose prefix coincides with the already-emitted unframed
    # text must stream identically under any chunking: at the flip the
    # pre-header region is flushed and the cursor re-anchored.
    text = "the<|start|>assistant to=user<|message|>the answer<|eot|>"
    _r1, charwise, _t1 = drive_tokenwise(tokenizer, text)
    _r2, chunked, _t2 = drive(tokenizer, ["the", text[3:]])
    assert charwise == chunked == "thethe answer"


def test_flip_delta_flushes_held_pre_header_text(tokenizer):
    # A delta spanning pre-header text and the framing start (MTP/spec-decode
    # batches several tokens per delta) must not lose the held-back prose.
    text = "xy<|start|>assistant to=user<|message|>ans<|eot|>"
    _r, charwise, _t = drive_tokenwise(tokenizer, text)
    assert charwise == "xyans"
    for chunks in ([text], ["x", text[1:]], ["xy", text[2:]]):
        _r, content, _t = drive(tokenizer, chunks)
        assert content == "xyans"


def test_flip_flushes_fragment_tailed_pre_header_text(tokenizer):
    # Pre-header prose ending in a ` to=…`/partial-marker fragment is frozen
    # at the flip: it flushes verbatim (minus trailing whitespace), identically
    # under any chunking. (The `to=calc`/`to=b` channels are tool channels:
    # their bodies are not content.)
    cases = [
        ("x to to=user<|message|>y<|eot|>", "x toy"),
        ("x to=a to=user<|message|>c<|eot|>", "x to=ac"),
        ("<|eo to=calc<|message|>body<|eot|>", "<|eo"),
    ]
    for text, expected in cases:
        assert drive_tokenwise(tokenizer, text) == ("", expected, [])
        assert drive(tokenizer, [text]) == ("", expected, [])


def test_flip_flush_matches_streaming_whitespace_class(tokenizer):
    # The frozen pre-header region flushes with the streaming path's exact
    # whitespace/marker rules: U+001C-U+001F are NOT whitespace there, and a
    # trailing end marker is framing and drops.
    for text, expected in [
        ("a\x1c to=user<|message|>body<|eot|>", "a\x1cbody"),
        ("abc<|eom|><|start|>assistant to=user<|message|>y<|eot|>", "abcy"),
    ]:
        assert drive_tokenwise(tokenizer, text) == ("", expected, [])
        assert drive(tokenizer, [text]) == ("", expected, [])


def test_unframed_stream_recovers_quoted_framing_at_finish(tokenizer):
    # Quoted framing in unframed (e.g. grammar-shaped JSON) text stalls the
    # stream while it might be a header, then flushes whole at finish.
    text = '{"doc": "use <|start|>assistant to begin", "ok": true}'
    for with_tool_parser in (True, False):
        reasoning, content, tools = drive_tokenwise(
            tokenizer, text, with_tool_parser=with_tool_parser
        )
        assert reasoning == ""
        assert content == text
        assert tools == []


def test_unframed_stream_flushes_held_tails_at_finish(tokenizer):
    # The unframed fallback holds back ` to=…` fragments and trailing
    # whitespace; they flush at finish.
    reasoning, content, tools = drive_tokenwise(tokenizer, "The answer is 42. to=")
    assert reasoning == ""
    assert content == "The answer is 42. to="
    assert tools == []


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


def test_untagged_body_keeps_atem_like_words(tokenizer):
    # Only a real invoke opener strips an untagged body: "<atem:invokeful"
    # is ordinary text (the strip shares the boundary pattern's word edge).
    reasoning, content, tools = drive_tokenwise(
        tokenizer, "<|start|>assistant<|message|>x <atem:invokeful y<|eot|>"
    )
    assert reasoning == ""
    assert content == "x <atem:invokeful y"
    assert tools == []


def test_finish_flushes_held_back_fragments_but_not_partial_markers(tokenizer):
    # Truncated mid-answer: the ` to=…` fragment is real text and flushes.
    reasoning, content, _tools = drive(
        tokenizer,
        [" to=user<|message|>the answer is about to"],
        with_tool_parser=False,
    )
    assert content == "the answer is about to"
    assert reasoning == ""

    # Truncated mid-marker: the partial framing stays dropped.
    reasoning, content, _tools = drive(
        tokenizer,
        [" to=user<|message|>the answer<|eo"],
        with_tool_parser=False,
    )
    assert content == "the answer"
    assert_no_framing(content)

    # Same on the reasoning side.
    reasoning, content, _tools = drive(
        tokenizer,
        [" to=self<|message|>thinking hard<|eo"],
        with_tool_parser=False,
    )
    assert reasoning == "thinking hard"
    assert_no_framing(reasoning)


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
            (
                " to=self<|message|>think"
                "<|start|>assistant<|message|>The answer is 42.<|eot|>"
            )
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


def test_responses_stray_response_format_rejected_with_tools():
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
            "response_format": {"type": "json_object"},
        }
    )
    parser = ParserManager.get_parser(
        reasoning_parser_name="muse_glimmer",
        tool_parser_name="muse_glimmer",
        enable_auto_tools=True,
    )(CharTokenizer())
    with pytest.raises(
        VLLMValidationError,
        match="cannot be combined",
    ) as exc_info:
        parser.adjust_request(request)
    assert exc_info.value.parameter == "response_format"


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
