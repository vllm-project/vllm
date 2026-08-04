# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.tool_parsers.iquest_coder_tool_parser import IquestCoderToolParser
from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

# iquest_coder shares the qwen3_coder XML tool-call format and tokenizer.
MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"


@pytest.fixture(scope="module")
def iquest_tokenizer():
    from vllm.tokenizers import get_tokenizer

    return get_tokenizer(tokenizer_name=MODEL)


@pytest.fixture
def iquest_tool_parser(iquest_tokenizer):
    return IquestCoderToolParser(iquest_tokenizer)


@pytest.fixture
def sample_tools():
    return [
        ChatCompletionToolsParam(
            type="function",
            function={
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "The city name"},
                        "state": {"type": "string", "description": "The state code"},
                        "unit": {"type": "string", "enum": ["fahrenheit", "celsius"]},
                    },
                    "required": ["city", "state"],
                },
            },
        ),
    ]


def test_is_subclass_of_qwen3_coder():
    assert issubclass(IquestCoderToolParser, Qwen3CoderToolParser)


def _feed_deltas(parser, deltas, request):
    """Drive the streaming parser with an explicit list of delta chunks.

    This does not go through the tokenizer, so the test can control exactly
    how text is chunked across deltas (e.g. fusing separator whitespace with
    the following <tool_call> start token).
    """
    previous_text = ""
    for delta_text in deltas:
        current_text = previous_text + delta_text
        delta_message = parser.extract_tool_calls_streaming(
            previous_text,
            current_text,
            delta_text,
            [],
            [],
            [],
            request=request,
        )
        if delta_message:
            yield delta_message
        previous_text = current_text


def test_streaming_no_content_between_parallel_tools(iquest_tool_parser, sample_tools):
    """Whitespace between parallel tool calls must not leak as content.

    When the separator "\\n" between </tool_call> and the next <tool_call>
    is fused into a single delta with the start token, the base qwen3_coder
    parser emits that whitespace as a stray content block sitting between the
    two tool_use blocks. IquestCoderToolParser drops it.
    """
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=sample_tools)

    # Deltas crafted so the inter-call "\n" arrives fused with the next
    # <tool_call> start token, mimicking a real tokenizer's "\n<tool_call>".
    deltas = [
        "<tool_call>\n",
        "<function=get_current_weather>\n",
        "<parameter=city>\nDallas\n</parameter>\n",
        "<parameter=state>\nTX\n</parameter>\n",
        "<parameter=unit>\nfahrenheit\n</parameter>\n",
        "</function>\n",
        "</tool_call>",
        "\n<tool_call>\n",  # separator whitespace fused with start token
        "<function=get_current_weather>\n",
        "<parameter=city>\nOrlando\n</parameter>\n",
        "<parameter=state>\nFL\n</parameter>\n",
        "<parameter=unit>\ncelsius\n</parameter>\n",
        "</function>\n",
        "</tool_call>",
    ]

    other_content = ""
    tool_indices = set()

    for delta_message in _feed_deltas(iquest_tool_parser, deltas, request):
        assert not delta_message.role
        if delta_message.content:
            other_content += delta_message.content
        if delta_message.tool_calls:
            for tool_call in delta_message.tool_calls:
                tool_indices.add(tool_call.index)

    # No stray whitespace content block should have been emitted.
    assert other_content == ""
    # Both parallel tool calls should have been streamed.
    assert tool_indices == {0, 1}


def test_streaming_preserves_real_leading_content(iquest_tool_parser, sample_tools):
    """Genuine text before the first tool call is still streamed."""
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=sample_tools)

    deltas = [
        "Let me check the weather.",
        "<tool_call>\n",
        "<function=get_current_weather>\n",
        "<parameter=city>\nDallas\n</parameter>\n",
        "<parameter=state>\nTX\n</parameter>\n",
        "<parameter=unit>\nfahrenheit\n</parameter>\n",
        "</function>\n",
        "</tool_call>",
    ]

    other_content = ""
    tool_indices = set()

    for delta_message in _feed_deltas(iquest_tool_parser, deltas, request):
        if delta_message.content:
            other_content += delta_message.content
        if delta_message.tool_calls:
            for tool_call in delta_message.tool_calls:
                tool_indices.add(tool_call.index)

    assert other_content == "Let me check the weather."
    assert tool_indices == {0}


def test_non_streaming_parallel_tools(iquest_tool_parser, sample_tools):
    """Non-streaming extraction is inherited unchanged from qwen3_coder."""
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=sample_tools)
    model_output = (
        "<tool_call>\n<function=get_current_weather>\n"
        "<parameter=city>\nDallas\n</parameter>\n"
        "<parameter=state>\nTX\n</parameter>\n"
        "<parameter=unit>\nfahrenheit\n</parameter>\n"
        "</function>\n</tool_call>\n"
        "<tool_call>\n<function=get_current_weather>\n"
        "<parameter=city>\nOrlando\n</parameter>\n"
        "<parameter=state>\nFL\n</parameter>\n"
        "<parameter=unit>\ncelsius\n</parameter>\n"
        "</function>\n</tool_call>"
    )

    extracted = iquest_tool_parser.extract_tool_calls(model_output, request=request)

    assert extracted.tools_called
    assert len(extracted.tool_calls) == 2
    assert json.loads(extracted.tool_calls[0].function.arguments) == {
        "city": "Dallas",
        "state": "TX",
        "unit": "fahrenheit",
    }
    assert json.loads(extracted.tool_calls[1].function.arguments) == {
        "city": "Orlando",
        "state": "FL",
        "unit": "celsius",
    }


# Argument type conversion across tool shapes.
#
# The argument-config lookup drives type conversion (e.g. an integer parameter
# is JSON-decoded rather than kept as a string). The base qwen3_coder parser
# only understands the Chat Completions shape (`.function.parameters`); on the
# /v1/responses path tools are Responses API FunctionTool / NamespaceTool with
# `name`/`parameters` on the tool itself. iquest_coder overrides
# `_get_arguments_config` to resolve all three shapes.

_TYPED_OUTPUT = (
    "<tool_call>\n<function={name}>\n"
    "<parameter=command>\napt-get update\n</parameter>\n"
    "<parameter=timeout_ms>\n300000\n</parameter>\n"
    "</function>\n</tool_call>"
)

_TYPED_PROPERTIES = {
    "type": "object",
    "properties": {
        "command": {"type": "string"},
        "timeout_ms": {"type": "integer"},
    },
}


def test_type_conversion_responses_function_tool(iquest_tool_parser):
    """Responses API FunctionTool: integer param is decoded, not left a string."""
    from types import SimpleNamespace

    from openai.types.responses import FunctionTool

    ft = FunctionTool(
        type="function",
        name="shell_command",
        description="run a shell command",
        strict=False,
        parameters=_TYPED_PROPERTIES,
    )
    request = SimpleNamespace(tools=[ft], tool_choice="auto")
    extracted = iquest_tool_parser.extract_tool_calls(
        _TYPED_OUTPUT.format(name="shell_command"), request=request
    )

    args = json.loads(extracted.tool_calls[0].function.arguments)
    assert args["command"] == "apt-get update"
    assert args["timeout_ms"] == 300000
    assert isinstance(args["timeout_ms"], int)


def test_type_conversion_responses_namespace_tool(iquest_tool_parser):
    """Namespace children resolve via the flattened `namespace__name`."""
    from types import SimpleNamespace

    from openai.types.responses import NamespaceTool

    ns = NamespaceTool(
        type="namespace",
        name="agents",
        description="agent bundle",
        tools=[
            {
                "type": "function",
                "name": "shell_command",
                "description": "run a shell command",
                "parameters": _TYPED_PROPERTIES,
            }
        ],
    )
    request = SimpleNamespace(tools=[ns], tool_choice="auto")
    extracted = iquest_tool_parser.extract_tool_calls(
        _TYPED_OUTPUT.format(name="agents__shell_command"), request=request
    )

    args = json.loads(extracted.tool_calls[0].function.arguments)
    assert args["timeout_ms"] == 300000
    assert isinstance(args["timeout_ms"], int)


def test_type_conversion_chat_completion_tool_fallback(iquest_tool_parser):
    """Chat Completions shape still works via the base-class lookup."""
    tool = ChatCompletionToolsParam(
        type="function",
        function={"name": "shell_command", "parameters": _TYPED_PROPERTIES},
    )
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=[tool])
    extracted = iquest_tool_parser.extract_tool_calls(
        _TYPED_OUTPUT.format(name="shell_command"), request=request
    )

    args = json.loads(extracted.tool_calls[0].function.arguments)
    assert args["timeout_ms"] == 300000
    assert isinstance(args["timeout_ms"], int)


def test_string_param_not_decoded_responses_shape(iquest_tool_parser):
    """A numeric-looking value for a string param stays a string."""
    from types import SimpleNamespace

    from openai.types.responses import FunctionTool

    ft = FunctionTool(
        type="function",
        name="shell_command",
        description="s",
        strict=False,
        parameters={
            "type": "object",
            "properties": {"command": {"type": "string"}},
        },
    )
    request = SimpleNamespace(tools=[ft], tool_choice="auto")
    model_output = (
        "<tool_call>\n<function=shell_command>\n"
        "<parameter=command>\n12345\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    extracted = iquest_tool_parser.extract_tool_calls(model_output, request=request)

    args = json.loads(extracted.tool_calls[0].function.arguments)
    assert args["command"] == "12345"
    assert isinstance(args["command"], str)


def _typed_tool():
    from openai.types.responses import FunctionTool

    return FunctionTool(
        type="function",
        name="do_it",
        description="every interesting param type",
        strict=False,
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "count": {"type": "integer"},
                "ratio": {"type": "number"},
                "flag": {"type": "boolean"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "opts": {"type": "object"},
                "numeric_str": {"type": "string"},
            },
        },
    )


_ALL_TYPES_OUTPUT = (
    "<tool_call>\n<function=do_it>\n"
    "<parameter=text>\nhello\n</parameter>\n"
    "<parameter=count>\n42\n</parameter>\n"
    "<parameter=ratio>\n3.14\n</parameter>\n"
    "<parameter=flag>\ntrue\n</parameter>\n"
    '<parameter=tags>\n["a", "b"]\n</parameter>\n'
    '<parameter=opts>\n{"k": 1, "nested": [1, 2, 3]}\n</parameter>\n'
    "<parameter=numeric_str>\n007\n</parameter>\n"
    "</function>\n</tool_call>"
)


def _assert_all_types(args):
    assert args["text"] == "hello" and isinstance(args["text"], str)
    assert args["count"] == 42 and isinstance(args["count"], int)
    assert args["ratio"] == 3.14 and isinstance(args["ratio"], float)
    assert args["flag"] is True
    assert args["tags"] == ["a", "b"] and isinstance(args["tags"], list)
    assert args["opts"] == {"k": 1, "nested": [1, 2, 3]}
    assert isinstance(args["opts"], dict)
    # A string-typed parameter keeps its (numeric-looking) value verbatim.
    assert args["numeric_str"] == "007" and isinstance(args["numeric_str"], str)


def test_all_param_types_preserved_non_streaming(iquest_tool_parser):
    """int/float/bool/array/object types survive /v1/responses (FunctionTool)."""
    from types import SimpleNamespace

    request = SimpleNamespace(tools=[_typed_tool()], tool_choice="auto")
    extracted = iquest_tool_parser.extract_tool_calls(
        _ALL_TYPES_OUTPUT, request=request
    )
    _assert_all_types(json.loads(extracted.tool_calls[0].function.arguments))


def test_all_param_types_preserved_streaming(iquest_tokenizer):
    """Same type preservation on the streaming path."""
    from types import SimpleNamespace

    parser = IquestCoderToolParser(iquest_tokenizer)
    request = SimpleNamespace(tools=[_typed_tool()], tool_choice="auto")

    ids = iquest_tokenizer.encode(_ALL_TYPES_OUTPUT, add_special_tokens=False)
    prev_text = ""
    prev_ids: list[int] = []
    streamed_args = ""
    for tid in ids:
        cur_ids = prev_ids + [tid]
        cur_text = iquest_tokenizer.decode(cur_ids, skip_special_tokens=False)
        delta = cur_text[len(prev_text) :]
        dm = parser.extract_tool_calls_streaming(
            prev_text, cur_text, delta, prev_ids, cur_ids, [tid], request
        )
        if dm and dm.tool_calls:
            for tc in dm.tool_calls:
                if tc.function and tc.function.arguments:
                    streamed_args += tc.function.arguments
        prev_text = cur_text
        prev_ids = cur_ids

    _assert_all_types(json.loads(streamed_args))


# streamed_args_for_tool bookkeeping.
#
# The serving layer flushes any unstreamed trailing tool arguments on the
# final chunk by indexing streamed_args_for_tool[len(prev_tool_call_arr) - 1].
# The base qwen3_coder parser emits argument deltas but never records them
# there, so the list stays empty and that index raises IndexError. This is
# only hit when a tool call is truncated mid-arguments (e.g. max_tokens before
# </function>), because the final delta still carries non-None arguments.
# IquestCoderToolParser mirrors every argument fragment into the list.


def test_streamed_args_tracks_emitted_arguments(iquest_tool_parser, sample_tools):
    """streamed_args_for_tool accumulates exactly what was streamed."""
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=sample_tools)

    deltas = [
        "<tool_call>\n",
        "<function=get_current_weather>\n",
        "<parameter=city>\nDallas\n</parameter>\n",
        "<parameter=state>\nTX\n</parameter>\n",
        "<parameter=unit>\nfahrenheit\n</parameter>\n",
        "</function>\n",
        "</tool_call>",
    ]

    streamed = ""
    for delta_message in _feed_deltas(iquest_tool_parser, deltas, request):
        for tc in delta_message.tool_calls or []:
            if tc.function and tc.function.arguments:
                streamed += tc.function.arguments

    # The parser mirrors emitted arguments verbatim into streamed_args_for_tool
    # so the serving layer's "unstreamed args" flush has an accurate baseline.
    assert iquest_tool_parser.streamed_args_for_tool == [streamed]
    # A single tool call produces a single slot.
    assert len(iquest_tool_parser.streamed_args_for_tool) == 1


def test_truncated_tool_call_does_not_desync_streamed_args(
    iquest_tool_parser, sample_tools
):
    """A tool call truncated mid-arguments keeps the two lists aligned.

    Reproduces the IndexError: the model starts a tool call (so the header is
    detected and prev_tool_call_arr gets an entry) but the stream ends before
    </function>. The serving layer then indexes
    streamed_args_for_tool[len(prev_tool_call_arr) - 1]; that must be in range.
    """
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=sample_tools)

    # No closing </parameter> / </function> / </tool_call>: truncated output.
    deltas = [
        "<tool_call>\n",
        "<function=get_current_weather>\n",
        "<parameter=city>\nDall",
    ]

    for _ in _feed_deltas(iquest_tool_parser, deltas, request):
        pass

    # prev_tool_call_arr has the (partial) call; streamed_args_for_tool must be
    # at least as long so the serving-layer index is valid.
    assert len(iquest_tool_parser.prev_tool_call_arr) >= 1
    index = len(iquest_tool_parser.prev_tool_call_arr) - 1
    assert index < len(iquest_tool_parser.streamed_args_for_tool)
    # This is the exact access that used to raise IndexError.
    iquest_tool_parser.streamed_args_for_tool[index]


def test_reset_clears_streamed_args(iquest_tool_parser, sample_tools):
    """A fresh stream (previous_text == "") clears prior streamed args."""
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=sample_tools)

    first = [
        "<tool_call>\n",
        "<function=get_current_weather>\n",
        "<parameter=city>\nDallas\n</parameter>\n",
        "</function>\n",
        "</tool_call>",
    ]
    for _ in _feed_deltas(iquest_tool_parser, first, request):
        pass
    assert iquest_tool_parser.streamed_args_for_tool  # populated

    # Starting a new stream resets state (previous_text == "" on first delta).
    for _ in _feed_deltas(iquest_tool_parser, first, request):
        pass
    # Exactly one tool's worth of args, not accumulated across both streams.
    assert len(iquest_tool_parser.streamed_args_for_tool) == 1


# --------------------------------------------------------------------------
# Chunk-size invariance.
#
# Speculative decoding (MTP) emits several tokens per step, so one streaming
# delta can carry a whole tool call. The inherited qwen3_coder implementation
# reacted to delta_text and tracked flags across calls, which made its output
# depend on how the text happened to be chunked: with
# num_speculative_tokens=2 arguments came out truncated -- even a
# single-parameter call arrived as '{"file_path": "/x/main.py"' with no closing
# brace, which clients reject as __unparsedToolInput. These tests pin the
# invariant that the emitted arguments depend only on the text, never on the
# delta boundaries.
# --------------------------------------------------------------------------

_SINGLE_PARAM_CALL = (
    "<tool_call>\n"
    "<function=get_current_weather>\n"
    "<parameter=city>\nDallas\n</parameter>\n"
    "</function>\n"
    "</tool_call>"
)

_TWO_PARAM_CALL = (
    "<tool_call>\n"
    "<function=get_current_weather>\n"
    "<parameter=city>\nDallas\n</parameter>\n"
    "<parameter=state>\nTX\n</parameter>\n"
    "</function>\n"
    "</tool_call>"
)


def _chunk(text: str, size: int | None) -> list[str]:
    """Split text into fixed-size chunks; size=None means one giant chunk."""
    if size is None:
        return [text]
    return [text[i : i + size] for i in range(0, len(text), size)]


def _collect_streamed(parser, text: str, size: int | None, request):
    """Feed `text` in `size`-sized chunks, returning (args_by_index, content).

    Also asserts the wire invariant that a single delta never carries two
    entries for the same tool index -- clients keep only one of them, so
    emitting the header and the first argument fragment separately under one
    index silently drops the opening brace.
    """
    args: dict[int, str] = {}
    names: dict[int, str] = {}
    content = ""
    for delta_message in _feed_deltas(parser, _chunk(text, size), request):
        if delta_message.content:
            content += delta_message.content
        seen_indices = set()
        for tool_call in delta_message.tool_calls or []:
            assert tool_call.index not in seen_indices, (
                f"two entries for tool index {tool_call.index} in one delta"
            )
            seen_indices.add(tool_call.index)
            if tool_call.function and tool_call.function.name:
                names[tool_call.index] = tool_call.function.name
            if tool_call.function and tool_call.function.arguments:
                args.setdefault(tool_call.index, "")
                args[tool_call.index] += tool_call.function.arguments
    return args, names, content


@pytest.mark.parametrize("size", [None, 1, 3, 7, 40, 250])
def test_streaming_chunk_size_invariance_single_call(
    iquest_tokenizer, sample_tools, size
):
    """One call's arguments are identical no matter how the text is chunked."""
    parser = IquestCoderToolParser(iquest_tokenizer)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=sample_tools)

    args, names, content = _collect_streamed(parser, _TWO_PARAM_CALL, size, request)

    assert names == {0: "get_current_weather"}
    assert json.loads(args[0]) == {"city": "Dallas", "state": "TX"}
    assert content == ""


@pytest.mark.parametrize("size", [None, 1, 3, 7, 40, 250])
def test_streaming_single_param_call_is_closed(iquest_tokenizer, sample_tools, size):
    """A one-parameter call must still emit its closing brace.

    Regression: this is the exact shape that broke under MTP -- the arguments
    arrived as '{"city": "Dallas"' and clients reported
    "input that could not be parsed as JSON".
    """
    parser = IquestCoderToolParser(iquest_tokenizer)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=sample_tools)

    args, _, _ = _collect_streamed(parser, _SINGLE_PARAM_CALL, size, request)

    assert args[0].endswith("}"), f"unterminated arguments: {args[0]!r}"
    assert json.loads(args[0]) == {"city": "Dallas"}


@pytest.mark.parametrize("size", [None, 1, 5, 60, 250])
def test_streaming_repeated_same_name_calls(iquest_tokenizer, sample_tools, size):
    """Three calls to the SAME tool must stay three distinct calls.

    prev_tool_call_arr used to be de-duplicated by function name, which
    collapsed repeated calls into one entry; the serving layer then flushed the
    last call's arguments onto the first ('{"city": "A"}{"city": "C"}') and left
    the last one unterminated.
    """
    parser = IquestCoderToolParser(iquest_tokenizer)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=sample_tools)

    text = "\n".join(
        "<tool_call>\n"
        "<function=get_current_weather>\n"
        f"<parameter=city>\n{city}\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
        for city in ("Dallas", "Orlando", "Boston")
    )

    args, _, content = _collect_streamed(parser, text, size, request)

    assert sorted(args) == [0, 1, 2]
    assert [json.loads(args[i])["city"] for i in (0, 1, 2)] == [
        "Dallas",
        "Orlando",
        "Boston",
    ]
    assert content == ""


@pytest.mark.parametrize("size", [None, 1, 3, 40])
def test_streaming_partial_start_token_not_leaked_as_content(
    iquest_tokenizer, sample_tools, size
):
    """A partial "<tool_call" prefix must be buffered, not emitted as content."""
    parser = IquestCoderToolParser(iquest_tokenizer)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=sample_tools)

    text = "Let me check the weather.\n" + _SINGLE_PARAM_CALL
    _, _, content = _collect_streamed(parser, text, size, request)

    assert "<tool_call" not in content
    assert content.strip() == "Let me check the weather."


@pytest.mark.parametrize("size", [None, 1, 4, 60])
def test_streaming_plain_content_without_tool_calls(
    iquest_tokenizer, sample_tools, size
):
    """Text with no tool call streams through unchanged at any chunk size."""
    parser = IquestCoderToolParser(iquest_tokenizer)
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=sample_tools)

    text = "No tool needed here, just a plain answer."
    args, _, content = _collect_streamed(parser, text, size, request)

    assert args == {}
    assert content == text


def test_adjust_request_keeps_special_tokens_with_tools(
    iquest_tool_parser, sample_tools
):
    """Tool requests must decode with special tokens intact.

    ``<tool_call>`` / ``</tool_call>`` are *special* tokens on the iQuest
    tokenizers, so under the default ``skip_special_tokens=True`` they are
    stripped before the parser sees the text and no tool call is recognised --
    the whole ``<function=...>`` body streams out as plain content. This lives
    on IquestCoderToolParser rather than the shared qwen3_coder base so the
    base parser's behaviour is untouched.
    """
    request = ChatCompletionRequest(model=MODEL, messages=[], tools=sample_tools)
    adjusted = iquest_tool_parser.adjust_request(request)
    assert adjusted.skip_special_tokens is False


def test_adjust_request_untouched_without_tools(iquest_tool_parser):
    """Without tools there is nothing to parse, so leave the request alone."""
    request = ChatCompletionRequest(model=MODEL, messages=[])
    before = request.skip_special_tokens
    adjusted = iquest_tool_parser.adjust_request(request)
    assert adjusted.skip_special_tokens == before


def test_adjust_request_untouched_when_tool_choice_none(
    iquest_tool_parser, sample_tools
):
    """tool_choice="none" means the model will not emit tool calls."""
    request = ChatCompletionRequest(
        model=MODEL, messages=[], tools=sample_tools, tool_choice="none"
    )
    before = request.skip_special_tokens
    adjusted = iquest_tool_parser.adjust_request(request)
    assert adjusted.skip_special_tokens == before


def test_qwen3_coder_base_left_unmodified(iquest_tokenizer):
    """The shared base parser must not inherit the iQuest-only adjustment."""
    base = Qwen3CoderToolParser(iquest_tokenizer)
    assert "adjust_request" not in Qwen3CoderToolParser.__dict__
    assert "adjust_request" in IquestCoderToolParser.__dict__
    # And the base still does not populate streamed_args_for_tool itself.
    assert base.streamed_args_for_tool == []
