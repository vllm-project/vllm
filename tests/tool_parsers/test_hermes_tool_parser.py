# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import ToolParser
from vllm.tool_parsers.granite4_tool_parser import Granite4ToolParser
from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser

CONFIGS = {
    "llama": {
        "tool_parser": Hermes2ProToolParser,
    },
    "granite4": {
        "tool_parser": Granite4ToolParser,
    },
}


@pytest.fixture
def qwen_tokenizer() -> TokenizerLike:
    from vllm.tokenizers import get_tokenizer

    return get_tokenizer("Qwen/Qwen3-32B")


@pytest.fixture(params=CONFIGS.keys())
def hermes_parser(request, qwen_tokenizer: TokenizerLike) -> ToolParser:
    config = CONFIGS[request.param]
    return config["tool_parser"](qwen_tokenizer)


@pytest.fixture
def any_chat_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        seed=42,
        model="Qwen/Qwen3-32B",
        messages=[],
    )


def test_hermes_parser_streaming_just_forward_text(
    qwen_tokenizer: TokenizerLike,
    hermes_parser: ToolParser,
    any_chat_request: ChatCompletionRequest,
) -> None:
    text = """This is some prior text that has nothing to do with tool calling."""
    tokens = qwen_tokenizer.encode(text)
    previous_text = ""
    delta_messages = []
    for token in tokens:
        delta_text = qwen_tokenizer.decode([token])
        current_text = previous_text + delta_text
        delta = hermes_parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=any_chat_request,
        )
        previous_text = current_text
        delta_messages.append(delta)

    for delta in delta_messages:
        assert delta is not None
        assert not delta.tool_calls

    print(delta_messages)
    assert "".join([delta.content for delta in delta_messages]) == text


def test_hermes_parser_streaming_failure_case_bug_19056(
    qwen_tokenizer: TokenizerLike,
    hermes_parser: ToolParser,
    any_chat_request: ChatCompletionRequest,
) -> None:
    text = """<tool_call>
{"name": "final_answer", "arguments": {"trigger": true}}
</tool_call>"""
    tokens = qwen_tokenizer.encode(text)
    previous_text = ""
    delta_messages = []
    for token in tokens:
        text = qwen_tokenizer.decode([token])
        current_text = previous_text + text
        delta = hermes_parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=any_chat_request,
        )
        previous_text = current_text
        if delta is not None:
            delta_messages.append(delta)

    assert delta_messages[0].tool_calls[0].function.name == "final_answer"
    tool_call_args = "".join(
        delta.tool_calls[0].function.arguments or "" for delta in delta_messages
    )
    assert tool_call_args == '{"trigger": true}'


def test_hermes_parser_streaming(
    qwen_tokenizer: TokenizerLike,
    hermes_parser: ToolParser,
    any_chat_request: ChatCompletionRequest,
) -> None:
    text = '<tool_call>\
{"name": "get_current_temperature",\
"arguments": {"location":\
"San Francisco, California, United States", "unit": "celsius"}}\
</tool_call>'

    tokens = qwen_tokenizer.encode(text)
    previous_text = ""
    delta_messages = []
    for token in tokens:
        text = qwen_tokenizer.decode([token])
        current_text = previous_text + text
        delta = hermes_parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[],
            request=any_chat_request,
        )
        previous_text = current_text
        if delta is not None:
            delta_messages.append(delta)
    print(delta_messages)
    assert delta_messages[0].tool_calls[0].function.name == "get_current_temperature"
    # load to normalize whitespace
    tool_call_args = json.loads(
        "".join(
            delta.tool_calls[0].function.arguments or "" for delta in delta_messages
        )
    )
    assert tool_call_args == {
        "location": "San Francisco, California, United States",
        "unit": "celsius",
    }


def _simulate_streaming(
    tokenizer: TokenizerLike,
    parser: ToolParser,
    request: ChatCompletionRequest,
    text: str,
    stream_interval: int = 1,
) -> list:
    """Simulate streaming with a given stream_interval.

    Tokens are batched into chunks of `stream_interval` tokens,
    mimicking how the output processor delivers them.
    Returns a list of non-None DeltaMessages.
    """
    tokens = tokenizer.encode(text)
    previous_text = ""
    delta_messages = []
    for i in range(0, len(tokens), stream_interval):
        chunk_ids = tokens[i : i + stream_interval]
        delta_text = tokenizer.decode(chunk_ids)
        current_text = previous_text + delta_text
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=chunk_ids,
            request=request,
        )
        previous_text = current_text
        if delta is not None:
            delta_messages.append(delta)
    return delta_messages


@pytest.mark.parametrize("stream_interval", [2, 3, 5, 8])
def test_hermes_streaming_tool_call_with_stream_interval(
    qwen_tokenizer: TokenizerLike,
    any_chat_request: ChatCompletionRequest,
    stream_interval: int,
) -> None:
    """Tool call streaming must produce correct name + args at any interval."""
    text = (
        '<tool_call>{"name": "get_current_temperature", '
        '"arguments": {"location": "San Francisco", "unit": "celsius"}}'
        "</tool_call>"
    )
    parser = Hermes2ProToolParser(qwen_tokenizer)
    deltas = _simulate_streaming(
        qwen_tokenizer, parser, any_chat_request, text, stream_interval
    )

    # Flatten all DeltaToolCalls across all deltas.
    tool_deltas = [tc for d in deltas if d.tool_calls for tc in d.tool_calls]
    assert tool_deltas, "Expected at least one tool call delta"
    assert tool_deltas[0].function.name == "get_current_temperature"

    # Concatenated arguments must be valid JSON matching the original.
    args_str = "".join(tc.function.arguments or "" for tc in tool_deltas)
    assert json.loads(args_str) == {
        "location": "San Francisco",
        "unit": "celsius",
    }


@pytest.mark.parametrize("stream_interval", [2, 3, 5, 8])
def test_hermes_streaming_content_then_tool_call_with_stream_interval(
    qwen_tokenizer: TokenizerLike,
    any_chat_request: ChatCompletionRequest,
    stream_interval: int,
) -> None:
    """Content before a tool call must be fully streamed, then tool call."""
    text = (
        "Sure, let me check the weather."
        '<tool_call>{"name": "get_weather", '
        '"arguments": {"city": "NYC"}}</tool_call>'
    )
    parser = Hermes2ProToolParser(qwen_tokenizer)
    deltas = _simulate_streaming(
        qwen_tokenizer, parser, any_chat_request, text, stream_interval
    )

    content_deltas = [d for d in deltas if d.content]
    tool_deltas = [d for d in deltas if d.tool_calls]

    # Content must reconstruct the prefix.
    content_str = "".join(d.content for d in content_deltas)
    assert content_str == "Sure, let me check the weather."

    # Tool call must be correct.
    tool_calls = [tc for d in tool_deltas for tc in d.tool_calls]
    assert tool_calls[0].function.name == "get_weather"
    args_str = "".join(tc.function.arguments or "" for tc in tool_calls)
    assert json.loads(args_str) == {"city": "NYC"}


@pytest.mark.parametrize("stream_interval", [1, 2, 4])
def test_hermes_streaming_multiple_tool_calls_with_stream_interval(
    qwen_tokenizer: TokenizerLike,
    any_chat_request: ChatCompletionRequest,
    stream_interval: int,
) -> None:
    """Multiple sequential tool calls must each be streamed correctly."""
    text = (
        '<tool_call>{"name": "search", "arguments": {"q": "cats"}}</tool_call>'
        '<tool_call>{"name": "search", "arguments": {"q": "dogs"}}</tool_call>'
    )
    parser = Hermes2ProToolParser(qwen_tokenizer)
    deltas = _simulate_streaming(
        qwen_tokenizer, parser, any_chat_request, text, stream_interval
    )

    # Flatten all DeltaToolCalls across all deltas.
    all_tool_calls = [tc for d in deltas if d.tool_calls for tc in d.tool_calls]

    # Separate by tool index.
    tool0 = [tc for tc in all_tool_calls if tc.index == 0]
    tool1 = [tc for tc in all_tool_calls if tc.index == 1]

    assert tool0[0].function.name == "search"
    args0 = "".join(tc.function.arguments or "" for tc in tool0)
    assert json.loads(args0) == {"q": "cats"}

    assert tool1[0].function.name == "search"
    args1 = "".join(tc.function.arguments or "" for tc in tool1)
    assert json.loads(args1) == {"q": "dogs"}


@pytest.mark.parametrize("stream_interval", [2, 5])
def test_hermes_streaming_boolean_args_with_stream_interval(
    qwen_tokenizer: TokenizerLike,
    any_chat_request: ChatCompletionRequest,
    stream_interval: int,
) -> None:
    """Regression test for bug #19056 with stream_interval > 1."""
    text = (
        "<tool_call>\n"
        '{"name": "final_answer", "arguments": {"trigger": true}}\n'
        "</tool_call>"
    )
    parser = Hermes2ProToolParser(qwen_tokenizer)
    deltas = _simulate_streaming(
        qwen_tokenizer, parser, any_chat_request, text, stream_interval
    )

    tool_calls = [tc for d in deltas if d.tool_calls for tc in d.tool_calls]
    assert tool_calls[0].function.name == "final_answer"
    args_str = "".join(tc.function.arguments or "" for tc in tool_calls)
    assert json.loads(args_str) == {"trigger": True}


@pytest.mark.parametrize("stream_interval", [2, 3, 5])
def test_hermes_streaming_just_forward_text_with_stream_interval(
    qwen_tokenizer: TokenizerLike,
    any_chat_request: ChatCompletionRequest,
    stream_interval: int,
) -> None:
    """Plain text with no tool calls must be fully forwarded."""
    text = "This is plain text with no tool calling involved."
    parser = Hermes2ProToolParser(qwen_tokenizer)
    deltas = _simulate_streaming(
        qwen_tokenizer, parser, any_chat_request, text, stream_interval
    )

    for d in deltas:
        assert not d.tool_calls
    assert "".join(d.content for d in deltas) == text


def test_hermes_parser_non_streaming_no_tool_call(
    hermes_parser: ToolParser,
    any_chat_request: ChatCompletionRequest,
) -> None:
    text = """This is not a tool call."""
    tool_call = hermes_parser.extract_tool_calls(
        model_output=text,
        request=any_chat_request,
    )

    assert tool_call is not None
    assert not tool_call.tools_called


def test_hermes_parser_non_streaming_tool_call_between_tags(
    hermes_parser: ToolParser,
    any_chat_request: ChatCompletionRequest,
) -> None:
    text = """<tool_call>
{"name": "final_answer", "arguments": {"trigger": true}}
</tool_call>"""
    tool_call = hermes_parser.extract_tool_calls(
        model_output=text,
        request=any_chat_request,
    )

    assert tool_call is not None
    assert tool_call.tools_called
    assert tool_call.tool_calls[0].function.name == "final_answer"
    assert tool_call.tool_calls[0].function.arguments == '{"trigger": true}'


def test_hermes_parser_non_streaming_tool_call_until_eos(
    hermes_parser: ToolParser,
    any_chat_request: ChatCompletionRequest,
) -> None:
    if isinstance(hermes_parser, Granite4ToolParser):
        pytest.skip(reason="The Granite4 tool parser enforces a complete response")

    text = """<tool_call>
{"name": "final_answer", "arguments": {"trigger": true}}"""
    tool_call = hermes_parser.extract_tool_calls(
        model_output=text,
        request=any_chat_request,
    )

    assert tool_call is not None
    assert tool_call.tools_called
    assert tool_call.tool_calls[0].function.name == "final_answer"
    assert tool_call.tool_calls[0].function.arguments == '{"trigger": true}'


def test_hermes_parser_non_streaming_tool_call_invalid_json(
    hermes_parser: ToolParser,
    any_chat_request: ChatCompletionRequest,
) -> None:
    # Missing closing brace to trigger exception
    text = """<tool_call>
{"name": "final_answer", "arguments": {"trigger": true}"""
    tool_call = hermes_parser.extract_tool_calls(
        model_output=text,
        request=any_chat_request,
    )

    assert tool_call is not None
    assert not tool_call.tools_called


def test_hermes_streaming_content_and_tool_call_in_single_chunk(
    qwen_tokenizer: TokenizerLike,
    any_chat_request: ChatCompletionRequest,
) -> None:
    """Content + complete tool call in one chunk must both be emitted."""
    text = 'Hi!<tool_call>{"name": "f", "arguments": {"x": 1}}</tool_call>'
    # Use a stream_interval large enough to guarantee a single chunk.
    parser = Hermes2ProToolParser(qwen_tokenizer)
    deltas = _simulate_streaming(
        qwen_tokenizer,
        parser,
        any_chat_request,
        text,
        stream_interval=9999,
    )

    content_parts = [d.content for d in deltas if d.content]
    tool_parts = [tc for d in deltas if d.tool_calls for tc in d.tool_calls]

    assert "".join(content_parts) == "Hi!"
    assert tool_parts[0].function.name == "f"
    args_str = "".join(tc.function.arguments or "" for tc in tool_parts)
    assert json.loads(args_str) == {"x": 1}


# Reproduction from https://github.com/vllm-project/vllm/issues/45167
END_TAG_IN_STRING_TEXT = (
    "<tool_call>\n"
    '{"name": "edit_file", "arguments": '
    '{"content": "예시 태그: </tool_call> 포함 한국어 본문"}}\n'
    "</tool_call>"
)


def test_hermes_parser_non_streaming_end_tag_inside_string(
    qwen_tokenizer: TokenizerLike,
    any_chat_request: ChatCompletionRequest,
) -> None:
    """A literal </tool_call> in a string argument must not end the tool call."""
    parser = Hermes2ProToolParser(qwen_tokenizer)
    tool_call = parser.extract_tool_calls(
        model_output=END_TAG_IN_STRING_TEXT,
        request=any_chat_request,
    )

    assert tool_call.tools_called
    assert tool_call.content is None
    assert len(tool_call.tool_calls) == 1
    assert tool_call.tool_calls[0].function.name == "edit_file"
    args = json.loads(tool_call.tool_calls[0].function.arguments)
    assert args["content"] == "예시 태그: </tool_call> 포함 한국어 본문"


def test_hermes_parser_non_streaming_end_tag_inside_first_of_two_calls(
    qwen_tokenizer: TokenizerLike,
    any_chat_request: ChatCompletionRequest,
) -> None:
    """A quoted end tag must not swallow the tool call that follows it."""
    text = (
        '<tool_call>{"name": "first", '
        '"arguments": {"q": "a </tool_call> b"}}</tool_call>'
        '<tool_call>{"name": "second", "arguments": {"q": "c"}}</tool_call>'
    )
    parser = Hermes2ProToolParser(qwen_tokenizer)
    tool_call = parser.extract_tool_calls(
        model_output=text,
        request=any_chat_request,
    )

    assert tool_call.tools_called
    assert [tc.function.name for tc in tool_call.tool_calls] == ["first", "second"]
    assert json.loads(tool_call.tool_calls[0].function.arguments) == {
        "q": "a </tool_call> b"
    }
    assert json.loads(tool_call.tool_calls[1].function.arguments) == {"q": "c"}


def test_hermes_parser_non_streaming_escaped_quote_before_end_tag(
    qwen_tokenizer: TokenizerLike,
    any_chat_request: ChatCompletionRequest,
) -> None:
    """Escaped quotes must not confuse the string tracking of the scanner."""
    text = (
        "<tool_call>"
        '{"name": "say", "arguments": {"text": "say \\"hi\\" </tool_call> done"}}'
        "</tool_call>"
    )
    parser = Hermes2ProToolParser(qwen_tokenizer)
    tool_call = parser.extract_tool_calls(
        model_output=text,
        request=any_chat_request,
    )

    assert tool_call.tools_called
    assert tool_call.tool_calls[0].function.name == "say"
    args = json.loads(tool_call.tool_calls[0].function.arguments)
    assert args["text"] == 'say "hi" </tool_call> done'


def test_hermes_parser_streaming_end_tag_inside_string(
    qwen_tokenizer: TokenizerLike,
    any_chat_request: ChatCompletionRequest,
) -> None:
    """Streaming a quoted end tag must yield the non-streaming tool call."""
    parser = Hermes2ProToolParser(qwen_tokenizer)
    deltas = _simulate_streaming(
        qwen_tokenizer, parser, any_chat_request, END_TAG_IN_STRING_TEXT
    )

    tool_calls = [tc for d in deltas if d.tool_calls for tc in d.tool_calls]
    assert {tc.index for tc in tool_calls} == {0}
    assert tool_calls[0].function.name == "edit_file"
    args = json.loads("".join(tc.function.arguments or "" for tc in tool_calls))
    assert "</tool_call>" in args["content"]

    # Compare against the text the deltas reconstruct: decoding one token at a
    # time is lossy for multi-byte characters.
    streamed_text = "".join(
        qwen_tokenizer.decode([token])
        for token in qwen_tokenizer.encode(END_TAG_IN_STRING_TEXT)
    )
    expected = Hermes2ProToolParser(qwen_tokenizer).extract_tool_calls(
        model_output=streamed_text,
        request=any_chat_request,
    )
    assert args == json.loads(expected.tool_calls[0].function.arguments)


def test_hermes_parser_streaming_start_tag_inside_string(
    qwen_tokenizer: TokenizerLike,
    any_chat_request: ChatCompletionRequest,
) -> None:
    """A quoted start tag must not open a second tool call while streaming."""
    text = (
        "<tool_call>"
        '{"name": "edit_file", "arguments": '
        '{"content": "opens <tool_call> and closes </tool_call>"}}'
        "</tool_call>"
    )
    parser = Hermes2ProToolParser(qwen_tokenizer)
    deltas = _simulate_streaming(qwen_tokenizer, parser, any_chat_request, text)

    tool_calls = [tc for d in deltas if d.tool_calls for tc in d.tool_calls]
    assert {tc.index for tc in tool_calls} == {0}
    assert tool_calls[0].function.name == "edit_file"
    args = json.loads("".join(tc.function.arguments or "" for tc in tool_calls))
    assert args["content"] == "opens <tool_call> and closes </tool_call>"
