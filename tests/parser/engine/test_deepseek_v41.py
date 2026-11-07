# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from tests.parser.engine.replay_harness import (
    DUMMY_TOOLS,
    MockTokenizer,
    _test_request,
    collect_output,
    replay_streaming,
)
from vllm.parser.deepseek_v41 import deepseek_v41_config
from vllm.parser.parser_manager import ParserManager

CALLS = (
    '\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="get_weather">\n'
    '<｜DSML｜ parameter name="city" string="true">杭州</｜DSML｜ parameter>\n'
    '<｜DSML｜ parameter name="count" string="false">42</｜DSML｜ parameter>\n'
    '</｜DSML｜ invoke>\n<｜DSML｜ invoke name="add">\n'
    '<｜DSML｜ parameter name="x" string="false">1.5</｜DSML｜ parameter>\n'
    '<｜DSML｜ parameter name="y" string="false">2.25</｜DSML｜ parameter>\n'
    "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
)

# V4's ``tool_calls`` wrapper name with V4.1's spacing (#62051).
V4_NAMED_CALLS = CALLS.replace("｜DSML｜ calls>", "｜DSML｜ tool_calls>")


def tokenizer_for(text, special_calls):
    vocab = {"<think>": 50, "</think>": 51, "<｜User｜>": 54, "<｜Assistant｜>": 55}
    if special_calls:
        vocab |= {"<｜DSML｜ calls>": 52, "</｜DSML｜ calls>": 53}
    tokens: list[tuple[int, str]] = []
    while text:
        special = next((marker for marker in vocab if text.startswith(marker)), None)
        piece = special or text[0]
        tokens.append((vocab[piece] if special else 100 + len(tokens), piece))
        text = text[len(piece) :]
    return MockTokenizer(vocab, tokens), tokens


def parser_for(tokenizer, controls):
    cls = ParserManager.get_parser(
        tool_parser_name="deepseek_v41",
        reasoning_parser_name="deepseek_v41",
        enable_auto_tools=True,
    )
    assert cls is not None
    return cls(tokenizer, chat_template_kwargs=controls)


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("special_calls", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 7, 10000])
def test_registered_adapters_parse_parallel_calls_across_chunks(
    thinking,
    special_calls,
    chunk_size,
):
    text = ("Plan.</think>" if thinking else "") + "Checking." + CALLS
    tokenizer, tokens = tokenizer_for(text, special_calls)
    parser = parser_for(tokenizer, {"thinking": thinking})
    output = collect_output(
        replay_streaming(
            parser,
            tokens,
            chunk_size=chunk_size,
            finished_on_last=True,
            tools=DUMMY_TOOLS,
            prompt_token_ids=[50 if thinking else 51],
        )
    )
    assert output.reasoning == ("Plan." if thinking else "")
    assert output.content.strip() == "Checking."
    assert [call["name"] for call in output.tool_calls] == ["get_weather", "add"]
    assert [json.loads(call["arguments"]) for call in output.tool_calls] == [
        {"city": "杭州", "count": 42},
        {"x": 1.5, "y": 2.25},
    ]


@pytest.mark.parametrize("thinking", [False, True])
def test_registered_adapters_parse_complete_output(thinking):
    text = ("Plan.</think>" if thinking else "") + CALLS
    tokenizer, tokens = tokenizer_for(text, False)
    parser = parser_for(tokenizer, {"thinking": thinking})
    reasoning, content, calls = parser.parse(
        text,
        _test_request(tools=DUMMY_TOOLS),
        enable_auto_tools=True,
        model_output_token_ids=[tid for tid, _ in tokens],
    )
    assert (reasoning or "") == ("Plan." if thinking else "")
    assert not (content or "").strip()
    assert [call.name for call in calls] == ["get_weather", "add"]
    assert json.loads(calls[0].arguments) == {"city": "杭州", "count": 42}


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 7, 10000])
def test_v4_wrapper_name_opens_tool_block(thinking, chunk_size):
    text = ("Plan.</think>" if thinking else "") + "Checking." + V4_NAMED_CALLS
    tokenizer, tokens = tokenizer_for(text, False)
    output = collect_output(
        replay_streaming(
            parser_for(tokenizer, {"thinking": thinking}),
            tokens,
            chunk_size=chunk_size,
            finished_on_last=True,
            tools=DUMMY_TOOLS,
            prompt_token_ids=[50 if thinking else 51],
        )
    )
    reasoning, content, calls = parser_for(tokenizer, {"thinking": thinking}).parse(
        text,
        _test_request(tools=DUMMY_TOOLS),
        enable_auto_tools=True,
        model_output_token_ids=[tid for tid, _ in tokens],
    )
    assert output.reasoning == (reasoning or "") == ("Plan." if thinking else "")
    assert output.content.strip() == (content or "").strip() == "Checking."
    assert [call["name"] for call in output.tool_calls] == ["get_weather", "add"]
    assert [call.name for call in calls] == ["get_weather", "add"]
    assert (
        [json.loads(call["arguments"]) for call in output.tool_calls]
        == [json.loads(call.arguments) for call in calls]
        == [{"city": "杭州", "count": 42}, {"x": 1.5, "y": 2.25}]
    )


def test_v4_wrapper_name_ends_reasoning():
    text = "Plan." + V4_NAMED_CALLS
    tokenizer, tokens = tokenizer_for(text, False)
    output = collect_output(
        replay_streaming(
            parser_for(tokenizer, {"thinking": True}),
            tokens,
            chunk_size=1,
            finished_on_last=True,
            tools=DUMMY_TOOLS,
            prompt_token_ids=[50],
        )
    )
    assert output.reasoning == "Plan."
    assert not output.content.strip()
    assert [call["name"] for call in output.tool_calls] == ["get_weather", "add"]


@pytest.mark.parametrize(
    ("controls", "text", "expected_reasoning", "reasoning_tokens"),
    [
        ({"thinking": False}, "12", "", 0),
        ({"enable_thinking": False}, "12", "", 0),
        ({"thinking": True, "reasoning_effort": "none"}, "12", "", 0),
        ({}, "</think>12", "", 0),
        ({}, "Plan.</think>12", "Plan.", 5),
    ],
)
def test_reasoning_adapter_controls_and_usage(
    controls,
    text,
    expected_reasoning,
    reasoning_tokens,
):
    tokenizer, tokens = tokenizer_for(text, False)
    parser = parser_for(tokenizer, controls)
    output = collect_output(
        replay_streaming(
            parser,
            tokens,
            chunk_size=1,
            finished_on_last=True,
        )
    )
    assert output.reasoning == expected_reasoning
    assert output.content == "12"
    assert (
        parser.reasoning_parser.count_reasoning_tokens([tid for tid, _ in tokens])
        == reasoning_tokens
    )
    assert parser.is_reasoning_end([50, 100, 51])
    assert not parser.is_reasoning_end([51, 100, 50])


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    ("prompt_token_ids", "text", "expected_reasoning", "expected_content"),
    [
        # continue_final_message: the current turn already closed reasoning.
        ([54, 100, 55, 50, 101, 51, 102], "12", "", "12"),
        # A closed block in an earlier turn does not end this turn's reasoning.
        (
            [54, 100, 55, 50, 101, 51, 102, 54, 103, 55, 50],
            "Plan.</think>12",
            "Plan.",
            "12",
        ),
    ],
)
def test_prompt_closed_reasoning_is_scoped_to_current_turn(
    streaming,
    prompt_token_ids,
    text,
    expected_reasoning,
    expected_content,
):
    tokenizer, tokens = tokenizer_for(text, False)
    parser = parser_for(tokenizer, {"thinking": True})
    output_ids = [tid for tid, _ in tokens]
    if streaming:
        output = collect_output(
            replay_streaming(
                parser,
                tokens,
                chunk_size=1,
                finished_on_last=True,
                prompt_token_ids=prompt_token_ids,
            )
        )
        reasoning, content = output.reasoning, output.content
    else:
        parser.set_prompt_token_ids(prompt_token_ids)
        reasoning, content, _ = parser.parse(
            text, _test_request(), model_output_token_ids=output_ids
        )
    assert (reasoning or "") == expected_reasoning
    assert content == expected_content
    assert (parser.reasoning_parser.count_reasoning_tokens(output_ids) == 0) == (
        not expected_reasoning
    )


def test_python_argument_conversion_and_partial_values():
    converter = deepseek_v41_config().arg_converter
    assert converter is not None
    raw = (
        '<｜DSML｜ parameter name="object" string="false">'
        '{"a": [true, null]}</｜DSML｜ parameter>'
        '<｜DSML｜ parameter name="bad" string="false">[broken</｜DSML｜ parameter>'
        '<｜DSML｜ parameter name="bare">7</｜DSML｜ parameter>'
        '<｜DSML｜ parameter name="text" string="true">  a<b'
    )
    assert json.loads(converter(raw, True)) == {
        "object": {"a": [True, None]},
        "bad": "[broken",
        "bare": 7,
        "text": "  a<b",
    }


@pytest.mark.parametrize(
    "partial_tag", ['<｜DSML｜ parameter name="j', "<｜DSML｜ parameter na"]
)
def test_partial_tag_after_gt_is_held_back(partial_tag):
    config = deepseek_v41_config()
    assert config.arg_converter is not None
    assert config.arg_value_extender is not None
    raw = '<｜DSML｜ parameter name="k" string="true">a > b' + partial_tag
    assert json.loads(config.arg_converter(raw, True)) == {"k": "a > b"}
    assert config.arg_value_extender(raw) is None


def test_long_string_argument_streams_before_close():
    # Regression for #52846: a string body has no ">" so it used to be
    # buffered until the closing parameter tag arrived.
    body = "A" * 4096
    text = (
        '<｜DSML｜ calls>\n<｜DSML｜ invoke name="emit">\n'
        f'<｜DSML｜ parameter name="text" string="true">{body}'
        "</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>"
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "emit",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                },
            },
        }
    ]
    tokenizer, tokens = tokenizer_for(text, False)
    parser = parser_for(tokenizer, {"thinking": False})
    results = replay_streaming(
        parser, tokens, chunk_size=32, finished_on_last=True, tools=tools
    )
    arg_chunks = [
        tool_call.function.arguments
        for result in results
        if result
        for tool_call in result.tool_calls or []
        if tool_call.function and tool_call.function.arguments
    ]
    assert len(arg_chunks) > len(body) // 64
    assert json.loads("".join(arg_chunks)) == {"text": body}


# ── A DSML closer inside a value belongs to the value ────────────────

_WRITE_FILE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
            },
        },
    }
]


def _write_file(path, content, param_close="</｜DSML｜ parameter>"):
    return (
        '<｜DSML｜ invoke name="write_file">\n'
        f'<｜DSML｜ parameter name="path" string="true">{path}</｜DSML｜ parameter>\n'
        f'<｜DSML｜ parameter name="content" string="true">{content}{param_close}\n'
        "</｜DSML｜ invoke>\n"
    )


def _streamed_and_parsed_args(text, chunk_size, special_calls=False):
    tokenizer, tokens = tokenizer_for(text, special_calls)
    streamed = collect_output(
        replay_streaming(
            parser_for(tokenizer, {"thinking": False}),
            tokens,
            chunk_size=chunk_size,
            finished_on_last=True,
            tools=_WRITE_FILE_TOOLS,
        )
    )
    _, _, calls = parser_for(tokenizer, {"thinking": False}).parse(
        text,
        _test_request(tools=_WRITE_FILE_TOOLS),
        enable_auto_tools=True,
        model_output_token_ids=[tid for tid, _ in tokens],
    )
    return (
        [json.loads(call["arguments"]) for call in streamed.tool_calls],
        [json.loads(call.arguments) for call in calls],
    )


_CLOSERS = ["</｜DSML｜ parameter>", "</｜DSML｜ invoke>", "</｜DSML｜ calls>"]


@pytest.mark.parametrize(
    ("chunk_size", "special_calls"), [(1, False), (7, False), (10000, False), (1, True)]
)
@pytest.mark.parametrize("closer", _CLOSERS)
def test_closer_inside_value_is_kept(closer, chunk_size, special_calls):
    content = f"A block ends with {closer} on its own line."
    text = "<｜DSML｜ calls>\n" + _write_file("dsml.md", content) + "</｜DSML｜ calls>"
    expected = [{"path": "dsml.md", "content": content}]
    assert _streamed_and_parsed_args(text, chunk_size, special_calls) == (
        expected,
        expected,
    )


@pytest.mark.parametrize("chunk_size", [1, 10000])
@pytest.mark.parametrize("closer", _CLOSERS)
def test_value_ending_with_closer_is_kept(closer, chunk_size):
    content = f"The closer is {closer}"
    text = "<｜DSML｜ calls>\n" + _write_file("dsml.md", content) + "</｜DSML｜ calls>"
    expected = [{"path": "dsml.md", "content": content}]
    assert _streamed_and_parsed_args(text, chunk_size) == (expected, expected)


@pytest.mark.parametrize("chunk_size", [1, 10000])
def test_closer_inside_value_before_next_call(chunk_size):
    first = "see </｜DSML｜ invoke> and </｜DSML｜ calls> here"
    text = (
        "<｜DSML｜ calls>\n"
        + _write_file("one.md", first)
        + _write_file("two.txt", "second")
        + "</｜DSML｜ calls>"
    )
    expected = [
        {"path": "one.md", "content": first},
        {"path": "two.txt", "content": "second"},
    ]
    assert _streamed_and_parsed_args(text, chunk_size) == (expected, expected)


@pytest.mark.parametrize("chunk_size", [1, 10000])
@pytest.mark.parametrize("block_end", ["</｜DSML｜ calls>", ""])
def test_unclosed_value_ends_at_invoke_end(chunk_size, block_end):
    # The model omitted the value's closer: the call still ends at
    # </invoke> and keeps the value, newline included.
    text = "<｜DSML｜ calls>\n" + _write_file("a.txt", "hello", "") + block_end
    expected = [{"path": "a.txt", "content": "hello\n"}]
    assert _streamed_and_parsed_args(text, chunk_size) == (expected, expected)


def test_long_value_with_closer_still_streams():
    content = "A" * 2048 + " </｜DSML｜ invoke> " + "B" * 2048
    text = "<｜DSML｜ calls>\n" + _write_file("big.md", content) + "</｜DSML｜ calls>"
    tokenizer, tokens = tokenizer_for(text, False)
    results = replay_streaming(
        parser_for(tokenizer, {"thinking": False}),
        tokens,
        chunk_size=32,
        finished_on_last=True,
        tools=_WRITE_FILE_TOOLS,
    )
    arg_chunks = [
        tool_call.function.arguments
        for result in results
        if result
        for tool_call in result.tool_calls or []
        if tool_call.function and tool_call.function.arguments
    ]
    assert len(arg_chunks) > len(content) // 64
    assert json.loads("".join(arg_chunks)) == {"path": "big.md", "content": content}
