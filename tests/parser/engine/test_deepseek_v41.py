# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
from openai.types.responses import CustomTool

from tests.parser.engine.replay_harness import (
    DUMMY_TOOLS,
    MockTokenizer,
    _test_request,
    collect_output,
    replay_streaming,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.exceptions import GenerationError
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


def custom_grammar_call(model: str, payload: str) -> str:
    if model == "deepseek_v4":
        begin = (
            '\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name="emit">\n'
            '<｜DSML｜parameter name="input" string="true">'
        )
        end = "</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
    else:
        begin = (
            '\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="emit">\n'
            '<｜DSML｜ parameter name="input" string="true">'
        )
        end = "</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>"
    return begin + payload + end


@pytest.mark.parametrize("model", ["deepseek_v4", "deepseek_v41"])
@pytest.mark.parametrize("chunk_size", [1, 7, 10000])
@pytest.mark.parametrize("reasoning", [False, True])
def test_custom_grammar_input_is_data_in_full_and_streamed_calls(
    model, chunk_size, reasoning
):
    payload = "\n你好😀\n" + (
        '</｜DSML｜parameter>\n</｜DSML｜invoke>\n<｜DSML｜invoke name="other">\n'
        if model == "deepseek_v4"
        else "</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n"
        '<｜DSML｜ invoke name="other">\n'
    )
    request = ResponsesRequest(
        input="Call emit.",
        tools=[
            CustomTool(
                type="custom",
                name="emit",
                format={
                    "type": "grammar",
                    "syntax": "lark",
                    "definition": "start: /[\\s\\S]*/",
                },
            )
        ],
    )
    parser_cls = ParserManager.get_parser(
        tool_parser_name=model,
        reasoning_parser_name=model if reasoning else None,
        enable_auto_tools=True,
    )
    assert parser_cls is not None
    text = ("<think>Plan.</think>" if reasoning else "") + custom_grammar_call(
        model, payload
    )
    tokenizer, _ = tokenizer_for(text, reasoning)

    parser = parser_cls(tokenizer, request.tools)
    parser.adjust_request(request)
    parsed_reasoning, _, calls = parser.parse(text, request, enable_auto_tools=True)
    assert calls is not None and len(calls) == 1
    assert calls[0].name == "emit"
    assert parsed_reasoning == ("Plan." if reasoning else None)
    assert json.loads(calls[0].arguments) == {"input": payload}

    parser = parser_cls(tokenizer, request.tools)
    deltas = []
    for start in range(0, len(text), chunk_size):
        deltas.append(
            parser.parse_delta(
                text[start : start + chunk_size],
                [],
                request,
                prompt_token_ids=([50] if reasoning and start == 0 else None)
                if reasoning
                else [],
                finished=start + chunk_size >= len(text),
            )
        )
    output = collect_output(deltas)
    assert len(output.tool_calls) == 1
    assert output.tool_calls[0]["name"] == "emit"
    assert json.loads(output.tool_calls[0]["arguments"]) == {"input": payload}


@pytest.mark.parametrize("model", ["deepseek_v4", "deepseek_v41"])
@pytest.mark.parametrize("chunk_size", [1, 10000])
def test_truncated_custom_grammar_input_does_not_parse_as_tools(model, chunk_size):
    payload = (
        '\n</｜DSML｜parameter>\n</｜DSML｜ invoke>\n<｜DSML｜invoke name="other">\n'
    )
    request = ResponsesRequest(
        input="Call emit.",
        tools=[
            CustomTool(
                type="custom",
                name="emit",
                format={
                    "type": "grammar",
                    "syntax": "lark",
                    "definition": "start: /[\\s\\S]*/",
                },
            )
        ],
    )
    parser_cls = ParserManager.get_parser(
        tool_parser_name=model,
        reasoning_parser_name=None,
        enable_auto_tools=True,
    )
    assert parser_cls is not None
    tool_end = (
        "</｜DSML｜tool_calls>" if model == "deepseek_v4" else "</｜DSML｜ calls>"
    )
    text = custom_grammar_call(model, payload).removesuffix(tool_end)
    tokenizer, _ = tokenizer_for(text, False)

    parser = parser_cls(tokenizer, request.tools)
    parser.adjust_request(request)
    _, _, calls = parser.parse(text, request, enable_auto_tools=True)
    assert not calls

    parser = parser_cls(tokenizer, request.tools)
    deltas = []
    for start in range(0, len(text), chunk_size):
        deltas.append(
            parser.parse_delta(
                text[start : start + chunk_size],
                [],
                request,
                prompt_token_ids=[] if start == 0 else None,
                finished=start + chunk_size >= len(text),
            )
        )
    assert not collect_output(deltas).tool_calls


def test_custom_grammar_validation_is_disabled_in_unguarded_fallback():
    tool = CustomTool(
        type="custom",
        name="emit",
        format={"type": "grammar", "syntax": "lark", "definition": 'start: "ok"'},
    )
    request = ResponsesRequest(input="Call emit.", tools=[tool])
    parser_cls = ParserManager.get_parser(
        tool_parser_name="deepseek_v41",
        reasoning_parser_name=None,
        enable_auto_tools=True,
    )
    assert parser_cls is not None
    text = custom_grammar_call("deepseek_v41", "not-ok")
    tokenizer, _ = tokenizer_for(text, False)

    _, _, calls = parser_cls(tokenizer, request.tools).parse(
        text, request, enable_auto_tools=True
    )

    assert calls is not None and len(calls) == 1
    assert json.loads(calls[0].arguments) == {"input": "not-ok"}


def test_unguarded_fallback_keeps_parallel_tool_calls_separate():
    custom = CustomTool(
        type="custom",
        name="emit",
        format={"type": "grammar", "syntax": "lark", "definition": 'start: "ok"'},
    )
    function = {
        "type": "function",
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
    request = ResponsesRequest.model_validate(
        {
            "input": "Call both.",
            "tools": [custom, function],
            "_custom_tool_grammar_enforced": True,
        }
    )
    assert request._custom_tool_grammar_enforced is False
    parser_cls = ParserManager.get_parser(
        tool_parser_name="deepseek_v41",
        reasoning_parser_name=None,
        enable_auto_tools=True,
    )
    assert parser_cls is not None
    text = (
        "\n\n<｜DSML｜ calls>\n"
        '<｜DSML｜ invoke name="emit">\n'
        '<｜DSML｜ parameter name="input" string="true">not-ok'
        "</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n"
        '<｜DSML｜ invoke name="get_weather">\n'
        '<｜DSML｜ parameter name="city" string="true">Paris'
        "</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n"
        "</｜DSML｜ calls>"
    )
    tokenizer, _ = tokenizer_for(text, False)

    _, _, calls = parser_cls(tokenizer, request.tools).parse(
        text, request, enable_auto_tools=True
    )

    assert calls is not None and [call.name for call in calls] == [
        "emit",
        "get_weather",
    ]


def test_native_custom_grammar_validation_rejects_mismatched_payload():
    tool = CustomTool(
        type="custom",
        name="emit",
        format={"type": "grammar", "syntax": "lark", "definition": 'start: "ok"'},
    )
    request = ResponsesRequest(input="Call emit.", tools=[tool])
    parser_cls = ParserManager.get_parser(
        tool_parser_name="deepseek_v41",
        reasoning_parser_name=None,
        enable_auto_tools=True,
    )
    assert parser_cls is not None
    text = custom_grammar_call("deepseek_v41", "not-ok")
    tokenizer, _ = tokenizer_for(text, False)
    parser = parser_cls(tokenizer, request.tools)
    parser.adjust_request(request)

    with pytest.raises(GenerationError):
        parser.parse(text, request, enable_auto_tools=True)


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
