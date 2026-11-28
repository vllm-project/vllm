# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from tests.parser.engine.replay_harness import (
    MockTokenizer,
    _test_request,
    collect_output,
    replay_streaming,
)
from vllm.parser.deepseek_v4 import DeepSeekV4Parser, _dsml_arg_converter
from vllm.parser.deepseek_v32 import DeepSeekV32Parser
from vllm.parser.deepseek_v41 import DeepSeekV41Parser


def _reject_constant(value):
    pytest.fail(f"Tool arguments contain a non-JSON constant: {value}")


def _parse_arguments(parser_cls, params, properties, chunk_size):
    tokenizer = MockTokenizer({"<think>": 50, "</think>": 51}, [])
    parser = parser_cls(tokenizer, chat_template_kwargs={"thinking": False})
    terminals = parser.parser_engine_config.terminals
    body = "".join(
        f'{terminals["PARAM_START"]} name="{name}"{attr}>{value}'
        f"{terminals['PARAM_CLOSE']}"
        for name, attr, value in params
    )
    opener = terminals["TOOL_START"]
    if isinstance(opener, tuple):
        opener = opener[0]
    text = (
        f"{opener}{terminals['INVOKE_PREFIX']}emit"
        f"{terminals['INVOKE_NAME_END']}{body}"
        f"{terminals['INVOKE_END']}{terminals['TOOL_END']}"
    )
    tokens = [(100 + i, char) for i, char in enumerate(text)]
    tokenizer = MockTokenizer({"<think>": 50, "</think>": 51}, tokens)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "emit",
                "parameters": {"type": "object", "properties": properties},
            },
        }
    ]
    request = _test_request(tools)
    parser = parser_cls(
        tokenizer, request.tools, chat_template_kwargs={"thinking": False}
    )
    if chunk_size == 0:
        _, _, calls = parser.parse(text, request, enable_auto_tools=True)
        assert len(calls) == 1
        assert calls[0].name == "emit"
        arguments = calls[0].arguments
    else:
        output = collect_output(
            replay_streaming(
                parser,
                tokens,
                chunk_size=chunk_size,
                finished_on_last=True,
                tools=tools,
            )
        )
        assert len(output.tool_calls) == 1
        assert output.tool_calls[0]["name"] == "emit"
        arguments = output.tool_calls[0]["arguments"]
    return json.loads(arguments, parse_constant=_reject_constant)


@pytest.mark.parametrize(
    "parser_cls", [DeepSeekV4Parser, DeepSeekV32Parser, DeepSeekV41Parser]
)
@pytest.mark.parametrize(
    "chunk_size", [0, 1, 7, None], ids=["batch", "char", "seven", "whole"]
)
@pytest.mark.parametrize("attr", [' string="false"', ""])
def test_nonfinite_parameters_preserve_literals(parser_cls, chunk_size, attr):
    cases = [
        ("overflow", "1e999", "1e999", "number"),
        ("negative", "-1e999", "-1e999", "number"),
        ("nan", "NaN", "NaN", "number"),
        ("array", "[1e999]", "[1e999]", "array"),
        ("object", '{"label":">","x":NaN}', '{"label":">","x":NaN}', "object"),
        ("finite", "1.5", 1.5, "number"),
        (
            "large",
            "123456789012345678901234567890",
            123456789012345678901234567890,
            "integer",
        ),
        ("quoted", '"Infinity"', "Infinity", "string"),
    ]
    result = _parse_arguments(
        parser_cls,
        [(name, attr, value) for name, value, _, _ in cases],
        {name: {"type": kind} for name, _, _, kind in cases},
        chunk_size,
    )
    assert result == {name: expected for name, _, expected, _ in cases}


@pytest.mark.parametrize("attr", [' string="false"', ' string="true"', ""])
@pytest.mark.parametrize(
    "chunk_size", [0, 1, 7, None], ids=["batch", "char", "seven", "whole"]
)
def test_string_wrapper_does_not_reintroduce_nonfinite_json(attr, chunk_size):
    value = '{"x":[1e999]}'
    result = _parse_arguments(
        DeepSeekV4Parser,
        [("arguments", attr, value)],
        {"x": {"type": "array"}},
        chunk_size,
    )
    assert result == {"arguments": value}


@pytest.mark.parametrize("attr", [' string="false"', ""])
def test_partial_overflow_waits_for_complete_parameter(attr):
    raw = f'<｜DSML｜parameter name="value"{attr}>1e99'
    assert json.loads(_dsml_arg_converter(raw, partial=True)) == {"value": 1e99}
    raw += "9"
    assert json.loads(_dsml_arg_converter(raw, partial=True)) == {}
    raw += "</｜DSML｜parameter>"
    assert json.loads(_dsml_arg_converter(raw, partial=True)) == {"value": "1e999"}
