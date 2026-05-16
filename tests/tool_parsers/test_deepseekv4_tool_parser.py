# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for DeepSeekV4ToolParser."""

import json
from unittest.mock import MagicMock

from vllm.tool_parsers import ToolParserManager
from vllm.tool_parsers.deepseekv4_tool_parser import DeepSeekV4ToolParser

MOCK_TOKENIZER = MagicMock()
MOCK_TOKENIZER.get_vocab.return_value = {}

TC_START = "<｜DSML｜tool_calls>"
TC_END = "</｜DSML｜tool_calls>"
INV_START = '<｜DSML｜invoke name="'
INV_END = "</｜DSML｜invoke>"
PARAM_START = '<｜DSML｜parameter name="'
PARAM_END = "</｜DSML｜parameter>"


def make_parser(tools=None) -> DeepSeekV4ToolParser:
    return DeepSeekV4ToolParser(MOCK_TOKENIZER, tools=tools)


def make_request(tools=None) -> MagicMock:
    req = MagicMock()
    req.tools = tools
    return req


def build_tool_call(func_name: str, params: dict[str, str]) -> str:
    param_strs = "".join(
        f'{PARAM_START}{k}" string="true">{v}{PARAM_END}\n' for k, v in params.items()
    )
    return f'{TC_START}\n{INV_START}{func_name}">\n{param_strs}{INV_END}\n{TC_END}'


def stream(parser: DeepSeekV4ToolParser, full_text: str, chunk_size: int = 7):
    deltas = []
    previous_text = ""
    for start in range(0, len(full_text), chunk_size):
        delta_text = full_text[start : start + chunk_size]
        current_text = previous_text + delta_text
        delta = parser.extract_tool_calls_streaming(
            previous_text=previous_text,
            current_text=current_text,
            delta_text=delta_text,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=[1],
            request=make_request(),
        )
        previous_text = current_text
        if delta is not None:
            deltas.append(delta)
    return deltas


def reconstruct_args(deltas, tool_index: int = 0) -> str:
    fragments = []
    for delta in deltas:
        if delta.tool_calls:
            for tool_call in delta.tool_calls:
                if (
                    tool_call.index == tool_index
                    and tool_call.function
                    and tool_call.function.arguments
                ):
                    fragments.append(tool_call.function.arguments)
    return "".join(fragments)


def test_registered():
    assert ToolParserManager.get_tool_parser("deepseek_v4") is DeepSeekV4ToolParser


def test_extract_tool_calls():
    parser = make_parser()
    model_output = "Let me check. " + build_tool_call(
        "get_weather", {"location": "Beijing", "unit": "celsius"}
    )

    result = parser.extract_tool_calls(model_output, make_request())

    assert result.tools_called
    assert result.content == "Let me check. "
    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.function.name == "get_weather"
    assert json.loads(tool_call.function.arguments) == {
        "location": "Beijing",
        "unit": "celsius",
    }


def test_function_calls_block_is_not_accepted():
    parser = make_parser()
    model_output = build_tool_call("search", {"query": "vllm"}).replace(
        "tool_calls", "function_calls"
    )

    result = parser.extract_tool_calls(model_output, make_request())

    assert not result.tools_called
    assert result.content == model_output


def test_streaming_extracts_complete_invokes():
    parser = make_parser()
    full_text = build_tool_call("search", {"query": "deepseek v4"})

    deltas = stream(parser, full_text, chunk_size=5)

    names = [
        tool_call.function.name
        for delta in deltas
        if delta.tool_calls
        for tool_call in delta.tool_calls
    ]
    assert names == ["search"]
    assert json.loads(reconstruct_args(deltas)) == {"query": "deepseek v4"}
