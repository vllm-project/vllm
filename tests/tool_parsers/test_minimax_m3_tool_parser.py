# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from typing import Any

import pytest

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.tool_parsers import ToolParserManager
from vllm.tool_parsers.minimax_m3_tool_parser import MinimaxM3ToolParser

# MinimaxM3ToolParser extends RustToolParser; skip when the PyO3 extension
# is absent (mirrors the guard in test_rust_tool_parser.py).
pytest.importorskip("vllm._rust_tool_parser")

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]

NS = "]<]minimax[>["
EOS_ID = 99


class FakeTokenizer:
    """Minimal fake tokenizer for unit tests."""

    def __init__(self):
        self.model_tokenizer = True
        self.vocab: dict[str, int] = {}

    def get_vocab(self) -> dict[str, int]:
        return self.vocab


def sample_tools() -> list[ChatCompletionToolsParam]:
    return [
        ChatCompletionToolsParam(
            function=FunctionDefinition(
                name="create_order",
                parameters={
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "integer"},
                        "urgent": {"type": "boolean"},
                        "note": {"type": "string"},
                        "shipping": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                                "zip": {"type": "integer"},
                            },
                        },
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "sku": {"type": "string"},
                                    "qty": {"type": "integer"},
                                },
                            },
                        },
                        "metadata": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                        },
                        "duplicate_demo": {"type": "object"},
                    },
                },
            ),
        )
    ]


@pytest.fixture
def parser() -> MinimaxM3ToolParser:
    return MinimaxM3ToolParser(FakeTokenizer(), tools=sample_tools())


def build_order_call() -> str:
    return (
        f"{NS}<tool_call>\n"
        f'{NS}<invoke name="create_order">'
        f"{NS}<user_id>42{NS}</user_id>"
        f"{NS}<urgent>true{NS}</urgent>"
        f"{NS}<note>Please leave at front desk.{NS}</note>"
        f"{NS}<shipping>"
        f"{NS}<city>Singapore{NS}</city>"
        f"{NS}<zip>018956{NS}</zip>"
        f"{NS}</shipping>"
        f"{NS}<items>"
        f"{NS}<item>{NS}<sku>book-001{NS}</sku>{NS}<qty>2{NS}</qty>{NS}</item>"
        f"{NS}<item>{NS}<sku>pen-007{NS}</sku>{NS}<qty>5{NS}</qty>{NS}</item>"
        f"{NS}</items>"
        f"{NS}<metadata>"
        f"{NS}<source>mobile{NS}</source>"
        f"{NS}<campaign>may-launch{NS}</campaign>"
        f"{NS}</metadata>"
        f"{NS}<duplicate_demo>"
        f"{NS}<tag>a{NS}</tag>"
        f"{NS}<tag>b{NS}</tag>"
        f"{NS}</duplicate_demo>"
        f"{NS}</invoke>\n"
        f"{NS}</tool_call>"
    )


def build_order_invocation(user_id: int) -> str:
    return (
        f'{NS}<invoke name="create_order">'
        f"{NS}<user_id>{user_id}{NS}</user_id>"
        f"{NS}</invoke>"
    )


def build_multiple_order_call() -> str:
    return (
        f"{NS}<tool_call>\n"
        f"{build_order_invocation(1)}\n"
        f"{build_order_invocation(2)}\n"
        f"{NS}</tool_call>"
    )


def _feed(
    parser: MinimaxM3ToolParser, chunks: list[str | tuple[str, list[int]]]
) -> list[DeltaMessage]:
    previous = ""
    results: list[DeltaMessage] = []
    for chunk in chunks:
        if isinstance(chunk, tuple):
            delta, delta_ids = chunk
        else:
            delta = chunk
            delta_ids = []

        current = previous + delta
        result = parser.extract_tool_calls_streaming(
            previous_text=previous,
            current_text=current,
            delta_text=delta,
            previous_token_ids=[],
            current_token_ids=[],
            delta_token_ids=delta_ids,
            request=None,
        )
        if result is not None:
            results.append(result)
        previous = current
    return results


def _collect_content(results: list[DeltaMessage]) -> str:
    return "".join(result.content for result in results if result.content)


def _collect_tool_calls(results: list[DeltaMessage]) -> dict[int, dict[str, Any]]:
    tool_calls: dict[int, dict[str, Any]] = {}
    for result in results:
        for tool_call in result.tool_calls or []:
            tool_calls.setdefault(
                tool_call.index,
                {"id": None, "name": "", "arguments": ""},
            )
            if tool_call.id:
                tool_calls[tool_call.index]["id"] = tool_call.id
            if tool_call.function:
                if tool_call.function.name:
                    tool_calls[tool_call.index]["name"] += tool_call.function.name
                if tool_call.function.arguments:
                    tool_calls[tool_call.index]["arguments"] += (
                        tool_call.function.arguments
                    )
    return tool_calls


def test_minimax_m3_parser_registered():
    assert ToolParserManager.get_tool_parser("minimax_m3") is MinimaxM3ToolParser


def test_non_streaming_nested_tool_call(parser):
    result = parser.extract_tool_calls(
        "I will create it.\n" + build_order_call(),
        request=None,
    )

    assert result.tools_called
    assert result.content == "I will create it.\n"
    assert len(result.tool_calls) == 1
    tool_call = result.tool_calls[0]
    assert tool_call.function.name == "create_order"
    assert json.loads(tool_call.function.arguments) == {
        "user_id": 42,
        "urgent": True,
        "note": "Please leave at front desk.",
        "shipping": {"city": "Singapore", "zip": 18956},
        "items": [
            {"sku": "book-001", "qty": 2},
            {"sku": "pen-007", "qty": 5},
        ],
        "metadata": {
            "source": "mobile",
            "campaign": "may-launch",
        },
        "duplicate_demo": {"tag": ["a", "b"]},
    }


def test_non_streaming_without_tool_call_keeps_content(parser):
    result = parser.extract_tool_calls("plain response", request=None)

    assert not result.tools_called
    assert result.tool_calls == []
    assert result.content == "plain response"


def test_non_streaming_multiple_tool_calls(parser):
    result = parser.extract_tool_calls(build_multiple_order_call(), request=None)

    assert result.tools_called
    assert result.content is None
    assert [tool_call.function.name for tool_call in result.tool_calls] == [
        "create_order",
        "create_order",
    ]
    assert [
        json.loads(tool_call.function.arguments)["user_id"]
        for tool_call in result.tool_calls
    ] == [1, 2]


def test_streaming_without_tool_call_emits_text(parser):
    results = _feed(parser, ["plain ", "response"])

    assert _collect_content(results) == "plain response"
    assert _collect_tool_calls(results) == {}


def test_streaming_nested_tool_call(parser):
    tool_call_text = build_order_call()
    results = _feed(
        parser,
        [
            "I will create it.\n",
            tool_call_text[:5],
            tool_call_text[5:17],
            tool_call_text[17:120],
            tool_call_text[120:],
            ("", [EOS_ID]),
        ],
    )

    assert _collect_content(results) == "I will create it.\n"
    tool_calls = _collect_tool_calls(results)
    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "create_order"
    assert tool_calls[0]["id"] is not None
    assert json.loads(tool_calls[0]["arguments"]) == json.loads(
        parser.streamed_args_for_tool[0]
    )
    assert json.loads(parser.prev_tool_call_arr[0]["arguments"])["items"][1]["qty"] == 5
    assert results[-1].content is None
