# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming HYV4 tool calls whose argument keys repeat."""

import json

import pytest

from tests.tool_parsers.utils import (
    run_tool_extraction,
    run_tool_extraction_streaming,
)
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.tool_parsers.hy_v4_tool_parser import HYV4ToolParser

STRUCTURAL_TOKENS = [
    "<tool_calls>",
    "</tool_calls>",
    "<tool_call>",
    "</tool_call>",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
]


class _StructuralTokenizer:
    """Only what the parser and test helpers read: the structural vocab."""

    init_kwargs: dict = {}

    def get_vocab(self) -> dict[str, int]:
        return {token: i for i, token in enumerate(STRUCTURAL_TOKENS)}

    def tokenize(self, text: str) -> list[str]:
        return _per_token_deltas(text)


REQUEST = ChatCompletionRequest(
    messages=[],
    model="test-model",
    tools=[
        ChatCompletionToolsParam(
            type="function",
            function=FunctionDefinition(
                name="get_weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "integer"},
                    },
                },
            ),
        )
    ],
)

# Two calls in one output, so the second call runs on reset streaming state.
MODEL_OUTPUT = (
    "<tool_calls>"
    "<tool_call>get_weather"
    "<arg_key>city</arg_key><arg_value>Paris</arg_value>"
    "<arg_key>city</arg_key><arg_value>Berlin</arg_value>"
    "</tool_call>"
    "<tool_call>get_weather"
    "<arg_key>days</arg_key><arg_value>1</arg_value>"
    "<arg_key>days</arg_key><arg_value>2</arg_value>"
    "</tool_call>"
    "</tool_calls>"
)


def _per_token_deltas(text: str) -> list[str]:
    """Structural tokens stay atomic; everything else is one char per delta."""
    deltas = []
    i = 0
    while i < len(text):
        token = next((t for t in STRUCTURAL_TOKENS if text.startswith(t, i)), text[i])
        deltas.append(token)
        i += len(token)
    return deltas


@pytest.mark.parametrize(
    "deltas",
    [_per_token_deltas(MODEL_OUTPUT), [MODEL_OUTPUT]],
    ids=["per_token", "one_delta"],
)
def test_streaming_duplicate_arg_key_matches_non_streaming(deltas):
    _, expected = run_tool_extraction(
        HYV4ToolParser(_StructuralTokenizer()), MODEL_OUTPUT, REQUEST
    )
    streamed = run_tool_extraction_streaming(
        HYV4ToolParser(_StructuralTokenizer()),
        deltas,
        REQUEST,
        assert_one_tool_per_delta=False,
    ).tool_calls

    assert len(expected) == 2
    assert [call.function.name for call in streamed] == ["get_weather"] * 2
    for streamed_call, expected_call in zip(streamed, expected):
        # Must be valid JSON; the later value wins, as in non-streaming.
        assert json.loads(streamed_call.function.arguments) == json.loads(
            expected_call.function.arguments
        )
        if len(deltas) == 1:
            # Nothing was sent before the calls closed, so keys are deduplicated.
            assert streamed_call.function.arguments == expected_call.function.arguments
