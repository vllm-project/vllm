# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from unittest.mock import MagicMock

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.reasoning.deepseek_v4_reasoning_parser import DeepSeekV4ReasoningParser
from vllm.tool_parsers.deepseekv4_tool_parser import DeepSeekV4ToolParser

TC_START = "<｜DSML｜tool_calls>"
TC_END = "</｜DSML｜tool_calls>"
DSML_TOOL_CALL = (
    f"{TC_START}\n"
    '<｜DSML｜invoke name="read_file">\n'
    '<｜DSML｜parameter name="path" string="true">/tmp/example.lean'
    "</｜DSML｜parameter>\n"
    "</｜DSML｜invoke>\n"
    f"{TC_END}"
)


class FakeDeepSeekV4Tokenizer:
    vocab = {"<think>": 1, "</think>": 2}

    def get_vocab(self) -> dict[str, int]:
        return self.vocab


def _request() -> ChatCompletionRequest:
    return ChatCompletionRequest(model="test-model", messages=[])


def _parser() -> DeepSeekV4ReasoningParser:
    return DeepSeekV4ReasoningParser(
        FakeDeepSeekV4Tokenizer(),
        chat_template_kwargs={"thinking": True},
    )


def test_tool_call_start_implicitly_ends_reasoning() -> None:
    reasoning, content = _parser().extract_reasoning(
        f"Let me search.\n\n{DSML_TOOL_CALL}",
        _request(),
    )

    assert reasoning == "Let me search.\n\n"
    assert content == DSML_TOOL_CALL


def test_implicit_content_is_parseable_as_tool_call() -> None:
    tool_parser = DeepSeekV4ToolParser(MagicMock(), tools=None)
    _, content = _parser().extract_reasoning(
        f"Let me search.\n\n{DSML_TOOL_CALL}",
        _request(),
    )

    result = tool_parser.extract_tool_calls(content or "", MagicMock())

    assert result.tools_called
    assert len(result.tool_calls) == 1
    parsed_tool_call = result.tool_calls[0]
    assert parsed_tool_call.function.name == "read_file"
    assert json.loads(parsed_tool_call.function.arguments) == {
        "path": "/tmp/example.lean",
    }
